"""Fast Trading Pipeline for Scalping/HFT

Optimized pipeline for high-frequency and scalping strategies:
    - Minimal overhead (skips slow components like sentiment/news)
    - Parallel processing for multiple stocks
    - Shorter timeframes (1Min, 5Min)
    - Tighter stops and profit targets
    - Pre-market scan with trade queue

Usage:
    # Fast mode - quick execution
    pipeline = FastPipeline(dry_run=True)
    results = pipeline.run_fast(["AAPL", "TSLA"], timeframe="1Min")

    # Pre-market scan and queue
    queue = pipeline.premarket_scan(["AAPL", "TSLA"])
    pipeline.execute_queue_at_open(queue)
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from utils.alpaca_client import AlpacaClient
from utils.position_sizer import PositionSizer
from utils.market_regime import MarketRegimeDetector
from utils.scalping_indicators import ScalpingIndicators, ScalpSignal

from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.risk_management_agent import RiskManagementAgent
from agents.execution_agent import ExecutionAgent

logger = logging.getLogger(__name__)


@dataclass
class QueuedTrade:
    """Trade queued for execution at market open."""
    symbol: str
    signal: str  # BUY or SELL
    shares: int
    entry_price: float  # Expected price
    stop_loss: float
    take_profit: float
    score: float
    queued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    executed: bool = False
    order_id: str = ""
    execution_price: float = 0.0
    status: str = "pending"  # pending, executed, cancelled, failed


@dataclass
class FastResult:
    """Lightweight result from fast pipeline."""
    symbol: str
    signal: str = "HOLD"
    score: float = 0.0
    price: float = 0.0
    shares: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    executed: bool = False
    order_id: str = ""
    status: str = ""
    error: str = ""
    latency_ms: float = 0.0  # Processing time


class FastPipeline:
    """Optimized pipeline for scalping and high-frequency trading."""

    def __init__(
        self,
        *,
        client: AlpacaClient | None = None,
        dry_run: bool = False,
        # Scalping parameters
        max_workers: int = 8,  # Parallel threads
        atr_multiplier_sl: float = 1.0,  # Tighter stop loss (1x ATR)
        atr_multiplier_tp: float = 1.5,  # Smaller profit target (1.5x ATR)
        max_position_pct: float = 0.02,  # 2% max per position (smaller)
        min_score: float = 0.1,  # Minimum score to trade
        skip_regime: bool = True,  # Skip regime for speed
        skip_correlation: bool = True,  # Skip correlation for speed
        # Scalping-specific settings
        use_scalp_indicators: bool = True,  # Use VWAP/momentum indicators
        profit_target_pct: float = 0.5,  # 0.5% profit target
        stop_loss_pct: float = 0.25,  # 0.25% stop loss
        vwap_entry_std: float = 1.5,  # Enter at 1.5 std from VWAP
        volume_spike_mult: float = 1.5,  # Volume must be 1.5x avg
    ):
        self.client = client
        self.dry_run = dry_run
        self.max_workers = max_workers
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        self.max_position_pct = max_position_pct
        self.min_score = min_score
        self.skip_regime = skip_regime
        self.skip_correlation = skip_correlation
        self.use_scalp_indicators = use_scalp_indicators
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.vwap_entry_std = vwap_entry_std
        self.volume_spike_mult = volume_spike_mult

        # Lazy-loaded components
        self._ta_agent = None
        self._scalp_indicators = None
        self._risk_agent = None
        self._exec_agent = None
        self._sizer = None
        self._regime_detector = None

        # Trade queue for pre-market
        self._trade_queue: list[QueuedTrade] = []
        self._queue_lock = threading.Lock()

    def _ensure_client(self):
        """Lazy load Alpaca client."""
        if self.client is None:
            self.client = AlpacaClient()
        return self.client

    def _ensure_ta(self):
        """Lazy load TA agent."""
        if self._ta_agent is None:
            self._ta_agent = TechnicalAnalysisAgent(
                client=self._ensure_client(),
                auto_trade=False,
            )
        return self._ta_agent

    def _ensure_scalp_indicators(self):
        """Lazy load scalping indicators."""
        if self._scalp_indicators is None:
            self._scalp_indicators = ScalpingIndicators(
                vwap_std_entry=self.vwap_entry_std,
                profit_target_pct=self.profit_target_pct,
                stop_loss_pct=self.stop_loss_pct,
                volume_spike_mult=self.volume_spike_mult,
            )
        return self._scalp_indicators

    def _ensure_sizer(self):
        """Lazy load position sizer with scalping settings."""
        if self._sizer is None:
            self._sizer = PositionSizer(
                kelly_factor=0.25,  # Reduced Kelly for faster trades
                max_position_pct=self.max_position_pct,
                max_portfolio_heat=0.06,  # Lower heat for more positions
                atr_risk_mult=self.atr_multiplier_sl,
                take_profit_ratio=self.atr_multiplier_tp / self.atr_multiplier_sl,
            )
        return self._sizer

    def _ensure_risk(self):
        """Lazy load risk agent."""
        if self._risk_agent is None:
            self._risk_agent = RiskManagementAgent(
                client=self._ensure_client(),
                auto_trade=False,
            )
        return self._risk_agent

    def _ensure_exec(self):
        """Lazy load execution agent."""
        if self._exec_agent is None:
            self._exec_agent = ExecutionAgent(
                client=self._ensure_client(),
                dry_run=self.dry_run,
                skip_market_check=True,
            )
        return self._exec_agent

    def _analyze_single(
        self,
        symbol: str,
        timeframe: str,
        equity: float,
    ) -> FastResult:
        """Analyze single symbol with minimal overhead."""
        start = datetime.now(timezone.utc)
        result = FastResult(symbol=symbol)

        try:
            # Get price data
            client = self._ensure_client()
            df = client.get_bars(symbol, timeframe=timeframe, limit=100)

            if df.empty or len(df) < 30:
                result.error = "Insufficient data"
                result.latency_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                return result

            result.price = float(df["close"].iloc[-1])

            # Use scalping indicators for fast signals
            if self.use_scalp_indicators:
                scalp = self._ensure_scalp_indicators()
                scalp_signal = scalp.analyze(df, symbol=symbol)

                result.signal = scalp_signal.signal
                result.score = scalp_signal.strength
                result.stop_loss = scalp_signal.stop_loss
                result.take_profit = scalp_signal.target_price

                if result.signal == "HOLD":
                    result.latency_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                    return result

                # Calculate shares based on fixed % of equity for scalping
                risk_amount = equity * (self.stop_loss_pct / 100)
                price_risk = abs(result.price - result.stop_loss)
                if price_risk > 0:
                    result.shares = int(risk_amount / price_risk)
                    # Cap at max position size
                    max_shares = int(equity * self.max_position_pct / result.price)
                    result.shares = min(result.shares, max_shares)
                else:
                    result.shares = 0

            else:
                # Fall back to regular TA analysis
                ta = self._ensure_ta()
                ta_result = ta.analyze(symbol, timeframe=timeframe)

                atr = ta_result.get("atr", 0)
                result.score = ta_result.get("composite_score", 0)

                # Determine signal
                if result.score >= self.min_score:
                    result.signal = "BUY"
                elif result.score <= -self.min_score:
                    result.signal = "SELL"
                else:
                    result.signal = "HOLD"
                    result.latency_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                    return result

                # Quick position sizing
                sizer = self._ensure_sizer()
                sizing = sizer.calculate(
                    symbol,
                    price=result.price,
                    equity=equity,
                    signal=result.signal,
                    atr=atr,
                )

                result.shares = sizing.shares
                result.stop_loss = sizing.stop_loss
                result.take_profit = sizing.take_profit

            if result.shares == 0:
                result.signal = "HOLD"
                result.error = "Zero shares calculated"

        except Exception as e:
            result.error = str(e)
            logger.error("Fast analysis failed for %s: %s", symbol, e)

        result.latency_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        return result

    def run_fast(
        self,
        symbols: list[str],
        *,
        timeframe: str = "1Min",
        execute: bool = True,
    ) -> list[FastResult]:
        """Run fast parallel analysis on multiple symbols.

        Args:
            symbols: List of stock tickers
            timeframe: Bar timeframe (default 1Min for scalping)
            execute: Whether to execute trades

        Returns:
            List of FastResult objects
        """
        self._ensure_client()
        start_time = datetime.now(timezone.utc)

        # Get account equity
        try:
            account = self.client.get_account()
            equity = account.get("equity", 0)
        except Exception as e:
            logger.error("Could not fetch account: %s", e)
            return [FastResult(symbol=s, error=f"Account error: {e}") for s in symbols]

        results: list[FastResult] = []

        # Parallel analysis
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._analyze_single, symbol, timeframe, equity): symbol
                for symbol in symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(FastResult(symbol=symbol, error=str(e)))

        # Execute trades if requested
        if execute:
            self._execute_trades(results)

        total_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Fast pipeline: %d symbols in %.0fms (avg %.0fms/symbol)",
            len(symbols), total_time, total_time / len(symbols) if symbols else 0
        )

        return results

    def _execute_trades(self, results: list[FastResult]) -> None:
        """Execute trades for actionable signals."""
        exec_agent = self._ensure_exec()

        for result in results:
            if result.signal not in ("BUY", "SELL") or result.shares == 0:
                continue

            try:
                decision = {
                    "symbol": result.symbol,
                    "signal": result.signal,
                    "position_size": result.shares,
                    "stop_loss": result.stop_loss,
                    "take_profit": result.take_profit,
                    "risk_approved": True,
                    "current_price": result.price,
                    "combined_score": result.score,
                }

                exec_analysis = exec_agent.analyze(result.symbol, decision=decision)
                exec_result = exec_agent.execute(result.symbol, exec_analysis)

                result.executed = exec_result.get("status") in ("filled", "dry_run", "pending")
                result.order_id = exec_result.get("order_id", "")
                result.status = exec_result.get("status", "")

            except Exception as e:
                result.error = f"Execution error: {e}"
                logger.error("Trade execution failed for %s: %s", result.symbol, e)

    # ── Pre-market Scanning ──────────────────────────────────────

    def premarket_scan(
        self,
        symbols: list[str],
        *,
        timeframe: str = "1Day",  # Use daily for pre-market analysis
    ) -> list[QueuedTrade]:
        """Scan stocks before market open and queue trades.

        This runs analysis using previous day's data to identify
        opportunities for market open execution.

        Args:
            symbols: List of stock tickers to scan
            timeframe: Timeframe for analysis (default 1Day)

        Returns:
            List of QueuedTrade objects ready for execution
        """
        self._ensure_client()
        logger.info("Pre-market scan: analyzing %d symbols", len(symbols))

        # Get account equity
        try:
            account = self.client.get_account()
            equity = account.get("equity", 0)
        except Exception as e:
            logger.error("Could not fetch account: %s", e)
            return []

        queued_trades: list[QueuedTrade] = []

        # Parallel analysis
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._analyze_single, symbol, timeframe, equity): symbol
                for symbol in symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()

                    if result.signal in ("BUY", "SELL") and result.shares > 0:
                        trade = QueuedTrade(
                            symbol=symbol,
                            signal=result.signal,
                            shares=result.shares,
                            entry_price=result.price,
                            stop_loss=result.stop_loss,
                            take_profit=result.take_profit,
                            score=result.score,
                        )
                        queued_trades.append(trade)
                        logger.info(
                            "Queued %s: %s %d @ ~$%.2f (score: %.3f)",
                            symbol, result.signal, result.shares, result.price, result.score
                        )

                except Exception as e:
                    logger.error("Pre-market analysis failed for %s: %s", symbol, e)

        # Sort by score (strongest signals first)
        queued_trades.sort(key=lambda x: abs(x.score), reverse=True)

        # Store in queue
        with self._queue_lock:
            self._trade_queue = queued_trades

        logger.info("Pre-market scan complete: %d trades queued", len(queued_trades))
        return queued_trades

    def get_queue(self) -> list[QueuedTrade]:
        """Get current trade queue."""
        with self._queue_lock:
            return list(self._trade_queue)

    def clear_queue(self) -> None:
        """Clear trade queue."""
        with self._queue_lock:
            self._trade_queue = []

    def execute_queue_at_open(
        self,
        queue: list[QueuedTrade] | None = None,
        max_trades: int = 5,
    ) -> list[QueuedTrade]:
        """Execute queued trades (call this at market open).

        Args:
            queue: Optional queue to execute (uses internal queue if None)
            max_trades: Maximum trades to execute

        Returns:
            List of executed QueuedTrade objects with results
        """
        if queue is None:
            with self._queue_lock:
                queue = self._trade_queue[:max_trades]

        if not queue:
            logger.info("No trades in queue to execute")
            return []

        self._ensure_client()
        exec_agent = self._ensure_exec()

        executed = []
        for trade in queue[:max_trades]:
            if trade.executed:
                continue

            try:
                # Get current price for execution
                bars = self.client.get_bars(trade.symbol, "1Min", limit=1)
                if bars:
                    current_price = bars[-1]["close"]
                else:
                    current_price = trade.entry_price

                decision = {
                    "symbol": trade.symbol,
                    "signal": trade.signal,
                    "position_size": trade.shares,
                    "stop_loss": trade.stop_loss,
                    "take_profit": trade.take_profit,
                    "risk_approved": True,
                    "current_price": current_price,
                    "combined_score": trade.score,
                }

                exec_analysis = exec_agent.analyze(trade.symbol, decision=decision)
                exec_result = exec_agent.execute(trade.symbol, exec_analysis)

                trade.executed = exec_result.get("status") in ("filled", "dry_run", "pending")
                trade.order_id = exec_result.get("order_id", "")
                trade.status = exec_result.get("status", "")
                trade.execution_price = exec_result.get("filled_avg_price", current_price)

                if trade.executed:
                    logger.info(
                        "Executed queued trade: %s %s %d @ $%.2f",
                        trade.signal, trade.symbol, trade.shares, trade.execution_price
                    )
                else:
                    trade.status = "failed"
                    logger.warning("Failed to execute: %s %s", trade.signal, trade.symbol)

                executed.append(trade)

            except Exception as e:
                trade.status = "failed"
                logger.error("Queue execution failed for %s: %s", trade.symbol, e)
                executed.append(trade)

        return executed

    def wait_and_execute_at_open(
        self,
        queue: list[QueuedTrade] | None = None,
        max_trades: int = 5,
        poll_interval: int = 5,
    ) -> list[QueuedTrade]:
        """Wait for market to open, then immediately execute queue.

        Args:
            queue: Optional queue to execute
            max_trades: Maximum trades to execute
            poll_interval: Seconds between market open checks

        Returns:
            List of executed trades
        """
        import time

        self._ensure_client()

        logger.info("Waiting for market to open...")

        while True:
            try:
                clock = self.client.api.get_clock()
                if clock.is_open:
                    logger.info("Market is OPEN - executing queued trades!")
                    return self.execute_queue_at_open(queue, max_trades)

                # Calculate wait time
                next_open = clock.next_open
                now = clock.timestamp
                wait_seconds = (next_open - now).total_seconds()

                if wait_seconds > 60:
                    hours = int(wait_seconds // 3600)
                    minutes = int((wait_seconds % 3600) // 60)
                    logger.info("Market opens in %dh %dm", hours, minutes)
                    time.sleep(min(60, wait_seconds))
                else:
                    # Close to open - poll frequently
                    time.sleep(poll_interval)

            except KeyboardInterrupt:
                logger.info("Wait cancelled by user")
                return []
            except Exception as e:
                logger.error("Error checking market status: %s", e)
                time.sleep(poll_interval)

    # ── Formatting ────────────────────────────────────────────────

    @staticmethod
    def format_results(results: list[FastResult]) -> str:
        """Format results as table."""
        if not results:
            return "No results."

        lines = [
            f"\n{'=' * 80}",
            f"  FAST PIPELINE RESULTS ({len(results)} stocks)",
            f"{'=' * 80}",
            f"  {'Symbol':8s} {'Signal':6s} {'Score':>8s} {'Price':>10s} {'Shares':>7s} {'Status':>10s} {'Latency':>8s}",
            f"  {'-' * 76}",
        ]

        for r in results:
            lines.append(
                f"  {r.symbol:8s} {r.signal:6s} {r.score:>+8.4f} "
                f"${r.price:>9.2f} {r.shares:>7d} {r.status or 'N/A':>10s} "
                f"{r.latency_ms:>6.0f}ms"
            )

        executed = sum(1 for r in results if r.executed)
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

        lines.append(f"{'~' * 80}")
        lines.append(f"  Executed: {executed}/{len(results)}  |  Avg latency: {avg_latency:.0f}ms")
        lines.append(f"{'=' * 80}\n")

        return "\n".join(lines)

    @staticmethod
    def format_queue(queue: list[QueuedTrade]) -> str:
        """Format trade queue as table."""
        if not queue:
            return "Trade queue is empty."

        lines = [
            f"\n{'=' * 80}",
            f"  TRADE QUEUE ({len(queue)} trades pending)",
            f"{'=' * 80}",
            f"  {'Symbol':8s} {'Signal':6s} {'Shares':>7s} {'Price':>10s} {'SL':>10s} {'TP':>10s} {'Score':>8s}",
            f"  {'-' * 76}",
        ]

        for t in queue:
            lines.append(
                f"  {t.symbol:8s} {t.signal:6s} {t.shares:>7d} "
                f"${t.entry_price:>9.2f} ${t.stop_loss:>9.2f} ${t.take_profit:>9.2f} "
                f"{t.score:>+8.4f}"
            )

        lines.append(f"{'=' * 80}\n")
        return "\n".join(lines)


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fast Trading Pipeline")
    parser.add_argument("symbols", nargs="+", help="Stock symbols")
    parser.add_argument("--timeframe", default="1Min", help="Timeframe")
    parser.add_argument("--premarket", action="store_true", help="Pre-market scan mode")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run")
    parser.add_argument("--execute", action="store_true", help="Execute trades")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    pipeline = FastPipeline(dry_run=args.dry_run)

    if args.premarket:
        queue = pipeline.premarket_scan(args.symbols)
        print(FastPipeline.format_queue(queue))
        if args.execute:
            results = pipeline.wait_and_execute_at_open(queue)
            for t in results:
                print(f"  {t.symbol}: {t.status}")
    else:
        results = pipeline.run_fast(args.symbols, timeframe=args.timeframe, execute=args.execute)
        print(FastPipeline.format_results(results))
