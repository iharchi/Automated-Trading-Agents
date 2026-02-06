"""Scanner Agent

Autonomous agent that continuously scans the market for trading opportunities
and can automatically execute trades based on configurable criteria.

Capabilities:
    - Scans multiple stock universes (VOLATILE, MOST_ACTIVE, TECH, etc.)
    - Parallel scanning with configurable workers
    - Market regime detection for context
    - Technical analysis scoring for each symbol
    - Automatic trade execution through the trading pipeline
    - Continuous loop mode for 24/7 operation

Pipeline position:
    **Scanner Agent** ──> Trading Pipeline ──> Execution

Usage:
    agent = ScannerAgent(auto_trade=True, max_trades=3)
    result = agent.run("VOLATILE")  # Scans the VOLATILE universe

    # Or with custom symbols
    result = agent.analyze("CUSTOM", symbols=["AAPL", "TSLA", "NVDA"])
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.base_agent import BaseAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from utils.alpaca_client import AlpacaClient
from utils.market_regime import MarketRegimeDetector

logger = logging.getLogger(__name__)


# ── Stock Universes ─────────────────────────────────────────────

UNIVERSES = {
    "DEFAULT": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "AMD", "COIN", "MARA", "SMCI", "PLTR", "RIVN", "LCID",
    ],
    "VOLATILE": [
        "NVDA", "AMD", "TSLA", "COIN", "MARA", "SMCI", "PLTR",
        "RIVN", "LCID", "RIOT", "MSTR", "AFRM", "UPST", "SOFI",
        "RBLX", "SNOW", "DKNG", "CRWD", "NET", "ROKU",
    ],
    "MOST_ACTIVE": [
        "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN",
        "META", "GOOGL", "NFLX", "COIN", "MARA", "RIOT", "SOFI",
        "PLTR", "INTC", "BAC", "F", "T", "PFE", "AAL", "NIO",
        "SNAP", "UBER", "HOOD", "RBLX", "DKNG", "LCID", "RIVN",
        "SMCI", "ARM", "MSTR", "CRWD", "NET", "ROKU", "SQ", "PYPL",
        "DIS", "WMT", "JPM", "V", "MA", "XOM", "CVX", "JNJ",
        "UNH", "HD", "PG", "KO",
    ],
    "SP500_TOP50": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "BRK.B", "UNH", "JNJ", "XOM", "JPM", "V", "PG", "MA",
        "HD", "CVX", "MRK", "ABBV", "LLY", "PEP", "KO", "COST",
        "AVGO", "MCD", "WMT", "CSCO", "TMO", "ACN", "ABT",
        "DHR", "CRM", "ADBE", "NKE", "NFLX", "INTC", "AMD",
        "TXN", "PM", "UPS", "RTX", "HON", "QCOM", "LOW",
        "NEE", "UNP", "ORCL", "IBM", "GS", "CAT",
    ],
    "TECH": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "AMD", "INTC", "CRM", "ADBE", "NFLX", "ORCL", "IBM",
        "CSCO", "QCOM", "TXN", "AVGO", "NOW", "SNOW",
    ],
    "CRYPTO_RELATED": [
        "COIN", "MARA", "RIOT", "MSTR", "HUT", "BITF", "CLSK",
        "SI", "SQ", "PYPL", "HOOD",
    ],
}


@dataclass
class ScanOpportunity:
    """A single trading opportunity found by the scanner."""
    symbol: str
    signal: str = "HOLD"
    ta_score: int = 0
    price: float = 0.0
    atr: float = 0.0
    rsi: float = 0.0
    combined_score: float = 0.0
    confidence: float = 0.0
    error: str = ""


@dataclass
class ScanResult:
    """Complete result from a market scan."""
    universe: str
    timestamp: str
    market_regime: str = "UNKNOWN"
    total_scanned: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0
    errors: int = 0
    opportunities: list[ScanOpportunity] = field(default_factory=list)
    top_buys: list[ScanOpportunity] = field(default_factory=list)
    top_sells: list[ScanOpportunity] = field(default_factory=list)
    scan_duration_sec: float = 0.0
    executions: list = field(default_factory=list)


class ScannerAgent(BaseAgent):
    """Agent that scans markets and executes trades on opportunities."""

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        auto_trade: bool = False,
        dry_run: bool = True,
        max_workers: int = 8,
        max_trades: int = 5,
        min_score: float = 0.05,
        top_n: int = 15,
    ):
        super().__init__(name="Scanner", client=client)
        self.auto_trade = auto_trade
        self.dry_run = dry_run
        self.max_workers = max_workers
        self.max_trades = max_trades
        self.min_score = min_score
        self.top_n = top_n

        # Sub-components
        self.ta_agent = TechnicalAnalysisAgent(client=self.client)
        self.regime_detector = MarketRegimeDetector(client=self.client)
        self._pipeline = None  # Lazy init to avoid circular import

    @property
    def pipeline(self):
        """Lazy-load the trading pipeline to avoid circular imports."""
        if self._pipeline is None:
            from utils.trading_pipeline import TradingPipeline
            self._pipeline = TradingPipeline(client=self.client, dry_run=self.dry_run)
        return self._pipeline

    # ── Symbol scanning ───────────────────────────────────────

    def _scan_single(self, symbol: str) -> ScanOpportunity:
        """Scan a single symbol for opportunities."""
        opp = ScanOpportunity(symbol=symbol)

        try:
            # Technical analysis
            ta = self.ta_agent.analyze(symbol)
            opp.signal = ta.get("signal", "HOLD")
            opp.ta_score = ta.get("composite_score", 0)
            opp.price = ta.get("current_price", 0.0)
            opp.atr = ta.get("atr", 0.0)

            # Extract RSI
            for ind in ta.get("indicators", []):
                if isinstance(ind, dict) and ind.get("name") == "RSI":
                    opp.rsi = ind.get("value", 0.0)
                elif hasattr(ind, "name") and ind.name == "RSI":
                    opp.rsi = ind.value

            # Combined score (normalized TA)
            opp.combined_score = round(opp.ta_score / 6.0, 4)
            opp.confidence = min(abs(opp.ta_score) / 6.0, 1.0)

        except Exception as e:
            opp.error = str(e)
            self.logger.warning("Scan failed for %s: %s", symbol, e)

        return opp

    def _get_symbols(self, universe: str, custom_symbols: list[str] | None = None) -> list[str]:
        """Get symbols to scan based on universe name."""
        if universe == "CUSTOM" and custom_symbols:
            return custom_symbols
        elif universe == "ALL":
            # Fetch all tradeable assets
            try:
                all_symbols = self.client.get_tradeable_assets()
                return all_symbols[:500]  # Limit to 500
            except Exception as e:
                self.logger.error("Failed to fetch assets: %s", e)
                return UNIVERSES["DEFAULT"]
        else:
            return UNIVERSES.get(universe, UNIVERSES["DEFAULT"])

    # ── Core lifecycle ────────────────────────────────────────

    def analyze(self, universe: str, **kwargs) -> dict:
        """Scan a universe of stocks and return opportunities.

        Args:
            universe: Name of universe (VOLATILE, MOST_ACTIVE, TECH, etc.)
                      or "CUSTOM" with symbols kwarg
            **kwargs:
                symbols: List of symbols for CUSTOM universe
                top_n: Number of top opportunities to return

        Returns:
            ScanResult as dict
        """
        import time
        start = time.time()

        symbols = kwargs.get("symbols") or []
        top_n = kwargs.get("top_n", self.top_n)
        symbols_to_scan = self._get_symbols(universe, symbols)

        result = ScanResult(
            universe=universe,
            timestamp=datetime.now().isoformat(),
        )

        # Detect market regime
        try:
            regime_obj = self.regime_detector.detect("SPY")
            if hasattr(regime_obj, "regime"):
                regime = regime_obj.regime
                result.market_regime = regime.value if hasattr(regime, "value") else str(regime)
            else:
                result.market_regime = str(regime_obj)
        except Exception as e:
            self.logger.warning("Regime detection failed: %s", e)
            result.market_regime = "UNKNOWN"

        # Parallel scanning
        opportunities: list[ScanOpportunity] = []

        self.logger.info("Scanning %d symbols from %s universe", len(symbols_to_scan), universe)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._scan_single, sym): sym
                for sym in symbols_to_scan
            }
            for future in as_completed(futures):
                opp = future.result()
                opportunities.append(opp)

        # Count signals
        for opp in opportunities:
            result.total_scanned += 1
            if opp.error:
                result.errors += 1
            elif opp.signal == "BUY":
                result.buy_signals += 1
            elif opp.signal == "SELL":
                result.sell_signals += 1
            else:
                result.hold_signals += 1

        result.opportunities = opportunities

        # Filter by minimum score
        filtered = [o for o in opportunities if not o.error and abs(o.combined_score) >= self.min_score]

        # Sort and get top opportunities
        buys = sorted(
            [o for o in filtered if o.signal == "BUY"],
            key=lambda x: x.combined_score,
            reverse=True,
        )[:top_n]

        sells = sorted(
            [o for o in filtered if o.signal == "SELL"],
            key=lambda x: x.combined_score,
        )[:top_n]

        result.top_buys = buys
        result.top_sells = sells
        result.scan_duration_sec = round(time.time() - start, 2)

        self.logger.info(
            "Scan complete: %d symbols, %d BUY, %d SELL, %d HOLD, %d errors (%.2fs)",
            result.total_scanned, result.buy_signals, result.sell_signals,
            result.hold_signals, result.errors, result.scan_duration_sec,
        )

        return result.__dict__

    def execute(self, universe: str, analysis: dict, **kwargs) -> dict:
        """Execute trades based on scan results.

        Args:
            universe: Universe name (for logging)
            analysis: ScanResult dict from analyze()

        Returns:
            Updated analysis dict with executions
        """
        if not self.auto_trade:
            self.logger.info("Auto-trade disabled, skipping execution")
            return analysis

        top_buys = analysis.get("top_buys", [])
        top_sells = analysis.get("top_sells", [])

        # Get symbols to execute
        buy_symbols = [
            (o["symbol"] if isinstance(o, dict) else o.symbol)
            for o in top_buys
        ][:self.max_trades]

        sell_symbols = [
            (o["symbol"] if isinstance(o, dict) else o.symbol)
            for o in top_sells
        ][:self.max_trades]

        all_symbols = buy_symbols + sell_symbols

        if not all_symbols:
            self.logger.info("No symbols meet execution criteria")
            return analysis

        mode = "DRY RUN" if self.dry_run else "LIVE"
        self.logger.info("Executing %d trades (%s): %s", len(all_symbols), mode, all_symbols)

        # Run through the trading pipeline
        results = self.pipeline.run(all_symbols)

        # Add executions to analysis
        analysis["executions"] = [r.__dict__ if hasattr(r, "__dict__") else r for r in results]

        executed = sum(1 for r in results if r.executed)
        self.logger.info("Execution complete: %d/%d trades executed", executed, len(all_symbols))

        return analysis

    # ── Convenience methods ───────────────────────────────────

    def scan(
        self,
        universe: str = "VOLATILE",
        *,
        symbols: list[str] | None = None,
        execute: bool = False,
    ) -> dict:
        """Convenience method to scan and optionally execute.

        Args:
            universe: Universe name or "CUSTOM"
            symbols: Custom symbols list (for CUSTOM universe)
            execute: Whether to execute trades (overrides auto_trade)

        Returns:
            ScanResult dict with opportunities and optional executions
        """
        analysis = self.analyze(universe, symbols=symbols)

        if execute or self.auto_trade:
            # Temporarily enable auto_trade for this call
            original = self.auto_trade
            self.auto_trade = True
            analysis = self.execute(universe, analysis)
            self.auto_trade = original

        return analysis

    def run_continuous(
        self,
        universe: str = "VOLATILE",
        *,
        interval: int = 60,
        callback: callable = None,
    ):
        """Run the scanner in continuous loop mode.

        Args:
            universe: Universe to scan
            interval: Seconds between scans
            callback: Optional function to call with each result
        """
        import time

        self.logger.info("Starting continuous scan mode (interval=%ds)", interval)

        try:
            while True:
                result = self.scan(universe, execute=self.auto_trade)

                if callback:
                    callback(result)
                else:
                    self._print_summary(result)

                self.logger.info("Next scan in %d seconds...", interval)
                time.sleep(interval)
        except KeyboardInterrupt:
            self.logger.info("Scanner stopped by user")

    # ── Formatting ────────────────────────────────────────────

    def _print_summary(self, result: dict):
        """Print a summary of scan results."""
        print(self.format_analysis(result))

    @staticmethod
    def format_analysis(analysis: dict) -> str:
        """Format scan results for display."""
        lines = [
            "",
            "=" * 70,
            "  SCANNER AGENT — MARKET SCAN",
            "=" * 70,
            f"  Universe   : {analysis.get('universe', 'N/A')}",
            f"  Regime     : {analysis.get('market_regime', 'UNKNOWN')}",
            f"  Scanned    : {analysis.get('total_scanned', 0)} symbols",
            f"  Duration   : {analysis.get('scan_duration_sec', 0)}s",
            f"  Signals    : {analysis.get('buy_signals', 0)} BUY / "
            f"{analysis.get('sell_signals', 0)} SELL / "
            f"{analysis.get('hold_signals', 0)} HOLD",
            "-" * 70,
        ]

        top_buys = analysis.get("top_buys", [])
        if top_buys:
            lines.append("")
            lines.append("  TOP BUY OPPORTUNITIES")
            lines.append("  " + "-" * 66)
            lines.append(
                f"  {'Symbol':<8} {'Score':>8} {'TA':>5} {'RSI':>6} {'Price':>10}"
            )
            lines.append("  " + "-" * 66)
            for o in top_buys:
                if isinstance(o, dict):
                    lines.append(
                        f"  {o.get('symbol', ''):<8} {o.get('combined_score', 0):>+8.4f} "
                        f"{o.get('ta_score', 0):>+5d} {o.get('rsi', 0):>6.1f} "
                        f"${o.get('price', 0):>9.2f}"
                    )
                else:
                    lines.append(
                        f"  {o.symbol:<8} {o.combined_score:>+8.4f} "
                        f"{o.ta_score:>+5d} {o.rsi:>6.1f} ${o.price:>9.2f}"
                    )

        top_sells = analysis.get("top_sells", [])
        if top_sells:
            lines.append("")
            lines.append("  TOP SELL SIGNALS")
            lines.append("  " + "-" * 66)
            for o in top_sells:
                if isinstance(o, dict):
                    lines.append(
                        f"  {o.get('symbol', ''):<8} {o.get('combined_score', 0):>+8.4f} "
                        f"{o.get('ta_score', 0):>+5d} {o.get('rsi', 0):>6.1f} "
                        f"${o.get('price', 0):>9.2f}"
                    )
                else:
                    lines.append(
                        f"  {o.symbol:<8} {o.combined_score:>+8.4f} "
                        f"{o.ta_score:>+5d} {o.rsi:>6.1f} ${o.price:>9.2f}"
                    )

        # Executions
        executions = analysis.get("executions", [])
        if executions:
            lines.append("")
            lines.append("  EXECUTIONS")
            lines.append("  " + "-" * 66)
            for ex in executions:
                if isinstance(ex, dict):
                    symbol = ex.get("symbol", "")
                    signal = ex.get("signal", "")
                    shares = ex.get("shares", 0)
                    status = ex.get("order_status", "")
                    lines.append(f"  {symbol:<8} {signal:<6} {shares:>5} shares — {status}")

        if not top_buys and not top_sells:
            lines.append("")
            lines.append("  No strong signals found.")

        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  Completed at {analysis.get('timestamp', '')[:19]}")
        lines.append("=" * 70)
        lines.append("")

        return "\n".join(lines)
