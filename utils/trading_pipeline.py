"""Unified Trading Pipeline

End-to-end pipeline that chains every component together for
automated trade execution:

    0. (Optional) Scanner Agent — scan market for opportunities
    1. Market Regime Detection (SPY proxy)
    2. Per-symbol: TA → Sentiment → Signal Aggregator (regime-aware)
    3. Correlation check against existing positions
    4. Kelly-optimal position sizing (regime + vol adjusted)
    5. Risk management gate
    6. Execution (bracket orders with stops)
    7. Trailing stop registration for new fills
    8. Journal + Notify + Event Bus throughout

Usage:
    # Standard mode - run on specific symbols
    pipeline = TradingPipeline(dry_run=True)
    results = pipeline.run(["AAPL", "MSFT"], timeframe="1Day")

    # Scan mode - scan market and execute on opportunities
    results = pipeline.scan_and_run(universe="VOLATILE", max_trades=3)
"""

import logging
from dataclasses import dataclass, field

from utils.alpaca_client import AlpacaClient
from utils.signal_aggregator import SignalAggregator
from utils.position_sizer import PositionSizer
from utils.market_regime import MarketRegimeDetector
from utils.correlation_filter import CorrelationFilter
from utils.trailing_stop_manager import TrailingStopManager
from utils.event_bus import EventBus
from utils.trade_journal import TradeJournal
from utils.notifier import Notifier

from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.sentiment_analysis_agent import SentimentAnalysisAgent
from agents.risk_management_agent import RiskManagementAgent
from agents.execution_agent import ExecutionAgent
from agents.scanner_agent import ScannerAgent
from agents.portfolio_guardian_agent import PortfolioGuardianAgent
from utils.profit_monitor import ProfitMonitor

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """Result from one pipeline stage."""
    name: str
    passed: bool = True
    detail: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Full result from the unified pipeline for one symbol."""
    symbol: str
    steps: list[PipelineStep] = field(default_factory=list)

    # Signal
    signal: str = "HOLD"
    combined_score: float = 0.0
    confidence: float = 0.0

    # Sizing
    shares: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_per_share: float = 0.0
    kelly_fraction: float = 0.0

    # Execution
    order_status: str = ""          # filled, dry_run, skipped, failed
    order_id: str = ""
    filled_qty: int = 0
    filled_avg_price: float = 0.0

    # Context
    regime: str = ""
    correlation_filtered: bool = False
    risk_approved: bool = False
    price: float = 0.0
    atr: float = 0.0
    error: str = ""

    @property
    def executed(self) -> bool:
        return self.order_status in ("filled", "dry_run", "pending", "submitted")


class TradingPipeline:
    """Unified end-to-end trading pipeline."""

    def __init__(
        self,
        *,
        client: AlpacaClient | None = None,
        dry_run: bool = False,
        # Component overrides (for testing / DI)
        aggregator: SignalAggregator | None = None,
        sizer: PositionSizer | None = None,
        regime_detector: MarketRegimeDetector | None = None,
        correlation_filter: CorrelationFilter | None = None,
        trail_manager: TrailingStopManager | None = None,
        event_bus: EventBus | None = None,
        journal: TradeJournal | None = None,
        notifier: Notifier | None = None,
        risk_agent: RiskManagementAgent | None = None,
        exec_agent: ExecutionAgent | None = None,
        ta_agent: TechnicalAnalysisAgent | None = None,
        sentiment_agent: SentimentAnalysisAgent | None = None,
        scanner_agent: ScannerAgent | None = None,
        guardian_agent: PortfolioGuardianAgent | None = None,
        profit_monitor: ProfitMonitor | None = None,
        # Feature toggles
        enable_regime: bool = True,
        enable_correlation: bool = True,
        enable_trailing_stops: bool = True,
        enable_events: bool = True,
        enable_journal: bool = True,
        enable_notifications: bool = True,
        enable_guardian: bool = True,
        # Profit protection settings
        take_profit_pct: float = 0.10,  # 10% profit target
        trailing_stop_pct: float = 0.05,  # 5% trailing stop
        daily_loss_limit_pct: float = 0.03,  # 3% daily loss limit
        max_drawdown_pct: float = 0.10,  # 10% max drawdown
    ):
        self.client = client
        self.dry_run = dry_run

        # Components (lazily initialised if client provided)
        self.aggregator = aggregator or SignalAggregator()
        self.sizer = sizer or PositionSizer()
        self.regime_detector = regime_detector
        self.correlation_filter = correlation_filter
        self.trail_manager = trail_manager
        self.event_bus = event_bus or (EventBus() if enable_events else None)
        self.journal = journal
        self.notifier = notifier
        self.risk_agent = risk_agent
        self.exec_agent = exec_agent
        self.ta_agent = ta_agent
        self.sentiment_agent = sentiment_agent
        self.scanner_agent = scanner_agent
        self.guardian_agent = guardian_agent
        self.profit_monitor = profit_monitor

        # Toggles
        self.enable_regime = enable_regime
        self.enable_correlation = enable_correlation
        self.enable_trailing_stops = enable_trailing_stops
        self.enable_events = enable_events
        self.enable_journal = enable_journal
        self.enable_notifications = enable_notifications
        self.enable_guardian = enable_guardian

        # Profit protection settings
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct

    # ── Lazy component init ───────────────────────────────────────

    def _ensure_components(self) -> None:
        """Create components that need a client, if not already provided."""
        if self.client is None:
            self.client = AlpacaClient()

        if self.ta_agent is None:
            self.ta_agent = TechnicalAnalysisAgent(
                client=self.client, auto_trade=False,
            )
        if self.sentiment_agent is None:
            self.sentiment_agent = SentimentAnalysisAgent(
                client=self.client, auto_trade=False,
            )
        if self.risk_agent is None:
            self.risk_agent = RiskManagementAgent(
                client=self.client, auto_trade=False,
            )
        if self.exec_agent is None:
            self.exec_agent = ExecutionAgent(
                client=self.client, dry_run=self.dry_run,
                skip_market_check=True,  # Allow paper trading outside market hours
            )
        if self.enable_regime and self.regime_detector is None:
            self.regime_detector = MarketRegimeDetector(client=self.client)
        if self.enable_correlation and self.correlation_filter is None:
            self.correlation_filter = CorrelationFilter(client=self.client)
        if self.enable_trailing_stops and self.trail_manager is None:
            self.trail_manager = TrailingStopManager(
                client=self.client, state_file="default",
            )
        if self.enable_journal and self.journal is None:
            self.journal = TradeJournal()

        if self.scanner_agent is None:
            self.scanner_agent = ScannerAgent(
                client=self.client,
                auto_trade=False,  # Pipeline handles execution
                dry_run=self.dry_run,
            )
        if self.enable_guardian and self.guardian_agent is None:
            self.guardian_agent = PortfolioGuardianAgent(
                client=self.client,
                take_profit_pct=self.take_profit_pct,
                trailing_stop_pct=self.trailing_stop_pct,
                daily_loss_limit_pct=self.daily_loss_limit_pct,
                max_drawdown_pct=self.max_drawdown_pct,
                auto_trade=not self.dry_run,  # Auto-close positions in live mode
                dry_run=self.dry_run,
            )
        if self.profit_monitor is None:
            self.profit_monitor = ProfitMonitor(
                client=self.client,
                take_profit_pct=self.take_profit_pct,
                trailing_stop_pct=self.trailing_stop_pct,
                daily_loss_limit_pct=self.daily_loss_limit_pct,
                max_drawdown_pct=self.max_drawdown_pct,
            )

    # ── Event helpers ─────────────────────────────────────────────

    def _emit(self, event_type: str, data: dict, symbol: str = "", source: str = ""):
        if self.event_bus and self.enable_events:
            self.event_bus.publish(event_type, data, source=source, symbol=symbol)

    def _notify_signal(self, symbol: str, signal: str, score: float, price: float, confidence: float):
        if self.notifier and self.enable_notifications and signal in ("BUY", "SELL"):
            self.notifier.notify_signal(
                symbol=symbol, signal=signal, score=score,
                price=price, confidence=confidence,
            )

    def _notify_order(self, symbol: str, exec_result: dict):
        if not self.notifier or not self.enable_notifications:
            return
        status = exec_result.get("status", "")
        side = exec_result.get("side", "")
        qty = exec_result.get("qty", 0)
        if status == "filled":
            self.notifier.notify_order_filled(
                symbol=symbol, side=side, qty=qty,
                avg_price=exec_result.get("filled_avg_price", 0),
                order_id=exec_result.get("order_id", ""),
            )
        elif status == "failed":
            self.notifier.notify_order_failed(
                symbol=symbol, side=side, qty=qty,
                error=exec_result.get("error", "Unknown"),
            )

    # ── Step runners ──────────────────────────────────────────────

    def _step_regime(self, timeframe: str) -> tuple[PipelineStep, dict | None]:
        """Step 1: Detect market regime via SPY."""
        if not self.enable_regime or self.regime_detector is None:
            return PipelineStep(name="regime", detail="skipped (disabled)"), None

        try:
            regime_obj = self.regime_detector.detect("SPY", timeframe=timeframe)
            regime_dict = regime_obj.__dict__
            self._emit(
                "regime_changed",
                {"new_regime": regime_obj.regime, "confidence": regime_obj.confidence},
                symbol="SPY",
                source="MarketRegimeDetector",
            )
            return PipelineStep(
                name="regime",
                detail=f"{regime_obj.regime} (confidence={regime_obj.confidence:.0%})",
                data=regime_dict,
            ), regime_dict
        except Exception as e:
            logger.warning("Regime detection failed: %s", e)
            return PipelineStep(name="regime", detail=f"error: {e}"), None

    def _step_ta(self, symbol: str, timeframe: str) -> tuple[PipelineStep, dict | None]:
        """Step 2a: Technical Analysis."""
        try:
            ta_result = self.ta_agent.analyze(symbol, timeframe=timeframe)
            self._emit(
                "signal_generated",
                {"signal": ta_result.get("signal", "HOLD"), "score": ta_result.get("composite_score", 0)},
                symbol=symbol, source="TechnicalAnalysis",
            )
            return PipelineStep(
                name="ta",
                detail=f"signal={ta_result.get('signal')} score={ta_result.get('composite_score')}",
                data=ta_result,
            ), ta_result
        except Exception as e:
            logger.error("TA failed for %s: %s", symbol, e)
            return PipelineStep(name="ta", passed=False, detail=str(e)), None

    def _step_sentiment(self, symbol: str, timeframe: str) -> tuple[PipelineStep, dict | None]:
        """Step 2b: Sentiment Analysis."""
        try:
            sent_result = self.sentiment_agent.analyze(symbol, timeframe=timeframe)
            self._emit(
                "signal_generated",
                {"signal": sent_result.get("signal", "HOLD"), "score": sent_result.get("composite_score", 0)},
                symbol=symbol, source="SentimentAnalysis",
            )
            return PipelineStep(
                name="sentiment",
                detail=f"signal={sent_result.get('signal')} score={sent_result.get('composite_score', 0):.3f}",
                data=sent_result,
            ), sent_result
        except Exception as e:
            logger.error("Sentiment failed for %s: %s", symbol, e)
            return PipelineStep(name="sentiment", passed=False, detail=str(e)), None

    def _step_aggregate(
        self, symbol: str, ta_result: dict | None, sent_result: dict | None,
        regime_result: dict | None, corr_result: dict | None,
    ) -> tuple[PipelineStep, object | None]:
        """Step 3: Signal Aggregation."""
        try:
            agg = self.aggregator.aggregate(
                symbol,
                ta_result=ta_result,
                sentiment_result=sent_result,
                regime_result=regime_result,
                correlation_result=corr_result,
            )
            self._emit(
                "aggregation_done",
                {"signal": agg.signal, "combined_score": agg.combined_score, "confidence": agg.confidence},
                symbol=symbol, source="SignalAggregator",
            )
            return PipelineStep(
                name="aggregate",
                detail=f"signal={agg.signal} score={agg.combined_score:+.4f} conf={agg.confidence:.0%}",
                data=agg.__dict__,
            ), agg
        except Exception as e:
            logger.error("Aggregation failed for %s: %s", symbol, e)
            return PipelineStep(name="aggregate", passed=False, detail=str(e)), None

    def _step_correlation(self, symbol: str, positions: list[str]) -> tuple[PipelineStep, dict | None]:
        """Step 4: Correlation check against existing positions."""
        if not self.enable_correlation or self.correlation_filter is None or not positions:
            return PipelineStep(name="correlation", detail="skipped"), None

        try:
            corr = self.correlation_filter.check_correlation(symbol, positions)
            corr_dict = corr.__dict__ if hasattr(corr, "__dict__") else corr
            is_corr = corr_dict.get("is_correlated", False)
            if is_corr:
                self._emit(
                    "correlation_alert",
                    {"correlated_with": corr_dict.get("correlated_with", [])},
                    symbol=symbol, source="CorrelationFilter",
                )
            return PipelineStep(
                name="correlation",
                detail=f"correlated={is_corr}" + (
                    f" with {corr_dict.get('correlated_with', [])}" if is_corr else ""
                ),
                data=corr_dict,
            ), corr_dict
        except Exception as e:
            logger.warning("Correlation check failed for %s: %s", symbol, e)
            return PipelineStep(name="correlation", detail=f"error: {e}"), None

    def _step_sizing(
        self, symbol: str, signal: str, price: float, atr: float,
        equity: float, regime_adjustments: dict | None,
        annual_vol: float | None, portfolio_heat: float,
    ) -> tuple[PipelineStep, object | None]:
        """Step 5: Kelly position sizing."""
        if signal not in ("BUY", "SELL"):
            return PipelineStep(name="sizing", detail="skipped (HOLD signal)"), None

        try:
            sizing = self.sizer.calculate(
                symbol,
                price=price,
                equity=equity,
                signal=signal,
                atr=atr,
                regime_adjustments=regime_adjustments,
                annual_vol=annual_vol,
                portfolio_heat=portfolio_heat,
            )
            self._emit(
                "sizing_done",
                {"shares": sizing.shares, "risk_pct": sizing.risk_pct, "method": sizing.method},
                symbol=symbol, source="PositionSizer",
            )
            return PipelineStep(
                name="sizing",
                detail=f"shares={sizing.shares} risk={sizing.risk_pct:.2%} kelly={sizing.kelly_fraction:.2%}",
                data=sizing.__dict__,
            ), sizing
        except Exception as e:
            logger.error("Sizing failed for %s: %s", symbol, e)
            return PipelineStep(name="sizing", passed=False, detail=str(e)), None

    def _step_risk(
        self, symbol: str, signal: str, price: float, atr: float,
        shares: int, stop_loss: float, take_profit: float,
    ) -> tuple[PipelineStep, dict | None]:
        """Step 6: Risk management gate."""
        if signal not in ("BUY", "SELL"):
            return PipelineStep(name="risk", detail="skipped (HOLD)"), None

        try:
            side = "buy" if signal == "BUY" else "sell"
            risk_result = self.risk_agent.analyze(
                symbol,
                proposed_side=side,
                price=price,  # Risk agent expects 'price', not 'proposed_price'
                atr=atr,
            )
            approved = risk_result.get("approved", False)

            if not approved:
                self._emit(
                    "risk_breach",
                    {"checks": risk_result.get("checks", [])},
                    symbol=symbol, source="RiskManagement",
                )

            return PipelineStep(
                name="risk",
                passed=approved,
                detail=f"approved={approved} size={risk_result.get('position_size', 0)}",
                data=risk_result,
            ), risk_result
        except Exception as e:
            logger.error("Risk check failed for %s: %s", symbol, e)
            return PipelineStep(name="risk", passed=False, detail=str(e)), None

    def _step_execute(
        self, symbol: str, decision: dict,
    ) -> tuple[PipelineStep, dict | None]:
        """Step 7: Order execution."""
        try:
            exec_analysis = self.exec_agent.analyze(symbol, decision=decision)
            exec_result = self.exec_agent.execute(symbol, exec_analysis)
            status = exec_result.get("status", "skipped")

            if status in ("filled", "dry_run", "pending", "submitted"):
                self._emit(
                    "position_opened",
                    {
                        "side": exec_result.get("side"),
                        "qty": exec_result.get("qty"),
                        "status": status,
                        "order_id": exec_result.get("order_id", ""),
                    },
                    symbol=symbol, source="ExecutionAgent",
                )

            return PipelineStep(
                name="execute",
                passed=status != "failed",
                detail=f"status={status} qty={exec_result.get('qty', 0)}",
                data=exec_result,
            ), exec_result
        except Exception as e:
            logger.error("Execution failed for %s: %s", symbol, e)
            return PipelineStep(name="execute", passed=False, detail=str(e)), None

    def _step_trailing_stop(
        self, symbol: str, signal: str, entry_price: float,
        shares: int, atr: float, stop_loss: float,
    ) -> PipelineStep:
        """Step 8: Register trailing stop for new fill."""
        if not self.enable_trailing_stops or self.trail_manager is None:
            return PipelineStep(name="trailing_stop", detail="skipped (disabled)")

        if signal not in ("BUY", "SELL") or shares == 0:
            return PipelineStep(name="trailing_stop", detail="skipped (no position)")

        try:
            side = "long" if signal == "BUY" else "short"
            self.trail_manager.add_position(
                symbol,
                entry_price=entry_price,
                qty=shares,
                side=side,
                atr=atr,
            )
            return PipelineStep(
                name="trailing_stop",
                detail=f"registered {side} {shares}@{entry_price:.2f}",
            )
        except Exception as e:
            logger.warning("Trailing stop registration failed for %s: %s", symbol, e)
            return PipelineStep(name="trailing_stop", detail=f"error: {e}")

    # ── Guardian checks ──────────────────────────────────────────

    def _step_guardian_check(self) -> tuple[PipelineStep, bool, str]:
        """Check if trading should be halted based on guardian rules."""
        if not self.enable_guardian or self.guardian_agent is None:
            return PipelineStep(name="guardian", detail="skipped (disabled)"), False, ""

        try:
            should_halt, halt_reason = self.profit_monitor.should_halt_trading()
            if should_halt:
                self._emit(
                    "trading_halted",
                    {"reason": halt_reason},
                    source="PortfolioGuardian",
                )
                return PipelineStep(
                    name="guardian",
                    passed=False,
                    detail=f"TRADING HALTED: {halt_reason}",
                ), True, halt_reason

            # Get portfolio metrics for logging
            metrics = self.profit_monitor.get_portfolio_metrics()
            return PipelineStep(
                name="guardian",
                detail=f"OK - Daily P&L: {metrics.daily_pnl_pct:+.2%}, Drawdown: {metrics.current_drawdown_pct:.2%}",
                data={"daily_pnl_pct": metrics.daily_pnl_pct, "drawdown_pct": metrics.current_drawdown_pct},
            ), False, ""

        except Exception as e:
            logger.warning("Guardian check failed: %s", e)
            return PipelineStep(name="guardian", detail=f"error: {e}"), False, ""

    def _step_close_profit_targets(self) -> tuple[PipelineStep, list[dict]]:
        """Close positions that hit profit targets or trailing stops."""
        if not self.enable_guardian or self.guardian_agent is None:
            return PipelineStep(name="profit_protection", detail="skipped (disabled)"), []

        try:
            positions_to_close = self.profit_monitor.get_positions_to_close()
            if not positions_to_close:
                return PipelineStep(
                    name="profit_protection",
                    detail="No positions at targets",
                ), []

            # Execute closes
            analysis = self.guardian_agent.analyze()
            results = self.guardian_agent.execute(analysis=analysis)
            actions = results.get("actions_taken", [])

            # Emit events for closed positions
            for action in actions:
                self._emit(
                    "position_closed",
                    {
                        "symbol": action["symbol"],
                        "reason": action["reason"],
                        "status": action["status"],
                    },
                    symbol=action["symbol"],
                    source="PortfolioGuardian",
                )

            return PipelineStep(
                name="profit_protection",
                detail=f"Closed {len(actions)} positions",
                data={"closed": actions},
            ), actions

        except Exception as e:
            logger.error("Profit protection step failed: %s", e)
            return PipelineStep(name="profit_protection", passed=False, detail=str(e)), []

    # ── Main run ──────────────────────────────────────────────────

    def run(
        self,
        symbols: list[str],
        *,
        timeframe: str = "1Day",
    ) -> list[PipelineResult]:
        """Run the full pipeline for a list of symbols.

        Returns one PipelineResult per symbol.
        """
        self._ensure_components()
        results: list[PipelineResult] = []

        # Step 0: Guardian check - should we halt trading?
        guardian_step, trading_halted, halt_reason = self._step_guardian_check()
        if trading_halted:
            logger.warning("TRADING HALTED: %s", halt_reason)
            for symbol in symbols:
                r = PipelineResult(symbol=symbol, error=f"Trading halted: {halt_reason}")
                r.steps.append(guardian_step)
                results.append(r)
            return results

        # Step 0b: Close positions at profit targets/stops
        profit_step, closed_positions = self._step_close_profit_targets()

        # Step 1: Market regime (once for all symbols)
        regime_step, regime_result = self._step_regime(timeframe)
        regime_adjustments = None
        if regime_result:
            regime_adjustments = regime_result.get("adjustments")

        # Get account equity for sizing
        try:
            account = self.client.get_account()
            equity = account.get("equity", 0)
        except Exception as e:
            logger.error("Could not fetch account: %s", e)
            for symbol in symbols:
                r = PipelineResult(symbol=symbol, error=f"Account fetch failed: {e}")
                r.steps.append(regime_step)
                results.append(r)
            return results

        # Get existing positions for correlation check
        existing_positions: list[str] = []
        try:
            positions = self.client.get_positions()
            existing_positions = [p["symbol"] for p in positions]
        except Exception as e:
            logger.warning("Could not fetch positions: %s", e)

        # Calculate portfolio heat (approximate)
        portfolio_heat = 0.0
        # Rough estimate: each position contributes ~2% risk (risk_per_trade default)
        portfolio_heat = len(existing_positions) * 0.02

        for symbol in symbols:
            result = PipelineResult(symbol=symbol)
            result.steps.append(guardian_step)
            result.steps.append(profit_step)
            result.steps.append(regime_step)
            result.regime = regime_result.get("regime", "") if regime_result else ""

            # Step 2a: Technical Analysis
            ta_step, ta_result = self._step_ta(symbol, timeframe)
            result.steps.append(ta_step)
            if not ta_step.passed:
                result.error = f"TA failed: {ta_step.detail}"
                results.append(result)
                continue

            price = ta_result.get("current_price", 0)
            atr = ta_result.get("atr", 0)
            result.price = price
            result.atr = atr

            # Step 2b: Sentiment Analysis
            sent_step, sent_result = self._step_sentiment(symbol, timeframe)
            result.steps.append(sent_step)
            # Sentiment failure is non-fatal — aggregator handles missing sources

            # Step 3: Correlation check
            corr_step, corr_result = self._step_correlation(symbol, existing_positions)
            result.steps.append(corr_step)

            # Step 4: Signal Aggregation
            agg_step, agg = self._step_aggregate(
                symbol, ta_result, sent_result, regime_result, corr_result,
            )
            result.steps.append(agg_step)
            if agg is None:
                result.error = f"Aggregation failed: {agg_step.detail}"
                results.append(result)
                continue

            result.signal = agg.signal
            result.combined_score = agg.combined_score
            result.confidence = agg.confidence
            result.correlation_filtered = agg.correlation_filtered

            # Notify on actionable signal
            self._notify_signal(symbol, agg.signal, agg.combined_score, price, agg.confidence)

            # Journal signal
            if self.journal and self.enable_journal:
                for src in agg.sources:
                    s = src if isinstance(src, dict) else src.__dict__
                    self.journal.log_signal(
                        symbol=symbol,
                        agent=s.get("name", ""),
                        signal=s.get("signal", "HOLD"),
                        score=s.get("normalised_score", 0),
                        price=price,
                    )

            # If HOLD, skip execution
            if agg.signal == "HOLD":
                result.steps.append(PipelineStep(name="sizing", detail="skipped (HOLD)"))
                result.steps.append(PipelineStep(name="risk", detail="skipped (HOLD)"))
                result.steps.append(PipelineStep(name="execute", detail="skipped (HOLD)"))
                results.append(result)
                continue

            # Step 5: Position sizing
            annual_vol = None
            if regime_result:
                indicators = regime_result.get("indicators")
                if indicators:
                    if hasattr(indicators, "historical_vol"):
                        annual_vol = indicators.historical_vol
                    elif isinstance(indicators, dict):
                        annual_vol = indicators.get("historical_vol")

            sizing_step, sizing = self._step_sizing(
                symbol, agg.signal, price, atr, equity,
                regime_adjustments, annual_vol, portfolio_heat,
            )
            result.steps.append(sizing_step)

            if sizing is None or sizing.shares == 0:
                result.error = sizing.error if sizing else "Sizing returned 0 shares"
                result.steps.append(PipelineStep(name="risk", detail="skipped (0 shares)"))
                result.steps.append(PipelineStep(name="execute", detail="skipped (0 shares)"))
                results.append(result)
                continue

            result.shares = sizing.shares
            result.stop_loss = sizing.stop_loss
            result.take_profit = sizing.take_profit
            result.risk_per_share = sizing.risk_per_share
            result.kelly_fraction = sizing.kelly_fraction

            # Step 6: Risk management gate
            risk_step, risk_result = self._step_risk(
                symbol, agg.signal, price, atr,
                sizing.shares, sizing.stop_loss, sizing.take_profit,
            )
            result.steps.append(risk_step)
            result.risk_approved = risk_step.passed

            if not risk_step.passed:
                result.error = f"Risk rejected: {risk_step.detail}"
                result.steps.append(PipelineStep(name="execute", detail="skipped (risk rejected)"))
                results.append(result)
                continue

            # Build decision dict for ExecutionAgent (matches expected format)
            # Use the SMALLER of Kelly-sized and risk-sized shares
            risk_shares = risk_result.get("position_size", sizing.shares) if risk_result else sizing.shares
            final_shares = min(sizing.shares, risk_shares)
            result.shares = final_shares

            decision = {
                "symbol": symbol,
                "signal": agg.signal,
                "position_size": final_shares,
                "stop_loss": sizing.stop_loss,
                "take_profit": sizing.take_profit,
                "risk_approved": True,
                "current_price": price,
                "atr": atr,
                "combined_score": agg.combined_score,
                "confidence": agg.confidence,
            }

            # Journal the decision
            if self.journal and self.enable_journal:
                self.journal.log_decision(decision)

            # Step 7: Execution
            exec_step, exec_result = self._step_execute(symbol, decision)
            result.steps.append(exec_step)

            if exec_result:
                result.order_status = exec_result.get("status", "")
                result.order_id = exec_result.get("order_id", "")
                result.filled_qty = exec_result.get("filled_qty", 0)
                result.filled_avg_price = exec_result.get("filled_avg_price", 0)

                # Journal the order
                if self.journal and self.enable_journal and result.order_status not in ("skipped",):
                    self.journal.log_order(
                        symbol=symbol,
                        side=exec_result.get("side", ""),
                        qty=exec_result.get("qty", 0),
                        order_result={
                            "id": result.order_id,
                            "status": result.order_status,
                            "type": exec_result.get("order_type", "market"),
                        },
                    )

                # Notify
                self._notify_order(symbol, exec_result)

            # Step 8: Trailing stop (only for fills or dry-runs)
            if result.order_status in ("filled", "dry_run"):
                fill_price = result.filled_avg_price if result.filled_avg_price > 0 else price
                trail_step = self._step_trailing_stop(
                    symbol, agg.signal, fill_price,
                    result.filled_qty or final_shares, atr, sizing.stop_loss,
                )
                result.steps.append(trail_step)

            # Update existing positions for subsequent correlation checks
            if result.executed:
                existing_positions.append(symbol)

            results.append(result)

        return results

    # ── Scan and Run ─────────────────────────────────────────────

    def scan_and_run(
        self,
        universe: str = "VOLATILE",
        *,
        symbols: list[str] | None = None,
        max_trades: int = 3,
        min_score: float = 0.0,
        timeframe: str = "1Day",
    ) -> tuple[dict, list["PipelineResult"]]:
        """Scan the market for opportunities and execute on the best ones.

        This is the fully autonomous mode: the Scanner Agent finds opportunities,
        then the pipeline executes trades on the top signals.

        Args:
            universe: Stock universe to scan (VOLATILE, MOST_ACTIVE, TECH, etc.)
            symbols: Custom symbols (for CUSTOM universe)
            max_trades: Maximum trades to execute
            min_score: Minimum combined score to consider
            timeframe: Timeframe for analysis

        Returns:
            Tuple of (scan_result_dict, list_of_pipeline_results)
        """
        self._ensure_components()

        logger.info("Starting scan_and_run: universe=%s, max_trades=%d", universe, max_trades)

        # Step 0: Scanner Agent finds opportunities
        scan_result = self.scanner_agent.analyze(
            universe,
            symbols=symbols,
            top_n=max_trades * 2,  # Get more candidates than needed
        )

        # Emit scan event
        self._emit(
            "scan_completed",
            {
                "universe": universe,
                "buy_signals": scan_result.get("buy_signals", 0),
                "sell_signals": scan_result.get("sell_signals", 0),
                "market_regime": scan_result.get("market_regime", ""),
            },
            source="ScannerAgent",
        )

        # Extract top symbols to trade
        top_buys = scan_result.get("top_buys", [])
        top_sells = scan_result.get("top_sells", [])

        buy_symbols = [
            (o["symbol"] if isinstance(o, dict) else o.symbol)
            for o in top_buys
            if (o.get("combined_score", 0) if isinstance(o, dict) else o.combined_score) >= min_score
        ][:max_trades]

        sell_symbols = [
            (o["symbol"] if isinstance(o, dict) else o.symbol)
            for o in top_sells
            if abs(o.get("combined_score", 0) if isinstance(o, dict) else o.combined_score) >= min_score
        ][:max_trades]

        all_symbols = buy_symbols + sell_symbols

        if not all_symbols:
            logger.info("No symbols meet criteria after scan")
            return scan_result, []

        mode = "DRY RUN" if self.dry_run else "LIVE"
        logger.info("Executing %d trades (%s): %s", len(all_symbols), mode, all_symbols)

        # Run the standard pipeline on selected symbols
        pipeline_results = self.run(all_symbols, timeframe=timeframe)

        # Log summary
        executed = sum(1 for r in pipeline_results if r.executed)
        logger.info(
            "scan_and_run complete: scanned=%d, signals=%d BUY / %d SELL, executed=%d/%d",
            scan_result.get("total_scanned", 0),
            scan_result.get("buy_signals", 0),
            scan_result.get("sell_signals", 0),
            executed,
            len(all_symbols),
        )

        return scan_result, pipeline_results

    def run_continuous(
        self,
        universe: str = "VOLATILE",
        *,
        interval: int = 60,
        max_trades: int = 3,
        callback: callable = None,
    ):
        """Run the pipeline in continuous scan-and-execute mode.

        Args:
            universe: Stock universe to scan
            interval: Seconds between scan cycles
            max_trades: Max trades per cycle
            callback: Optional function called with (scan_result, pipeline_results)
        """
        import time

        logger.info(
            "Starting continuous pipeline: universe=%s, interval=%ds, max_trades=%d",
            universe, interval, max_trades,
        )

        try:
            while True:
                scan_result, pipeline_results = self.scan_and_run(
                    universe, max_trades=max_trades,
                )

                if callback:
                    callback(scan_result, pipeline_results)
                else:
                    # Default: print summary
                    print(ScannerAgent.format_analysis(scan_result))
                    if pipeline_results:
                        print(self.format_summary(pipeline_results))

                logger.info("Next scan in %d seconds...", interval)
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Continuous pipeline stopped by user")

    # ── Formatting ────────────────────────────────────────────────

    @staticmethod
    def format_result(result: "PipelineResult") -> str:
        """Human-readable summary of one pipeline result."""
        lines = [
            f"\n{'=' * 66}",
            f"  PIPELINE RESULT -- {result.symbol}",
            f"{'=' * 66}",
        ]

        if result.error:
            lines.append(f"  Error: {result.error}")

        # Steps
        for step in result.steps:
            icon = "+" if step.passed else "x"
            lines.append(f"  [{icon}] {step.name:18s} {step.detail}")

        lines.append(f"{'~' * 66}")

        # Decision
        lines.append(
            f"  Signal   : {result.signal}  "
            f"(score={result.combined_score:+.4f} conf={result.confidence:.0%})"
        )
        if result.regime:
            lines.append(f"  Regime   : {result.regime}")
        if result.correlation_filtered:
            lines.append(f"  Corr     : FILTERED (downgraded to HOLD)")

        # Sizing
        if result.shares > 0:
            lines.append(
                f"  Shares   : {result.shares}  "
                f"SL=${result.stop_loss:.2f}  TP=${result.take_profit:.2f}"
            )
            lines.append(
                f"  Kelly    : {result.kelly_fraction:.2%}  "
                f"Risk/share=${result.risk_per_share:.2f}"
            )

        # Execution
        if result.order_status:
            lines.append(
                f"  Order    : {result.order_status}  "
                f"id={result.order_id or 'N/A'}  "
                f"filled={result.filled_qty}@${result.filled_avg_price:.2f}"
            )

        lines.append(f"{'=' * 66}\n")
        return "\n".join(lines)

    @staticmethod
    def format_summary(results: list["PipelineResult"]) -> str:
        """One-line-per-symbol summary table."""
        lines = [
            f"\n{'=' * 80}",
            f"  PIPELINE SUMMARY",
            f"{'=' * 80}",
            f"  {'Symbol':8s} {'Signal':6s} {'Score':>8s} {'Conf':>6s} {'Shares':>7s} {'Status':>10s} {'Regime':>12s}",
            f"  {'-'*8} {'-'*6} {'-'*8} {'-'*6} {'-'*7} {'-'*10} {'-'*12}",
        ]
        for r in results:
            lines.append(
                f"  {r.symbol:8s} {r.signal:6s} {r.combined_score:>+8.4f} "
                f"{r.confidence:>5.0%} {r.shares:>7d} "
                f"{r.order_status or 'N/A':>10s} {r.regime or 'N/A':>12s}"
            )

        # Counts
        buys = sum(1 for r in results if r.signal == "BUY")
        sells = sum(1 for r in results if r.signal == "SELL")
        holds = sum(1 for r in results if r.signal == "HOLD")
        executed = sum(1 for r in results if r.executed)
        lines.append(f"{'~' * 80}")
        lines.append(
            f"  Signals: {buys} BUY / {sells} SELL / {holds} HOLD  |  "
            f"Executed: {executed}/{len(results)}"
        )
        lines.append(f"{'=' * 80}\n")
        return "\n".join(lines)
