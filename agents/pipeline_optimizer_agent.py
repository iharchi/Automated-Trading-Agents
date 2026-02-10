"""Pipeline Optimizer Agent

Continuously monitors trading results and provides actionable suggestions
to enhance the pipeline for more profit and fewer losses.

Analysis Categories:
    - Performance metrics (win rate, profit factor, Sharpe ratio)
    - Parameter optimization suggestions (thresholds, stops, sizing)
    - Pattern detection (best/worst symbols, time patterns, regime performance)
    - Risk analysis (drawdown, consecutive losses, exposure)

Usage:
    optimizer = PipelineOptimizerAgent(client)

    # One-time analysis
    analysis = optimizer.analyze()
    suggestions = optimizer.execute(analysis=analysis)
    print(PipelineOptimizerAgent.format_suggestions(suggestions))

    # Continuous monitoring
    optimizer.monitor(interval=300)  # Check every 5 minutes
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from agents.base_agent import BaseAgent
from utils.alpaca_client import AlpacaClient
from utils.trade_journal import TradeJournal
from utils.performance_analytics import PerformanceAnalytics, PerformanceReport, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class OptimizationSuggestion:
    """A single optimization suggestion."""
    category: str           # "parameter", "risk", "symbol", "timing", "strategy"
    priority: str           # "high", "medium", "low"
    title: str              # Short title
    description: str        # Detailed description
    current_value: Any = None
    suggested_value: Any = None
    expected_impact: str = ""
    confidence: float = 0.0  # 0-1 confidence in the suggestion


@dataclass
class OptimizerAnalysis:
    """Full analysis from the optimizer."""
    timestamp: str = ""

    # Performance summary
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    expectancy: float = 0.0

    # Pattern analysis
    best_symbols: list = field(default_factory=list)
    worst_symbols: list = field(default_factory=list)
    best_regime: str = ""
    worst_regime: str = ""
    avg_hold_time: float = 0.0

    # Risk metrics
    current_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    daily_pnl_pct: float = 0.0
    portfolio_heat: float = 0.0

    # Raw data for suggestions
    performance_report: PerformanceReport | None = None
    signals_data: list = field(default_factory=list)
    decisions_data: list = field(default_factory=list)
    orders_data: list = field(default_factory=list)

    # Issues detected
    issues: list = field(default_factory=list)


@dataclass
class OptimizerResult:
    """Result containing all suggestions."""
    timestamp: str = ""
    analysis: OptimizerAnalysis | None = None
    suggestions: list[OptimizationSuggestion] = field(default_factory=list)
    summary: str = ""
    health_score: float = 0.0  # 0-100 overall pipeline health


class PipelineOptimizerAgent(BaseAgent):
    """Agent that monitors pipeline results and suggests optimizations."""

    # Thresholds for generating suggestions
    MIN_TRADES_FOR_ANALYSIS = 5
    LOW_WIN_RATE = 0.40
    HIGH_WIN_RATE = 0.65
    LOW_PROFIT_FACTOR = 1.0
    HIGH_PROFIT_FACTOR = 2.0
    LOW_SHARPE = 0.5
    HIGH_DRAWDOWN = 0.10
    MAX_CONSECUTIVE_LOSSES = 5
    LOW_EXPECTANCY = 0.0

    # Parameter bounds for suggestions
    PARAM_BOUNDS = {
        "take_profit_pct": (0.05, 0.30),
        "trailing_stop_pct": (0.03, 0.15),
        "daily_loss_limit_pct": (0.02, 0.08),
        "max_drawdown_pct": (0.05, 0.20),
        "buy_threshold": (0.05, 0.40),
        "sell_threshold": (-0.40, -0.05),
        "kelly_factor": (0.25, 0.75),
        "max_position_pct": (0.05, 0.30),
        "atr_stop_multiplier": (1.0, 3.0),
    }

    def __init__(
        self,
        client: AlpacaClient | None = None,
        journal: TradeJournal | None = None,
        lookback_days: int = 30,
        auto_apply: bool = False,
    ):
        super().__init__(name="PipelineOptimizer", client=client)
        self.journal = journal or TradeJournal()
        self.analytics = PerformanceAnalytics()
        self.lookback_days = lookback_days
        self.auto_apply = auto_apply
        self._last_analysis: OptimizerAnalysis | None = None
        self._suggestion_history: list[OptimizerResult] = []

    # ── Core Agent Methods ───────────────────────────────────────

    def analyze(self, symbol: str = "", **kwargs) -> dict:
        """Analyze pipeline performance and identify issues.

        Args:
            symbol: Ignored (analyzes full portfolio)
            **kwargs: Optional overrides (lookback_days, etc.)

        Returns:
            Analysis dict with performance metrics and issues
        """
        lookback = kwargs.get("lookback_days", self.lookback_days)

        analysis = OptimizerAnalysis(
            timestamp=datetime.now().isoformat(),
        )

        # Load journal data
        try:
            analysis.signals_data = self.journal.read_signals(tail=1000)
            analysis.decisions_data = self.journal.read_decisions(tail=500)
            analysis.orders_data = self.journal.read_orders(tail=500)
        except Exception as e:
            logger.warning("Failed to read journal data: %s", e)
            analysis.issues.append(f"Journal read error: {e}")

        # Filter by lookback period
        cutoff = datetime.now() - timedelta(days=lookback)
        analysis.orders_data = self._filter_by_date(analysis.orders_data, cutoff)
        analysis.decisions_data = self._filter_by_date(analysis.decisions_data, cutoff)
        analysis.signals_data = self._filter_by_date(analysis.signals_data, cutoff)

        # Build trade records for performance analysis
        trades = self._build_trade_records(analysis.orders_data, analysis.decisions_data)
        analysis.total_trades = len(trades)

        # Run performance analytics
        if trades:
            report = self.analytics.analyze_trades(trades)
            analysis.performance_report = report
            analysis.win_rate = report.win_rate
            analysis.profit_factor = report.profit_factor
            analysis.expectancy = report.expectancy
            analysis.max_drawdown_pct = report.max_drawdown_pct
            analysis.current_drawdown_pct = report.current_drawdown_pct
            analysis.consecutive_losses = abs(report.current_streak) if report.current_streak_type == "loss" else 0

            # Extract best/worst symbols
            if report.per_symbol:
                sorted_symbols = sorted(
                    report.per_symbol.items(),
                    key=lambda x: x[1].get("total_pnl", 0),
                    reverse=True,
                )
                analysis.best_symbols = [s[0] for s in sorted_symbols[:3] if s[1].get("total_pnl", 0) > 0]
                analysis.worst_symbols = [s[0] for s in sorted_symbols[-3:] if s[1].get("total_pnl", 0) < 0]

        # Get current portfolio metrics from Alpaca
        try:
            if self.client:
                account = self.client.get_account()
                positions = self.client.get_positions()

                equity = float(account.get("equity", 0))
                last_equity = float(account.get("last_equity", equity))

                if last_equity > 0:
                    analysis.daily_pnl_pct = (equity - last_equity) / last_equity

                # Calculate portfolio heat (total position value / equity)
                total_position_value = sum(
                    abs(float(p.get("market_value", 0))) for p in positions
                )
                if equity > 0:
                    analysis.portfolio_heat = total_position_value / equity

        except Exception as e:
            logger.warning("Failed to fetch account data: %s", e)

        # Analyze regime performance
        analysis.best_regime, analysis.worst_regime = self._analyze_regime_performance(
            analysis.decisions_data
        )

        # Detect issues
        analysis.issues.extend(self._detect_issues(analysis))

        self._last_analysis = analysis
        return analysis.__dict__

    def execute(self, symbol: str = "", analysis: dict | None = None, **kwargs) -> dict:
        """Generate optimization suggestions based on analysis.

        Args:
            symbol: Ignored
            analysis: Analysis dict from analyze() method
            **kwargs: Optional parameters

        Returns:
            Result dict with suggestions and health score
        """
        if analysis is None:
            analysis = self.analyze()

        # Convert dict back to dataclass if needed
        if isinstance(analysis, dict):
            analysis_obj = self._dict_to_analysis(analysis)
        else:
            analysis_obj = analysis

        result = OptimizerResult(
            timestamp=datetime.now().isoformat(),
            analysis=analysis_obj,
        )

        # Generate suggestions by category
        result.suggestions.extend(self._suggest_parameter_changes(analysis_obj))
        result.suggestions.extend(self._suggest_risk_adjustments(analysis_obj))
        result.suggestions.extend(self._suggest_symbol_changes(analysis_obj))
        result.suggestions.extend(self._suggest_strategy_changes(analysis_obj))

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        result.suggestions.sort(key=lambda s: priority_order.get(s.priority, 3))

        # Calculate health score (0-100)
        result.health_score = self._calculate_health_score(analysis_obj)

        # Generate summary
        result.summary = self._generate_summary(result)

        # Store in history
        self._suggestion_history.append(result)
        if len(self._suggestion_history) > 100:
            self._suggestion_history = self._suggestion_history[-100:]

        return {
            "timestamp": result.timestamp,
            "suggestions": [s.__dict__ for s in result.suggestions],
            "summary": result.summary,
            "health_score": result.health_score,
            "total_suggestions": len(result.suggestions),
            "high_priority": sum(1 for s in result.suggestions if s.priority == "high"),
            "medium_priority": sum(1 for s in result.suggestions if s.priority == "medium"),
            "low_priority": sum(1 for s in result.suggestions if s.priority == "low"),
        }

    # ── Suggestion Generators ────────────────────────────────────

    def _suggest_parameter_changes(self, analysis: OptimizerAnalysis) -> list[OptimizationSuggestion]:
        """Generate parameter optimization suggestions."""
        suggestions = []

        if analysis.total_trades < self.MIN_TRADES_FOR_ANALYSIS:
            return suggestions

        report = analysis.performance_report
        if not report:
            return suggestions

        # Win rate too low - suggest tighter entry criteria
        if report.win_rate < self.LOW_WIN_RATE:
            suggestions.append(OptimizationSuggestion(
                category="parameter",
                priority="high",
                title="Tighten Entry Thresholds",
                description=(
                    f"Win rate is {report.win_rate:.1%}, below target of {self.LOW_WIN_RATE:.0%}. "
                    "Consider raising buy_threshold to filter weaker signals."
                ),
                current_value=0.10,  # Current threshold from codebase
                suggested_value=0.20,
                expected_impact="Fewer trades but higher quality entries",
                confidence=0.75,
            ))

        # Win rate very high but low profit factor - let winners run longer
        if report.win_rate > self.HIGH_WIN_RATE and report.profit_factor < self.HIGH_PROFIT_FACTOR:
            suggestions.append(OptimizationSuggestion(
                category="parameter",
                priority="medium",
                title="Increase Take Profit Target",
                description=(
                    f"High win rate ({report.win_rate:.1%}) but profit factor only {report.profit_factor:.2f}. "
                    "Winners may be closing too early. Consider raising take_profit_pct."
                ),
                current_value=0.15,
                suggested_value=0.20,
                expected_impact="Larger average wins, improved profit factor",
                confidence=0.70,
            ))

        # Large average loss vs win - tighten stops
        if report.avg_loss != 0 and abs(report.avg_loss) > report.avg_win * 1.5:
            suggestions.append(OptimizationSuggestion(
                category="parameter",
                priority="high",
                title="Tighten Stop Loss",
                description=(
                    f"Average loss (${abs(report.avg_loss):,.2f}) is much larger than "
                    f"average win (${report.avg_win:,.2f}). Tighten trailing_stop_pct."
                ),
                current_value=0.08,
                suggested_value=0.06,
                expected_impact="Smaller losses, may slightly reduce win rate",
                confidence=0.80,
            ))

        # Low profit factor overall
        if report.profit_factor < self.LOW_PROFIT_FACTOR and report.total_trades >= 10:
            suggestions.append(OptimizationSuggestion(
                category="parameter",
                priority="high",
                title="Review Signal Weights",
                description=(
                    f"Profit factor is {report.profit_factor:.2f} (below 1.0 means losing money). "
                    "Consider adjusting TA/Sentiment/MTF weights or raising min confidence."
                ),
                current_value="TA:40%, Sentiment:20%, MTF:30%",
                suggested_value="TA:50%, Sentiment:15%, MTF:25%",
                expected_impact="Better signal quality from technical analysis",
                confidence=0.65,
            ))

        return suggestions

    def _suggest_risk_adjustments(self, analysis: OptimizerAnalysis) -> list[OptimizationSuggestion]:
        """Generate risk management suggestions."""
        suggestions = []

        # High drawdown
        if analysis.current_drawdown_pct > self.HIGH_DRAWDOWN:
            suggestions.append(OptimizationSuggestion(
                category="risk",
                priority="high",
                title="Reduce Position Sizes",
                description=(
                    f"Current drawdown is {analysis.current_drawdown_pct:.1%}, exceeding "
                    f"{self.HIGH_DRAWDOWN:.0%} threshold. Reduce max_position_pct to limit exposure."
                ),
                current_value=0.30,
                suggested_value=0.15,
                expected_impact="Smaller individual losses, slower recovery but more stable",
                confidence=0.85,
            ))

        # Consecutive losses
        if analysis.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            suggestions.append(OptimizationSuggestion(
                category="risk",
                priority="high",
                title="Pause Trading / Review Strategy",
                description=(
                    f"{analysis.consecutive_losses} consecutive losses detected. "
                    "Consider pausing to review market conditions and strategy fit."
                ),
                expected_impact="Prevent further losses during unfavorable conditions",
                confidence=0.90,
            ))

        # High portfolio heat
        if analysis.portfolio_heat > 0.80:
            suggestions.append(OptimizationSuggestion(
                category="risk",
                priority="medium",
                title="Reduce Portfolio Concentration",
                description=(
                    f"Portfolio heat is {analysis.portfolio_heat:.0%} of equity. "
                    "Consider closing some positions or avoiding new entries."
                ),
                current_value=analysis.portfolio_heat,
                suggested_value=0.60,
                expected_impact="Lower overall portfolio risk",
                confidence=0.75,
            ))

        # Bad daily P&L
        if analysis.daily_pnl_pct < -0.03:
            suggestions.append(OptimizationSuggestion(
                category="risk",
                priority="high",
                title="Daily Loss Limit Approaching",
                description=(
                    f"Daily P&L is {analysis.daily_pnl_pct:.1%}. "
                    "Consider halting new trades for today to preserve capital."
                ),
                expected_impact="Prevent emotional/revenge trading",
                confidence=0.85,
            ))

        return suggestions

    def _suggest_symbol_changes(self, analysis: OptimizerAnalysis) -> list[OptimizationSuggestion]:
        """Generate symbol-specific suggestions."""
        suggestions = []

        report = analysis.performance_report
        if not report or not report.per_symbol:
            return suggestions

        # Identify consistently losing symbols
        for sym, stats in report.per_symbol.items():
            if stats["trades"] >= 3 and stats["win_rate"] < 0.30:
                suggestions.append(OptimizationSuggestion(
                    category="symbol",
                    priority="medium",
                    title=f"Consider Removing {sym}",
                    description=(
                        f"{sym} has {stats['win_rate']:.0%} win rate over {stats['trades']} trades "
                        f"with total P&L of ${stats['total_pnl']:,.2f}. "
                        "Consider removing from watchlist."
                    ),
                    expected_impact=f"Avoid ~${abs(stats['total_pnl']):,.2f} in potential losses",
                    confidence=0.70,
                ))

        # Highlight best performers
        if analysis.best_symbols:
            suggestions.append(OptimizationSuggestion(
                category="symbol",
                priority="low",
                title="Focus on Top Performers",
                description=(
                    f"Best performing symbols: {', '.join(analysis.best_symbols)}. "
                    "Consider increasing position sizes for these or adding similar stocks."
                ),
                expected_impact="Potentially higher returns from proven winners",
                confidence=0.60,
            ))

        return suggestions

    def _suggest_strategy_changes(self, analysis: OptimizerAnalysis) -> list[OptimizationSuggestion]:
        """Generate strategy-level suggestions."""
        suggestions = []

        report = analysis.performance_report
        if not report:
            return suggestions

        # Negative expectancy
        if report.expectancy < self.LOW_EXPECTANCY and analysis.total_trades >= 10:
            suggestions.append(OptimizationSuggestion(
                category="strategy",
                priority="high",
                title="Strategy Has Negative Expectancy",
                description=(
                    f"Expected value per trade is ${report.expectancy:,.2f}. "
                    "The current strategy is expected to lose money over time. "
                    "Major parameter review or strategy change recommended."
                ),
                expected_impact="Critical - current approach is unprofitable",
                confidence=0.90,
            ))

        # Low Sharpe ratio
        if report.sharpe_ratio < self.LOW_SHARPE and analysis.total_trades >= 20:
            suggestions.append(OptimizationSuggestion(
                category="strategy",
                priority="medium",
                title="Poor Risk-Adjusted Returns",
                description=(
                    f"Sharpe ratio is {report.sharpe_ratio:.2f}, indicating poor "
                    "risk-adjusted performance. Consider reducing trade frequency "
                    "or increasing signal quality requirements."
                ),
                expected_impact="Better returns per unit of risk taken",
                confidence=0.65,
            ))

        # Regime-specific issues
        if analysis.worst_regime:
            suggestions.append(OptimizationSuggestion(
                category="strategy",
                priority="medium",
                title=f"Avoid Trading in {analysis.worst_regime} Regime",
                description=(
                    f"Performance is worst during {analysis.worst_regime} market conditions. "
                    "Consider reducing position sizes or skipping trades in this regime."
                ),
                expected_impact="Fewer losses during unfavorable market conditions",
                confidence=0.70,
            ))

        # Too many trades (overtrading)
        if analysis.total_trades > 50 and report.win_rate < 0.50:
            suggestions.append(OptimizationSuggestion(
                category="strategy",
                priority="medium",
                title="Potential Overtrading",
                description=(
                    f"{analysis.total_trades} trades with {report.win_rate:.0%} win rate "
                    "suggests overtrading. Consider stricter entry criteria to focus on "
                    "higher-quality setups."
                ),
                current_value=analysis.total_trades,
                suggested_value="Reduce by 30-50%",
                expected_impact="Higher win rate, lower transaction costs",
                confidence=0.70,
            ))

        return suggestions

    # ── Helper Methods ───────────────────────────────────────────

    def _filter_by_date(self, data: list[dict], cutoff: datetime) -> list[dict]:
        """Filter records by timestamp."""
        filtered = []
        for row in data:
            ts_str = row.get("timestamp", "")
            if not ts_str:
                continue
            try:
                # Handle various timestamp formats
                ts_str = ts_str.replace("+00:00", "").replace("Z", "")
                if "T" in ts_str:
                    ts = datetime.fromisoformat(ts_str)
                else:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                if ts >= cutoff:
                    filtered.append(row)
            except (ValueError, TypeError):
                filtered.append(row)  # Keep if can't parse
        return filtered

    def _build_trade_records(
        self, orders: list[dict], decisions: list[dict]
    ) -> list[TradeRecord]:
        """Build TradeRecord objects from journal data."""
        trades = []

        # Create a lookup for decisions by symbol+timestamp
        decision_lookup = {}
        for d in decisions:
            key = (d.get("symbol", ""), d.get("timestamp", "")[:10])  # Date only
            decision_lookup[key] = d

        for order in orders:
            status = order.get("status", "")
            if status not in ("filled", "dry_run"):
                continue

            symbol = order.get("symbol", "")
            qty = int(order.get("qty", 0) or 0)
            if qty == 0:
                continue

            # Try to get P&L from order or estimate from decision
            pnl = float(order.get("pnl", 0) or 0)
            pnl_pct = float(order.get("pnl_pct", 0) or 0)

            trades.append(TradeRecord(
                symbol=symbol,
                side=order.get("side", ""),
                qty=qty,
                price=float(order.get("filled_avg_price", 0) or order.get("price", 0) or 0),
                pnl=pnl,
                pnl_pct=pnl_pct,
                timestamp=order.get("timestamp", ""),
                is_win=pnl > 0,
            ))

        return trades

    def _analyze_regime_performance(self, decisions: list[dict]) -> tuple[str, str]:
        """Find best and worst performing regimes."""
        regime_stats: dict[str, dict] = {}

        for d in decisions:
            regime = d.get("regime", "UNKNOWN")
            if not regime:
                regime = "UNKNOWN"

            if regime not in regime_stats:
                regime_stats[regime] = {"count": 0, "score_sum": 0.0}

            regime_stats[regime]["count"] += 1
            regime_stats[regime]["score_sum"] += float(d.get("combined_score", 0) or 0)

        if not regime_stats:
            return "", ""

        # Calculate average score per regime
        for regime, stats in regime_stats.items():
            if stats["count"] > 0:
                stats["avg_score"] = stats["score_sum"] / stats["count"]
            else:
                stats["avg_score"] = 0

        sorted_regimes = sorted(
            regime_stats.items(),
            key=lambda x: x[1]["avg_score"],
            reverse=True,
        )

        best = sorted_regimes[0][0] if sorted_regimes else ""
        worst = sorted_regimes[-1][0] if len(sorted_regimes) > 1 else ""

        return best, worst

    def _detect_issues(self, analysis: OptimizerAnalysis) -> list[str]:
        """Detect critical issues that need attention."""
        issues = []

        if analysis.total_trades == 0:
            issues.append("No trades found in analysis period")
        elif analysis.total_trades < self.MIN_TRADES_FOR_ANALYSIS:
            issues.append(f"Only {analysis.total_trades} trades - need {self.MIN_TRADES_FOR_ANALYSIS}+ for reliable analysis")

        if analysis.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            issues.append(f"CRITICAL: {analysis.consecutive_losses} consecutive losses")

        if analysis.current_drawdown_pct > self.HIGH_DRAWDOWN:
            issues.append(f"HIGH DRAWDOWN: {analysis.current_drawdown_pct:.1%}")

        if analysis.daily_pnl_pct < -0.04:
            issues.append(f"DAILY LOSS LIMIT: {analysis.daily_pnl_pct:.1%}")

        report = analysis.performance_report
        if report and report.profit_factor < 1.0 and analysis.total_trades >= 10:
            issues.append(f"UNPROFITABLE: Profit factor {report.profit_factor:.2f}")

        return issues

    def _calculate_health_score(self, analysis: OptimizerAnalysis) -> float:
        """Calculate overall pipeline health (0-100)."""
        score = 50.0  # Base score

        report = analysis.performance_report
        if not report or analysis.total_trades < self.MIN_TRADES_FOR_ANALYSIS:
            return score  # Neutral if insufficient data

        # Win rate contribution (+/- 15 points)
        if report.win_rate >= 0.55:
            score += 15
        elif report.win_rate >= 0.45:
            score += 5
        elif report.win_rate < 0.35:
            score -= 15
        else:
            score -= 5

        # Profit factor contribution (+/- 15 points)
        if report.profit_factor >= 2.0:
            score += 15
        elif report.profit_factor >= 1.5:
            score += 10
        elif report.profit_factor >= 1.0:
            score += 5
        elif report.profit_factor < 0.8:
            score -= 15
        else:
            score -= 5

        # Drawdown contribution (+/- 10 points)
        if analysis.current_drawdown_pct < 0.03:
            score += 10
        elif analysis.current_drawdown_pct < 0.08:
            score += 5
        elif analysis.current_drawdown_pct > 0.15:
            score -= 10

        # Consecutive losses (-5 per loss after 3)
        if analysis.consecutive_losses > 3:
            score -= (analysis.consecutive_losses - 3) * 5

        # Expectancy contribution (+/- 10 points)
        if report.expectancy > 50:
            score += 10
        elif report.expectancy > 0:
            score += 5
        elif report.expectancy < -50:
            score -= 10

        return max(0, min(100, score))

    def _generate_summary(self, result: OptimizerResult) -> str:
        """Generate a human-readable summary."""
        analysis = result.analysis
        if not analysis:
            return "No analysis available"

        high_priority = sum(1 for s in result.suggestions if s.priority == "high")

        health_status = (
            "HEALTHY" if result.health_score >= 70 else
            "NEEDS ATTENTION" if result.health_score >= 50 else
            "CRITICAL"
        )

        lines = [
            f"Pipeline Health: {health_status} ({result.health_score:.0f}/100)",
            f"Trades Analyzed: {analysis.total_trades}",
        ]

        if analysis.performance_report:
            report = analysis.performance_report
            lines.extend([
                f"Win Rate: {report.win_rate:.1%}",
                f"Profit Factor: {report.profit_factor:.2f}",
            ])

        if high_priority > 0:
            lines.append(f"HIGH PRIORITY ISSUES: {high_priority}")

        if analysis.issues:
            lines.append(f"Detected Issues: {', '.join(analysis.issues[:3])}")

        return " | ".join(lines)

    def _dict_to_analysis(self, d: dict) -> OptimizerAnalysis:
        """Convert analysis dict back to dataclass."""
        analysis = OptimizerAnalysis()
        for key, value in d.items():
            if hasattr(analysis, key):
                setattr(analysis, key, value)
        return analysis

    # ── Continuous Monitoring ────────────────────────────────────

    def monitor(
        self,
        interval: int = 300,
        callback: callable = None,
        max_iterations: int | None = None,
    ):
        """Run continuous monitoring loop.

        Args:
            interval: Seconds between checks
            callback: Optional function(result_dict) called each iteration
            max_iterations: Stop after N iterations (None = forever)
        """
        logger.info(
            "Starting pipeline optimizer monitoring (interval=%ds)",
            interval,
        )

        iteration = 0
        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                logger.info("Optimizer check #%d", iteration)

                try:
                    analysis = self.analyze()
                    result = self.execute(analysis=analysis)

                    if callback:
                        callback(result)
                    else:
                        # Default: print formatted output
                        print(self.format_suggestions(result))

                    # Log high-priority suggestions
                    high_priority = [
                        s for s in result.get("suggestions", [])
                        if s.get("priority") == "high"
                    ]
                    if high_priority:
                        logger.warning(
                            "HIGH PRIORITY: %d suggestions require attention",
                            len(high_priority),
                        )

                except Exception as e:
                    logger.error("Optimizer iteration failed: %s", e)

                if max_iterations is None or iteration < max_iterations:
                    logger.info("Next check in %d seconds...", interval)
                    time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Optimizer monitoring stopped by user")

    # ── Formatting ───────────────────────────────────────────────

    @staticmethod
    def format_suggestions(result: dict) -> str:
        """Format suggestions for display."""
        lines = [
            f"\n{'=' * 70}",
            f"  PIPELINE OPTIMIZER REPORT",
            f"{'=' * 70}",
            f"  Timestamp: {result.get('timestamp', 'N/A')}",
            f"  Health Score: {result.get('health_score', 0):.0f}/100",
            f"  Summary: {result.get('summary', 'N/A')}",
            f"{'~' * 70}",
        ]

        suggestions = result.get("suggestions", [])
        if not suggestions:
            lines.append("  No suggestions at this time.")
        else:
            lines.append(f"  {len(suggestions)} SUGGESTIONS:")
            lines.append("")

            for i, s in enumerate(suggestions, 1):
                priority_icon = {
                    "high": "[!]",
                    "medium": "[*]",
                    "low": "[-]",
                }.get(s.get("priority", ""), "[ ]")

                lines.append(f"  {i}. {priority_icon} {s.get('title', 'Untitled')}")
                lines.append(f"     Category: {s.get('category', 'N/A')}")
                lines.append(f"     {s.get('description', '')}")

                if s.get("current_value") is not None:
                    lines.append(f"     Current: {s.get('current_value')}")
                if s.get("suggested_value") is not None:
                    lines.append(f"     Suggested: {s.get('suggested_value')}")
                if s.get("expected_impact"):
                    lines.append(f"     Impact: {s.get('expected_impact')}")
                if s.get("confidence"):
                    lines.append(f"     Confidence: {s.get('confidence'):.0%}")
                lines.append("")

        lines.append(f"{'=' * 70}\n")
        return "\n".join(lines)

    @staticmethod
    def format_analysis(analysis: dict) -> str:
        """Format analysis for display."""
        lines = [
            f"\n{'=' * 70}",
            f"  PIPELINE ANALYSIS",
            f"{'=' * 70}",
            f"  Period: Last {analysis.get('lookback_days', 30)} days",
            f"  Total Trades: {analysis.get('total_trades', 0)}",
            f"{'~' * 70}",
        ]

        lines.extend([
            f"\n  Performance Metrics:",
            f"  {'~' * 60}",
            f"  Win Rate       : {analysis.get('win_rate', 0):.1%}",
            f"  Profit Factor  : {analysis.get('profit_factor', 0):.2f}",
            f"  Expectancy     : ${analysis.get('expectancy', 0):,.2f}",
            f"  Max Drawdown   : {analysis.get('max_drawdown_pct', 0):.1%}",
            f"  Current DD     : {analysis.get('current_drawdown_pct', 0):.1%}",
        ])

        lines.extend([
            f"\n  Current Status:",
            f"  {'~' * 60}",
            f"  Daily P&L      : {analysis.get('daily_pnl_pct', 0):+.2%}",
            f"  Portfolio Heat : {analysis.get('portfolio_heat', 0):.0%}",
            f"  Consec Losses  : {analysis.get('consecutive_losses', 0)}",
        ])

        if analysis.get("best_symbols"):
            lines.append(f"  Best Symbols   : {', '.join(analysis['best_symbols'])}")
        if analysis.get("worst_symbols"):
            lines.append(f"  Worst Symbols  : {', '.join(analysis['worst_symbols'])}")
        if analysis.get("best_regime"):
            lines.append(f"  Best Regime    : {analysis['best_regime']}")

        issues = analysis.get("issues", [])
        if issues:
            lines.extend([
                f"\n  Issues Detected:",
                f"  {'~' * 60}",
            ])
            for issue in issues:
                lines.append(f"  - {issue}")

        lines.append(f"\n{'=' * 70}\n")
        return "\n".join(lines)
