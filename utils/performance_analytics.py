"""Performance Analytics Engine

Computes advanced trading performance metrics from the trade journal,
providing time-series analysis of returns, risk-adjusted ratios, and
trade quality statistics.

Metrics:
    - Total / annualised return
    - Rolling Sharpe ratio (annualised)
    - Sortino ratio
    - Max drawdown & current drawdown
    - Win rate, profit factor, expectancy
    - Average win / loss, largest win / loss
    - Consecutive win / loss streaks

Usage:
    engine = PerformanceAnalytics()
    report = engine.analyze(orders, decisions, equity_curve)
    print(PerformanceAnalytics.format_report(report))
"""

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Normalised trade record extracted from journal orders."""
    symbol: str
    side: str
    qty: int
    price: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    timestamp: str = ""
    is_win: bool = False


@dataclass
class PerformanceReport:
    """Full performance analytics report."""
    # Returns
    total_return_pct: float = 0.0
    annualised_return_pct: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Drawdown
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0

    # Trade quality
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # Win / loss details
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0

    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0         # positive = wins, negative = losses
    current_streak_type: str = ""   # "win" or "loss" or ""

    # Equity
    peak_equity: float = 0.0
    final_equity: float = 0.0
    trading_days: int = 0

    # Per-symbol breakdown
    per_symbol: dict = field(default_factory=dict)

    @property
    def risk_reward_ratio(self) -> float:
        if self.avg_loss == 0:
            return 0.0
        return abs(self.avg_win / self.avg_loss)


class PerformanceAnalytics:
    """Compute advanced performance metrics from trading data."""

    def __init__(self, *, risk_free_rate: float = 0.05, trading_days_year: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days_year = trading_days_year

    # ── Core computation ─────────────────────────────────────────

    def analyze_trades(self, trades: list[TradeRecord]) -> PerformanceReport:
        """Analyze a list of trade records."""
        report = PerformanceReport()

        if not trades:
            return report

        report.total_trades = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        report.winning_trades = len(wins)
        report.losing_trades = len(losses)
        report.win_rate = len(wins) / len(trades) if trades else 0.0

        # P&L aggregates
        gross_wins = sum(t.pnl for t in wins)
        gross_losses = abs(sum(t.pnl for t in losses))

        report.profit_factor = (
            gross_wins / gross_losses if gross_losses > 0 else
            float("inf") if gross_wins > 0 else 0.0
        )

        # Averages
        report.avg_win = gross_wins / len(wins) if wins else 0.0
        report.avg_loss = -gross_losses / len(losses) if losses else 0.0
        report.avg_win_pct = (
            sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0
        )
        report.avg_loss_pct = (
            sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0
        )

        # Expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
        report.expectancy = (
            report.win_rate * report.avg_win +
            (1 - report.win_rate) * report.avg_loss
        )

        # Extremes
        all_pnls = [t.pnl for t in trades]
        report.largest_win = max(all_pnls) if all_pnls else 0.0
        report.largest_loss = min(all_pnls) if all_pnls else 0.0

        # Streaks
        self._compute_streaks(trades, report)

        # Per-symbol breakdown
        self._compute_per_symbol(trades, report)

        return report

    def analyze_equity_curve(
        self,
        equity_values: list[float],
        initial_capital: float = 100000.0,
    ) -> PerformanceReport:
        """Analyze an equity curve (list of daily equity values)."""
        report = PerformanceReport()

        if not equity_values or len(equity_values) < 2:
            report.final_equity = equity_values[-1] if equity_values else initial_capital
            return report

        report.trading_days = len(equity_values)
        report.final_equity = equity_values[-1]
        report.peak_equity = max(equity_values)

        # Returns
        total_ret = (equity_values[-1] - initial_capital) / initial_capital
        report.total_return_pct = total_ret * 100

        years = report.trading_days / self.trading_days_year
        if years > 0:
            report.annualised_return_pct = (
                ((1 + total_ret) ** (1 / years) - 1) * 100
                if total_ret > -1 else -100.0
            )

        # Daily returns
        daily_returns = []
        for i in range(1, len(equity_values)):
            if equity_values[i - 1] > 0:
                daily_returns.append(
                    (equity_values[i] - equity_values[i - 1]) / equity_values[i - 1]
                )

        if daily_returns:
            # Sharpe ratio
            report.sharpe_ratio = self._sharpe(daily_returns)
            # Sortino ratio
            report.sortino_ratio = self._sortino(daily_returns)

        # Drawdown
        report.max_drawdown_pct, report.current_drawdown_pct = self._drawdowns(
            equity_values
        )

        return report

    def analyze_full(
        self,
        trades: list[TradeRecord],
        equity_values: list[float] | None = None,
        initial_capital: float = 100000.0,
    ) -> PerformanceReport:
        """Combined trade + equity curve analysis."""
        trade_report = self.analyze_trades(trades)

        if equity_values and len(equity_values) >= 2:
            eq_report = self.analyze_equity_curve(equity_values, initial_capital)
            # Merge equity curve metrics into trade report
            trade_report.total_return_pct = eq_report.total_return_pct
            trade_report.annualised_return_pct = eq_report.annualised_return_pct
            trade_report.sharpe_ratio = eq_report.sharpe_ratio
            trade_report.sortino_ratio = eq_report.sortino_ratio
            trade_report.max_drawdown_pct = eq_report.max_drawdown_pct
            trade_report.current_drawdown_pct = eq_report.current_drawdown_pct
            trade_report.peak_equity = eq_report.peak_equity
            trade_report.final_equity = eq_report.final_equity
            trade_report.trading_days = eq_report.trading_days

        return trade_report

    # ── Helpers ──────────────────────────────────────────────────

    def _sharpe(self, daily_returns: list[float]) -> float:
        if len(daily_returns) < 2:
            return 0.0
        mean_r = sum(daily_returns) / len(daily_returns)
        std_r = self._std(daily_returns)
        if std_r == 0:
            return 0.0
        daily_rf = self.risk_free_rate / self.trading_days_year
        return (mean_r - daily_rf) / std_r * math.sqrt(self.trading_days_year)

    def _sortino(self, daily_returns: list[float]) -> float:
        if len(daily_returns) < 2:
            return 0.0
        mean_r = sum(daily_returns) / len(daily_returns)
        daily_rf = self.risk_free_rate / self.trading_days_year
        downside = [r for r in daily_returns if r < daily_rf]
        if not downside:
            return float("inf") if mean_r > daily_rf else 0.0
        down_dev = math.sqrt(
            sum((r - daily_rf) ** 2 for r in downside) / len(downside)
        )
        if down_dev == 0:
            return 0.0
        return (mean_r - daily_rf) / down_dev * math.sqrt(self.trading_days_year)

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    @staticmethod
    def _drawdowns(equity: list[float]) -> tuple[float, float]:
        """Return (max_drawdown_pct, current_drawdown_pct)."""
        if not equity:
            return 0.0, 0.0
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        current_dd = (peak - equity[-1]) / peak if peak > 0 else 0.0
        return max_dd * 100, current_dd * 100

    @staticmethod
    def _compute_streaks(trades: list[TradeRecord], report: PerformanceReport):
        max_wins = 0
        max_losses = 0
        current = 0
        for t in trades:
            if t.pnl > 0:
                if current > 0:
                    current += 1
                else:
                    current = 1
                max_wins = max(max_wins, current)
            else:
                if current < 0:
                    current -= 1
                else:
                    current = -1
                max_losses = max(max_losses, abs(current))

        report.max_consecutive_wins = max_wins
        report.max_consecutive_losses = max_losses
        report.current_streak = current
        report.current_streak_type = "win" if current > 0 else ("loss" if current < 0 else "")

    @staticmethod
    def _compute_per_symbol(trades: list[TradeRecord], report: PerformanceReport):
        by_sym: dict[str, dict] = {}
        for t in trades:
            if t.symbol not in by_sym:
                by_sym[t.symbol] = {
                    "trades": 0, "wins": 0, "total_pnl": 0.0,
                    "gross_wins": 0.0, "gross_losses": 0.0,
                }
            s = by_sym[t.symbol]
            s["trades"] += 1
            s["total_pnl"] += t.pnl
            if t.pnl > 0:
                s["wins"] += 1
                s["gross_wins"] += t.pnl
            else:
                s["gross_losses"] += abs(t.pnl)

        for sym, s in by_sym.items():
            s["win_rate"] = s["wins"] / s["trades"] if s["trades"] else 0
            s["profit_factor"] = (
                s["gross_wins"] / s["gross_losses"]
                if s["gross_losses"] > 0 else float("inf")
            )
        report.per_symbol = by_sym

    # ── Journal parsing ──────────────────────────────────────────

    @staticmethod
    def trades_from_orders(orders: list[dict]) -> list[TradeRecord]:
        """Convert journal order rows into TradeRecord objects.

        Note: This is an approximation since the journal doesn't track
        exit prices. Each filled order becomes one trade record.
        """
        trades = []
        for o in orders:
            status = o.get("status", "")
            if status not in ("filled", "dry_run"):
                continue
            qty = int(o.get("qty", 0))
            if qty == 0:
                continue
            trades.append(TradeRecord(
                symbol=o.get("symbol", ""),
                side=o.get("side", ""),
                qty=qty,
                price=0.0,  # not tracked in basic journal
                pnl=0.0,
                timestamp=o.get("timestamp", ""),
            ))
        return trades

    # ── Formatting ───────────────────────────────────────────────

    @staticmethod
    def format_report(report: PerformanceReport) -> str:
        lines = [
            f"\n{'=' * 66}",
            f"  PERFORMANCE ANALYTICS",
            f"{'=' * 66}",
        ]

        # Returns
        lines.append(f"\n  Returns")
        lines.append(f"  {'~' * 60}")
        lines.append(f"  Total Return      : {report.total_return_pct:+.2f}%")
        lines.append(f"  Annualised Return : {report.annualised_return_pct:+.2f}%")

        # Risk-adjusted
        lines.append(f"\n  Risk-Adjusted Metrics")
        lines.append(f"  {'~' * 60}")
        lines.append(f"  Sharpe Ratio      : {report.sharpe_ratio:.3f}")
        lines.append(f"  Sortino Ratio     : {report.sortino_ratio:.3f}")
        lines.append(f"  Max Drawdown      : {report.max_drawdown_pct:.2f}%")
        lines.append(f"  Current Drawdown  : {report.current_drawdown_pct:.2f}%")

        # Trade quality
        lines.append(f"\n  Trade Quality")
        lines.append(f"  {'~' * 60}")
        lines.append(f"  Total Trades      : {report.total_trades}")
        lines.append(f"  Win Rate          : {report.win_rate:.1%}")
        lines.append(f"  Profit Factor     : {report.profit_factor:.2f}")
        lines.append(f"  Expectancy        : ${report.expectancy:,.2f}")
        lines.append(f"  Risk/Reward       : {report.risk_reward_ratio:.2f}")

        # Win / loss
        lines.append(f"\n  Win / Loss Detail")
        lines.append(f"  {'~' * 60}")
        lines.append(f"  Avg Win           : ${report.avg_win:,.2f} ({report.avg_win_pct:+.2f}%)")
        lines.append(f"  Avg Loss          : ${report.avg_loss:,.2f} ({report.avg_loss_pct:+.2f}%)")
        lines.append(f"  Largest Win       : ${report.largest_win:,.2f}")
        lines.append(f"  Largest Loss      : ${report.largest_loss:,.2f}")

        # Streaks
        lines.append(f"\n  Streaks")
        lines.append(f"  {'~' * 60}")
        lines.append(f"  Max Consec Wins   : {report.max_consecutive_wins}")
        lines.append(f"  Max Consec Losses : {report.max_consecutive_losses}")
        if report.current_streak != 0:
            lines.append(
                f"  Current Streak    : {abs(report.current_streak)} "
                f"{report.current_streak_type}(s)"
            )

        # Per-symbol
        if report.per_symbol:
            lines.append(f"\n  Per-Symbol Breakdown")
            lines.append(f"  {'~' * 60}")
            lines.append(
                f"  {'Symbol':8s} {'Trades':>7s} {'Win%':>7s} {'PnL':>12s} {'PF':>7s}"
            )
            lines.append(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*12} {'-'*7}")
            for sym in sorted(report.per_symbol):
                s = report.per_symbol[sym]
                lines.append(
                    f"  {sym:8s} {s['trades']:>7d} {s['win_rate']:>6.1%} "
                    f"${s['total_pnl']:>11,.2f} {s['profit_factor']:>7.2f}"
                )

        lines.append(f"\n{'=' * 66}\n")
        return "\n".join(lines)
