"""Profit Monitor

Real-time P&L tracking and profit protection for all positions.

Features:
    - Track unrealized P&L for each position
    - Monitor daily P&L across portfolio
    - Track peak prices for trailing stops
    - Calculate profit percentages and targets
    - Trigger alerts on profit/loss thresholds
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


@dataclass
class PositionMetrics:
    """Metrics for a single position."""
    symbol: str
    qty: int
    side: str  # "long" or "short"
    entry_price: float
    current_price: float
    peak_price: float  # Highest price since entry (for trailing stops)
    trough_price: float  # Lowest price since entry

    # P&L
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # Targets
    take_profit_price: float = 0.0
    stop_loss_price: float = 0.0
    trailing_stop_price: float = 0.0

    # Status
    hit_take_profit: bool = False
    hit_stop_loss: bool = False
    hit_trailing_stop: bool = False
    hit_max_loss: bool = False  # Hit $50 max loss limit

    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioMetrics:
    """Aggregate portfolio metrics."""
    total_equity: float = 0.0
    total_market_value: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_unrealized_pnl_pct: float = 0.0

    # Daily tracking
    day_start_equity: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0

    # Peak tracking (for drawdown)
    peak_equity: float = 0.0
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0

    # Counts
    total_positions: int = 0
    profitable_positions: int = 0
    losing_positions: int = 0

    positions: dict[str, PositionMetrics] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class ProfitMonitor:
    """Monitors positions and portfolio P&L in real-time."""

    # Default profit/loss thresholds (optimized for profitability)
    DEFAULT_TAKE_PROFIT_PCT = 0.15  # 15% profit target (let winners run)
    DEFAULT_STOP_LOSS_PCT = 0.07    # 7% stop loss (allow room)
    DEFAULT_TRAILING_STOP_PCT = 0.08  # 8% trailing stop from peak (reduce whipsaws)
    DEFAULT_DAILY_LOSS_LIMIT_PCT = 0.04  # 4% daily loss limit
    DEFAULT_MAX_DRAWDOWN_PCT = 0.12  # 12% max drawdown
    DEFAULT_MAX_LOSS_PER_POSITION = 50.0  # $50 max loss per position

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
        trailing_stop_pct: float = DEFAULT_TRAILING_STOP_PCT,
        daily_loss_limit_pct: float = DEFAULT_DAILY_LOSS_LIMIT_PCT,
        max_drawdown_pct: float = DEFAULT_MAX_DRAWDOWN_PCT,
        max_loss_per_position: float = DEFAULT_MAX_LOSS_PER_POSITION,
    ):
        self.client = client or AlpacaClient()
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_loss_per_position = max_loss_per_position

        # State tracking
        self._position_peaks: dict[str, float] = {}  # symbol -> peak price
        self._position_troughs: dict[str, float] = {}  # symbol -> trough price
        self._position_entries: dict[str, float] = {}  # symbol -> entry price
        self._day_start_equity: float = 0.0
        self._peak_equity: float = 0.0
        self._current_date: date = date.today()

    def _reset_daily_tracking(self, equity: float) -> None:
        """Reset daily tracking at start of new day."""
        today = date.today()
        if today != self._current_date:
            self._current_date = today
            self._day_start_equity = equity
            logger.info("New trading day - reset daily P&L tracking. Start equity: $%.2f", equity)

    def _update_peak_equity(self, equity: float) -> None:
        """Update peak equity for drawdown tracking."""
        if equity > self._peak_equity:
            self._peak_equity = equity
            logger.debug("New peak equity: $%.2f", equity)

    def _calculate_position_metrics(
        self,
        symbol: str,
        qty: int,
        side: str,
        entry_price: float,
        current_price: float,
    ) -> PositionMetrics:
        """Calculate metrics for a single position."""

        # Update peak/trough tracking
        if symbol not in self._position_peaks:
            self._position_peaks[symbol] = current_price
            self._position_troughs[symbol] = current_price
            self._position_entries[symbol] = entry_price
        else:
            if current_price > self._position_peaks[symbol]:
                self._position_peaks[symbol] = current_price
            if current_price < self._position_troughs[symbol]:
                self._position_troughs[symbol] = current_price

        peak_price = self._position_peaks[symbol]
        trough_price = self._position_troughs[symbol]

        # Calculate unrealized P&L
        if side == "long":
            unrealized_pnl = (current_price - entry_price) * qty
            unrealized_pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0

            # Calculate target prices
            take_profit_price = entry_price * (1 + self.take_profit_pct)
            stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            trailing_stop_price = peak_price * (1 - self.trailing_stop_pct)

            # Check triggers
            hit_take_profit = current_price >= take_profit_price
            hit_stop_loss = current_price <= stop_loss_price
            hit_trailing_stop = current_price <= trailing_stop_price and peak_price > entry_price
        else:  # short
            unrealized_pnl = (entry_price - current_price) * qty
            unrealized_pnl_pct = (entry_price - current_price) / entry_price if entry_price > 0 else 0

            # For shorts, targets are inverted
            take_profit_price = entry_price * (1 - self.take_profit_pct)
            stop_loss_price = entry_price * (1 + self.stop_loss_pct)
            trailing_stop_price = trough_price * (1 + self.trailing_stop_pct)

            hit_take_profit = current_price <= take_profit_price
            hit_stop_loss = current_price >= stop_loss_price
            hit_trailing_stop = current_price >= trailing_stop_price and trough_price < entry_price

        # Check max dollar loss per position ($50 default)
        hit_max_loss = unrealized_pnl <= -self.max_loss_per_position

        return PositionMetrics(
            symbol=symbol,
            qty=qty,
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            peak_price=peak_price,
            trough_price=trough_price,
            unrealized_pnl=round(unrealized_pnl, 2),
            unrealized_pnl_pct=round(unrealized_pnl_pct, 4),
            take_profit_price=round(take_profit_price, 2),
            stop_loss_price=round(stop_loss_price, 2),
            trailing_stop_price=round(trailing_stop_price, 2),
            hit_take_profit=hit_take_profit,
            hit_stop_loss=hit_stop_loss,
            hit_trailing_stop=hit_trailing_stop,
            hit_max_loss=hit_max_loss,
        )

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get current portfolio metrics with all P&L calculations."""

        # Fetch account and positions
        account = self.client.get_account()
        positions = self.client.get_positions()

        equity = account["equity"]

        # Reset daily tracking if new day
        if self._day_start_equity == 0:
            self._day_start_equity = equity
        self._reset_daily_tracking(equity)
        self._update_peak_equity(equity)

        # Calculate position metrics
        position_metrics: dict[str, PositionMetrics] = {}
        total_unrealized_pnl = 0.0
        profitable = 0
        losing = 0

        for pos in positions:
            symbol = pos["symbol"]
            qty = pos["qty"]
            entry_price = pos["avg_entry_price"]
            current_price = pos["current_price"]
            market_value = pos["market_value"]

            # Determine side from market value sign
            side = "long" if market_value > 0 else "short"

            metrics = self._calculate_position_metrics(
                symbol, abs(qty), side, entry_price, current_price
            )
            position_metrics[symbol] = metrics
            total_unrealized_pnl += metrics.unrealized_pnl

            if metrics.unrealized_pnl > 0:
                profitable += 1
            elif metrics.unrealized_pnl < 0:
                losing += 1

        # Calculate portfolio-level metrics
        total_market_value = sum(abs(p["market_value"]) for p in positions)
        daily_pnl = equity - self._day_start_equity
        daily_pnl_pct = daily_pnl / self._day_start_equity if self._day_start_equity > 0 else 0

        current_drawdown = self._peak_equity - equity
        drawdown_pct = current_drawdown / self._peak_equity if self._peak_equity > 0 else 0

        unrealized_pnl_pct = total_unrealized_pnl / equity if equity > 0 else 0

        return PortfolioMetrics(
            total_equity=equity,
            total_market_value=total_market_value,
            total_unrealized_pnl=round(total_unrealized_pnl, 2),
            total_unrealized_pnl_pct=round(unrealized_pnl_pct, 4),
            day_start_equity=self._day_start_equity,
            daily_pnl=round(daily_pnl, 2),
            daily_pnl_pct=round(daily_pnl_pct, 4),
            peak_equity=self._peak_equity,
            current_drawdown=round(current_drawdown, 2),
            current_drawdown_pct=round(drawdown_pct, 4),
            total_positions=len(positions),
            profitable_positions=profitable,
            losing_positions=losing,
            positions=position_metrics,
        )

    def get_positions_to_close(self) -> list[dict]:
        """Get positions that should be closed based on profit/loss rules.

        Returns list of dicts with:
            - symbol: str
            - reason: "take_profit" | "stop_loss" | "trailing_stop"
            - current_price: float
            - target_price: float
            - pnl_pct: float
        """
        metrics = self.get_portfolio_metrics()
        to_close = []

        for symbol, pos in metrics.positions.items():
            if pos.hit_take_profit:
                to_close.append({
                    "symbol": symbol,
                    "reason": "take_profit",
                    "side": pos.side,
                    "qty": pos.qty,
                    "current_price": pos.current_price,
                    "target_price": pos.take_profit_price,
                    "pnl_pct": pos.unrealized_pnl_pct,
                    "pnl": pos.unrealized_pnl,
                })
            elif pos.hit_trailing_stop:
                to_close.append({
                    "symbol": symbol,
                    "reason": "trailing_stop",
                    "side": pos.side,
                    "qty": pos.qty,
                    "current_price": pos.current_price,
                    "target_price": pos.trailing_stop_price,
                    "pnl_pct": pos.unrealized_pnl_pct,
                    "pnl": pos.unrealized_pnl,
                })
            elif pos.hit_stop_loss:
                to_close.append({
                    "symbol": symbol,
                    "reason": "stop_loss",
                    "side": pos.side,
                    "qty": pos.qty,
                    "current_price": pos.current_price,
                    "target_price": pos.stop_loss_price,
                    "pnl_pct": pos.unrealized_pnl_pct,
                    "pnl": pos.unrealized_pnl,
                })
            elif pos.hit_max_loss:
                to_close.append({
                    "symbol": symbol,
                    "reason": "max_loss_$50",
                    "side": pos.side,
                    "qty": pos.qty,
                    "current_price": pos.current_price,
                    "target_price": pos.entry_price,
                    "pnl_pct": pos.unrealized_pnl_pct,
                    "pnl": pos.unrealized_pnl,
                })

        return to_close

    def check_daily_loss_limit(self) -> tuple[bool, float]:
        """Check if daily loss limit has been breached.

        Returns:
            (breached: bool, daily_pnl_pct: float)
        """
        metrics = self.get_portfolio_metrics()
        breached = metrics.daily_pnl_pct <= -self.daily_loss_limit_pct
        return breached, metrics.daily_pnl_pct

    def check_max_drawdown(self) -> tuple[bool, float]:
        """Check if maximum drawdown has been breached.

        Returns:
            (breached: bool, drawdown_pct: float)
        """
        metrics = self.get_portfolio_metrics()
        breached = metrics.current_drawdown_pct >= self.max_drawdown_pct
        return breached, metrics.current_drawdown_pct

    def should_halt_trading(self) -> tuple[bool, str]:
        """Check if trading should be halted due to risk limits.

        Returns:
            (should_halt: bool, reason: str)
        """
        # Check daily loss limit
        daily_breached, daily_pnl_pct = self.check_daily_loss_limit()
        if daily_breached:
            return True, f"Daily loss limit breached: {daily_pnl_pct:.2%} (limit: {-self.daily_loss_limit_pct:.2%})"

        # Check max drawdown
        dd_breached, dd_pct = self.check_max_drawdown()
        if dd_breached:
            return True, f"Max drawdown breached: {dd_pct:.2%} (limit: {self.max_drawdown_pct:.2%})"

        return False, ""

    def reset_position_tracking(self, symbol: str) -> None:
        """Reset tracking for a position (call after closing)."""
        self._position_peaks.pop(symbol, None)
        self._position_troughs.pop(symbol, None)
        self._position_entries.pop(symbol, None)

    @staticmethod
    def format_metrics(metrics: PortfolioMetrics) -> str:
        """Human-readable portfolio metrics summary."""
        lines = [
            f"\n{'=' * 70}",
            f"  PORTFOLIO MONITOR",
            f"{'=' * 70}",
            f"  Equity: ${metrics.total_equity:,.2f}  |  "
            f"Market Value: ${metrics.total_market_value:,.2f}",
            f"  Daily P&L: ${metrics.daily_pnl:+,.2f} ({metrics.daily_pnl_pct:+.2%})",
            f"  Unrealized: ${metrics.total_unrealized_pnl:+,.2f} ({metrics.total_unrealized_pnl_pct:+.2%})",
            f"  Drawdown: ${metrics.current_drawdown:,.2f} ({metrics.current_drawdown_pct:.2%})",
            f"  Positions: {metrics.total_positions} "
            f"({metrics.profitable_positions} profit / {metrics.losing_positions} loss)",
            f"{'─' * 70}",
        ]

        if metrics.positions:
            lines.append(
                f"  {'Symbol':8s} {'Side':5s} {'Qty':>6s} {'Entry':>10s} "
                f"{'Current':>10s} {'P&L':>12s} {'P&L%':>8s} {'Status':>12s}"
            )
            lines.append(f"  {'-' * 68}")

            for symbol, pos in metrics.positions.items():
                status = ""
                if pos.hit_take_profit:
                    status = "TAKE PROFIT"
                elif pos.hit_trailing_stop:
                    status = "TRAIL STOP"
                elif pos.hit_stop_loss:
                    status = "STOP LOSS"

                lines.append(
                    f"  {symbol:8s} {pos.side:5s} {pos.qty:>6d} "
                    f"${pos.entry_price:>9.2f} ${pos.current_price:>9.2f} "
                    f"${pos.unrealized_pnl:>+11.2f} {pos.unrealized_pnl_pct:>+7.2%} "
                    f"{status:>12s}"
                )

        lines.append(f"{'=' * 70}\n")
        return "\n".join(lines)
