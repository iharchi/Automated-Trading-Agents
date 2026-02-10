"""Scalping Position Manager

Manages open positions for scalping strategies with:
- Real-time P&L monitoring
- Quick exit on profit target or stop loss
- Time-based exits (don't hold positions too long)
- Trailing profit locks
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

from utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


@dataclass
class ScalpPosition:
    """Track a scalping position."""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    qty: int
    stop_loss: float
    take_profit: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Tracking
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    high_price: float = 0.0  # Highest price since entry (for trailing)
    low_price: float = 999999.0  # Lowest price since entry

    # Exit info
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    realized_pnl: float = 0.0

    # Status
    status: str = "open"  # open, closing, closed


@dataclass
class ScalpStats:
    """Scalping session statistics."""
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_hold_time_seconds: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0


class ScalpingManager:
    """Manage scalping positions with real-time monitoring."""

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        check_interval: float = 1.0,  # Check positions every 1 second
        max_hold_minutes: int = 30,  # Max time to hold a position
        trailing_lock_pct: float = 0.3,  # Lock in 30% of gains with trailing
        min_profit_to_trail: float = 0.2,  # Start trailing at 0.2% profit
        dry_run: bool = False,
        on_exit: Callable[[ScalpPosition], None] | None = None,
    ):
        self.client = client or AlpacaClient()
        self.check_interval = check_interval
        self.max_hold_minutes = max_hold_minutes
        self.trailing_lock_pct = trailing_lock_pct
        self.min_profit_to_trail = min_profit_to_trail
        self.dry_run = dry_run
        self.on_exit = on_exit

        # Position tracking
        self._positions: dict[str, ScalpPosition] = {}
        self._closed_positions: list[ScalpPosition] = []
        self._lock = threading.Lock()

        # Monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

    # ── Position Management ──────────────────────────────────────

    def add_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        qty: int,
        stop_loss: float,
        take_profit: float,
    ) -> ScalpPosition:
        """Add a new scalping position to track."""
        pos = ScalpPosition(
            symbol=symbol,
            side=side.lower(),
            entry_price=entry_price,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=entry_price,
            high_price=entry_price,
            low_price=entry_price,
        )

        with self._lock:
            self._positions[symbol] = pos

        logger.info(
            "SCALP OPEN: %s %s %d @ $%.2f | SL: $%.2f | TP: $%.2f",
            side.upper(), symbol, qty, entry_price, stop_loss, take_profit
        )

        return pos

    def get_position(self, symbol: str) -> Optional[ScalpPosition]:
        """Get position for a symbol."""
        with self._lock:
            return self._positions.get(symbol)

    def get_all_positions(self) -> list[ScalpPosition]:
        """Get all open positions."""
        with self._lock:
            return list(self._positions.values())

    def remove_position(self, symbol: str) -> Optional[ScalpPosition]:
        """Remove and return a position."""
        with self._lock:
            return self._positions.pop(symbol, None)

    # ── Price Updates ────────────────────────────────────────────

    def update_price(self, symbol: str, current_price: float):
        """Update current price and check for exits."""
        with self._lock:
            pos = self._positions.get(symbol)
            if pos is None or pos.status != "open":
                return

            pos.current_price = current_price

            # Update high/low
            if current_price > pos.high_price:
                pos.high_price = current_price
            if current_price < pos.low_price:
                pos.low_price = current_price

            # Calculate unrealized P&L
            if pos.side == "long":
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.qty
            else:  # short
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.qty

            pos.unrealized_pnl_pct = (pos.unrealized_pnl / (pos.entry_price * pos.qty)) * 100

            # Check exit conditions
            exit_reason = self._check_exit_conditions(pos)
            if exit_reason:
                self._trigger_exit(pos, exit_reason)

    def _check_exit_conditions(self, pos: ScalpPosition) -> str:
        """Check if position should be exited. Returns exit reason or empty string."""
        price = pos.current_price

        # Stop loss hit
        if pos.side == "long" and price <= pos.stop_loss:
            return "stop_loss"
        if pos.side == "short" and price >= pos.stop_loss:
            return "stop_loss"

        # Take profit hit
        if pos.side == "long" and price >= pos.take_profit:
            return "take_profit"
        if pos.side == "short" and price <= pos.take_profit:
            return "take_profit"

        # Trailing stop (lock in profits)
        if pos.unrealized_pnl_pct >= self.min_profit_to_trail:
            # Calculate trailing stop level
            if pos.side == "long":
                trail_stop = pos.high_price * (1 - self.trailing_lock_pct / 100)
                if price <= trail_stop and pos.high_price > pos.entry_price * 1.002:
                    return "trailing_stop"
            else:
                trail_stop = pos.low_price * (1 + self.trailing_lock_pct / 100)
                if price >= trail_stop and pos.low_price < pos.entry_price * 0.998:
                    return "trailing_stop"

        # Time-based exit
        hold_time = datetime.now(timezone.utc) - pos.entry_time
        if hold_time > timedelta(minutes=self.max_hold_minutes):
            return "time_limit"

        return ""

    def _trigger_exit(self, pos: ScalpPosition, reason: str):
        """Trigger position exit."""
        pos.status = "closing"
        pos.exit_reason = reason
        pos.exit_price = pos.current_price
        pos.exit_time = datetime.now(timezone.utc)

        # Calculate realized P&L
        if pos.side == "long":
            pos.realized_pnl = (pos.exit_price - pos.entry_price) * pos.qty
        else:
            pos.realized_pnl = (pos.entry_price - pos.exit_price) * pos.qty

        logger.info(
            "SCALP EXIT: %s %s @ $%.2f | Reason: %s | P&L: $%.2f (%.2f%%)",
            pos.symbol, pos.side.upper(), pos.exit_price, reason,
            pos.realized_pnl, pos.unrealized_pnl_pct
        )

        # Execute the exit order
        if not self.dry_run:
            try:
                exit_side = "sell" if pos.side == "long" else "buy"
                self.client.submit_order(
                    symbol=pos.symbol,
                    qty=pos.qty,
                    side=exit_side,
                    order_type="market",
                )
            except Exception as e:
                logger.error("Failed to close scalp position %s: %s", pos.symbol, e)
                pos.status = "open"  # Revert status
                return

        pos.status = "closed"

        # Move to closed positions
        with self._lock:
            self._positions.pop(pos.symbol, None)
            self._closed_positions.append(pos)

        # Callback
        if self.on_exit:
            try:
                self.on_exit(pos)
            except Exception as e:
                logger.warning("Exit callback failed: %s", e)

    # ── Monitoring Thread ────────────────────────────────────────

    def start_monitoring(self):
        """Start background thread to monitor positions."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return

        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Scalping monitor started (interval: %.1fs)", self.check_interval)

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Scalping monitor stopped")

    def _monitor_loop(self):
        """Background loop to update prices and check exits."""
        while not self._stop_monitoring.is_set():
            try:
                with self._lock:
                    symbols = list(self._positions.keys())

                for symbol in symbols:
                    try:
                        # Get latest price
                        bars = self.client.get_bars(symbol, timeframe="1Min", limit=1)
                        if not bars.empty:
                            current_price = float(bars["close"].iloc[-1])
                            self.update_price(symbol, current_price)
                    except Exception as e:
                        logger.debug("Price update failed for %s: %s", symbol, e)

            except Exception as e:
                logger.error("Monitor loop error: %s", e)

            self._stop_monitoring.wait(self.check_interval)

    # ── Manual Exit ──────────────────────────────────────────────

    def exit_position(self, symbol: str, reason: str = "manual"):
        """Manually exit a position."""
        with self._lock:
            pos = self._positions.get(symbol)
            if pos and pos.status == "open":
                self._trigger_exit(pos, reason)

    def exit_all(self, reason: str = "exit_all"):
        """Exit all open positions."""
        with self._lock:
            symbols = list(self._positions.keys())

        for symbol in symbols:
            self.exit_position(symbol, reason)

    # ── Statistics ───────────────────────────────────────────────

    def get_stats(self) -> ScalpStats:
        """Get scalping session statistics."""
        stats = ScalpStats()

        with self._lock:
            closed = self._closed_positions.copy()

        if not closed:
            return stats

        stats.total_trades = len(closed)
        wins = [p for p in closed if p.realized_pnl > 0]
        losses = [p for p in closed if p.realized_pnl <= 0]

        stats.winners = len(wins)
        stats.losers = len(losses)
        stats.total_pnl = sum(p.realized_pnl for p in closed)

        if wins:
            stats.avg_win = sum(p.realized_pnl for p in wins) / len(wins)
        if losses:
            stats.avg_loss = sum(p.realized_pnl for p in losses) / len(losses)

        # Average hold time
        hold_times = [
            (p.exit_time - p.entry_time).total_seconds()
            for p in closed if p.exit_time
        ]
        if hold_times:
            stats.avg_hold_time_seconds = sum(hold_times) / len(hold_times)

        # Win rate
        if stats.total_trades > 0:
            stats.win_rate = stats.winners / stats.total_trades

        # Profit factor
        gross_profit = sum(p.realized_pnl for p in wins) if wins else 0
        gross_loss = abs(sum(p.realized_pnl for p in losses)) if losses else 0
        if gross_loss > 0:
            stats.profit_factor = gross_profit / gross_loss

        return stats

    # ── Formatting ───────────────────────────────────────────────

    def format_positions(self) -> str:
        """Format current positions for display."""
        with self._lock:
            positions = list(self._positions.values())

        if not positions:
            return "No open scalp positions"

        lines = [
            f"\n{'=' * 70}",
            "  SCALP POSITIONS",
            f"{'=' * 70}",
            f"  {'Symbol':<8} {'Side':<6} {'Qty':>6} {'Entry':>10} {'Current':>10} {'P&L':>12} {'Time':>8}",
            f"  {'-' * 66}",
        ]

        for pos in positions:
            hold_time = datetime.now(timezone.utc) - pos.entry_time
            mins = int(hold_time.total_seconds() / 60)
            secs = int(hold_time.total_seconds() % 60)

            pnl_str = f"${pos.unrealized_pnl:+,.2f} ({pos.unrealized_pnl_pct:+.2f}%)"
            emoji = "+" if pos.unrealized_pnl >= 0 else "-"

            lines.append(
                f"  {pos.symbol:<8} {pos.side.upper():<6} {pos.qty:>6} "
                f"${pos.entry_price:>9.2f} ${pos.current_price:>9.2f} "
                f"{pnl_str:>12} {mins:>3}m{secs:02d}s"
            )

        lines.append(f"{'=' * 70}\n")
        return "\n".join(lines)

    def format_stats(self) -> str:
        """Format session statistics for display."""
        stats = self.get_stats()

        return f"""
{'=' * 50}
  SCALP SESSION STATS
{'=' * 50}
  Total Trades:    {stats.total_trades}
  Win Rate:        {stats.win_rate:.1%}
  Winners/Losers:  {stats.winners}/{stats.losers}
  Total P&L:       ${stats.total_pnl:+,.2f}
  Avg Win:         ${stats.avg_win:+,.2f}
  Avg Loss:        ${stats.avg_loss:+,.2f}
  Profit Factor:   {stats.profit_factor:.2f}
  Avg Hold Time:   {stats.avg_hold_time_seconds:.0f}s
{'=' * 50}
"""
