"""Trailing Stop Manager

Manages dynamic trailing stop-loss orders that move up with price to lock in profits.

Trailing Modes:
    - percentage: Trail by X% below the high watermark
    - atr: Trail by X * ATR below high watermark (adaptive to volatility)
    - fixed: Trail by fixed dollar amount below high watermark
    - stepped: Move stop only when price moves by a threshold (reduces order churn)

Usage:
    manager = TrailingStopManager()
    manager.add_position("AAPL", entry_price=150.0, atr=3.5, mode="percentage", trail_value=5.0)

    # On price updates (e.g., from scheduler or streaming)
    updates = manager.update_stops({"AAPL": 160.0})
    for update in updates:
        print(f"{update['symbol']}: stop moved from ${update['old_stop']:.2f} to ${update['new_stop']:.2f}")
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

TrailingMode = Literal["percentage", "atr", "fixed", "stepped"]

# Default file for persisting trailing stop state
DEFAULT_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "trailing_stops.json"


@dataclass
class TrailingStop:
    """Trailing stop configuration and state for a single position."""

    symbol: str
    entry_price: float
    entry_time: str = ""
    qty: int = 0
    side: str = "long"  # "long" or "short"

    # Trailing configuration
    mode: TrailingMode = "percentage"
    trail_value: float = 5.0  # Interpretation depends on mode
    step_threshold: float = 0.0  # For stepped mode: min price move to trigger update

    # Current state
    high_watermark: float = 0.0  # Highest price since entry (for longs)
    low_watermark: float = 0.0  # Lowest price since entry (for shorts)
    current_stop: float = 0.0
    initial_stop: float = 0.0  # Stop at entry
    atr: float = 0.0  # ATR at entry time (for ATR mode)

    # Broker integration
    stop_order_id: str = ""
    last_update_time: str = ""
    activated: bool = True  # Whether trailing is active

    def __post_init__(self):
        if not self.entry_time:
            self.entry_time = datetime.utcnow().isoformat()
        if self.high_watermark == 0:
            self.high_watermark = self.entry_price
        if self.low_watermark == 0:
            self.low_watermark = self.entry_price


@dataclass
class StopUpdate:
    """Record of a trailing stop adjustment."""

    symbol: str
    old_stop: float
    new_stop: float
    trigger_price: float  # Price that triggered the update
    high_watermark: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class TrailingStopManager:
    """Manages trailing stops for open positions."""

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        state_file: Path | str | None = "default",
        auto_sync: bool = True,
    ):
        """Initialize the trailing stop manager.

        Args:
            client: AlpacaClient for broker integration
            state_file: Path to persist trailing stop state.
                       "default" = use default file, None = disable persistence
            auto_sync: Whether to sync with broker positions on init
        """
        self.client = client
        if state_file == "default":
            self.state_file = DEFAULT_STATE_FILE
        elif state_file is None:
            self.state_file = None
        else:
            self.state_file = Path(state_file)
        self.positions: dict[str, TrailingStop] = {}
        self.update_history: list[StopUpdate] = []

        # Load persisted state
        self._load_state()

        # Sync with broker if client provided
        if auto_sync and client:
            self.sync_with_broker()

    # ── Position Management ────────────────────────────────────────

    def add_position(
        self,
        symbol: str,
        entry_price: float,
        *,
        qty: int = 0,
        side: str = "long",
        mode: TrailingMode = "percentage",
        trail_value: float = 5.0,
        atr: float = 0.0,
        initial_stop: float = 0.0,
        step_threshold: float = 0.0,
    ) -> TrailingStop:
        """Add a new position with trailing stop.

        Args:
            symbol: Ticker symbol
            entry_price: Entry price of the position
            qty: Number of shares
            side: "long" or "short"
            mode: Trailing mode (percentage, atr, fixed, stepped)
            trail_value: Trail amount (% for percentage, multiplier for atr, $ for fixed)
            atr: ATR value at entry (required for atr mode)
            initial_stop: Override initial stop price (otherwise calculated)
            step_threshold: For stepped mode, minimum price move to trigger update

        Returns:
            The created TrailingStop object
        """
        # Calculate initial stop if not provided
        if initial_stop <= 0:
            initial_stop = self._calculate_stop(
                entry_price, entry_price, mode, trail_value, atr, side
            )

        stop = TrailingStop(
            symbol=symbol,
            entry_price=entry_price,
            qty=qty,
            side=side,
            mode=mode,
            trail_value=trail_value,
            atr=atr,
            initial_stop=initial_stop,
            current_stop=initial_stop,
            high_watermark=entry_price,
            low_watermark=entry_price,
            step_threshold=step_threshold,
        )

        self.positions[symbol] = stop
        self._save_state()

        logger.info(
            "Added trailing stop for %s: entry=$%.2f stop=$%.2f mode=%s trail=%.2f",
            symbol, entry_price, initial_stop, mode, trail_value,
        )
        return stop

    def remove_position(self, symbol: str) -> TrailingStop | None:
        """Remove a position from trailing stop management.

        Returns the removed TrailingStop or None if not found.
        """
        stop = self.positions.pop(symbol, None)
        if stop:
            self._save_state()
            logger.info("Removed trailing stop for %s", symbol)
        return stop

    def get_position(self, symbol: str) -> TrailingStop | None:
        """Get trailing stop info for a symbol."""
        return self.positions.get(symbol)

    def list_positions(self) -> list[TrailingStop]:
        """List all tracked positions."""
        return list(self.positions.values())

    # ── Stop Calculation ───────────────────────────────────────────

    def _calculate_stop(
        self,
        reference_price: float,
        watermark: float,
        mode: TrailingMode,
        trail_value: float,
        atr: float,
        side: str = "long",
    ) -> float:
        """Calculate stop price based on mode.

        Args:
            reference_price: Current price for reference
            watermark: High watermark (long) or low watermark (short)
            mode: Trailing mode
            trail_value: Trail amount
            atr: ATR value (for atr mode)
            side: Position side

        Returns:
            Calculated stop price
        """
        if side == "long":
            if mode == "percentage":
                return watermark * (1 - trail_value / 100)
            elif mode == "atr":
                return watermark - (trail_value * atr)
            elif mode == "fixed":
                return watermark - trail_value
            elif mode == "stepped":
                # Stepped uses percentage but only moves on threshold
                return watermark * (1 - trail_value / 100)
        else:  # short
            if mode == "percentage":
                return watermark * (1 + trail_value / 100)
            elif mode == "atr":
                return watermark + (trail_value * atr)
            elif mode == "fixed":
                return watermark + trail_value
            elif mode == "stepped":
                return watermark * (1 + trail_value / 100)

        return reference_price  # Fallback

    # ── Stop Updates ───────────────────────────────────────────────

    def update_stops(self, prices: dict[str, float]) -> list[StopUpdate]:
        """Update trailing stops based on current prices.

        Args:
            prices: Dict of symbol -> current price

        Returns:
            List of StopUpdate objects for stops that were adjusted
        """
        updates: list[StopUpdate] = []

        for symbol, current_price in prices.items():
            stop = self.positions.get(symbol)
            if not stop or not stop.activated:
                continue

            update = self._update_single_stop(stop, current_price)
            if update:
                updates.append(update)
                self.update_history.append(update)

        if updates:
            self._save_state()

        return updates

    def _update_single_stop(
        self, stop: TrailingStop, current_price: float
    ) -> StopUpdate | None:
        """Update a single trailing stop.

        Returns StopUpdate if stop was moved, None otherwise.
        """
        old_stop = stop.current_stop

        if stop.side == "long":
            # Update high watermark
            if current_price > stop.high_watermark:
                stop.high_watermark = current_price

            # Calculate new stop based on high watermark
            new_stop = self._calculate_stop(
                current_price,
                stop.high_watermark,
                stop.mode,
                stop.trail_value,
                stop.atr,
                stop.side,
            )

            # For stepped mode, only move if threshold exceeded
            if stop.mode == "stepped" and stop.step_threshold > 0:
                price_move = stop.high_watermark - stop.entry_price
                if price_move < stop.step_threshold:
                    return None

            # Only move stop up, never down
            if new_stop > old_stop:
                stop.current_stop = new_stop
                stop.last_update_time = datetime.utcnow().isoformat()

                return StopUpdate(
                    symbol=stop.symbol,
                    old_stop=old_stop,
                    new_stop=new_stop,
                    trigger_price=current_price,
                    high_watermark=stop.high_watermark,
                )

        else:  # short position
            # Update low watermark
            if current_price < stop.low_watermark:
                stop.low_watermark = current_price

            # Calculate new stop based on low watermark
            new_stop = self._calculate_stop(
                current_price,
                stop.low_watermark,
                stop.mode,
                stop.trail_value,
                stop.atr,
                stop.side,
            )

            # For stepped mode, only move if threshold exceeded
            if stop.mode == "stepped" and stop.step_threshold > 0:
                price_move = stop.entry_price - stop.low_watermark
                if price_move < stop.step_threshold:
                    return None

            # Only move stop down for shorts, never up
            if new_stop < old_stop:
                stop.current_stop = new_stop
                stop.last_update_time = datetime.utcnow().isoformat()

                return StopUpdate(
                    symbol=stop.symbol,
                    old_stop=old_stop,
                    new_stop=new_stop,
                    trigger_price=current_price,
                    high_watermark=stop.low_watermark,
                )

        return None

    def check_stop_triggered(
        self, symbol: str, current_price: float
    ) -> bool:
        """Check if a stop has been triggered.

        Args:
            symbol: Ticker symbol
            current_price: Current market price

        Returns:
            True if stop was triggered (price crossed stop level)
        """
        stop = self.positions.get(symbol)
        if not stop or not stop.activated:
            return False

        if stop.side == "long":
            return current_price <= stop.current_stop
        else:
            return current_price >= stop.current_stop

    # ── Broker Integration ─────────────────────────────────────────

    def sync_with_broker(self) -> dict:
        """Sync trailing stops with broker positions.

        Adds trailing stops for new positions and removes closed ones.

        Returns:
            Dict with 'added' and 'removed' symbol lists
        """
        if not self.client:
            logger.warning("No client configured for broker sync")
            return {"added": [], "removed": []}

        try:
            broker_positions = self.client.get_positions()
        except Exception as e:
            logger.error("Failed to sync with broker: %s", e)
            return {"added": [], "removed": []}

        broker_symbols = {p["symbol"] for p in broker_positions}
        tracked_symbols = set(self.positions.keys())

        added = []
        removed = []

        # Add new positions
        for pos in broker_positions:
            symbol = pos["symbol"]
            if symbol not in tracked_symbols:
                # New position - add with default trailing stop
                self.add_position(
                    symbol=symbol,
                    entry_price=pos["current_price"],
                    qty=pos["qty"],
                    side="long" if pos["side"] == "long" else "short",
                )
                added.append(symbol)

        # Remove closed positions
        for symbol in tracked_symbols - broker_symbols:
            self.remove_position(symbol)
            removed.append(symbol)

        logger.info(
            "Broker sync: added %d, removed %d positions",
            len(added), len(removed),
        )
        return {"added": added, "removed": removed}

    def submit_stop_order(self, symbol: str) -> dict | None:
        """Submit a stop-loss order to the broker.

        Args:
            symbol: Ticker symbol

        Returns:
            Order dict if successful, None otherwise
        """
        if not self.client:
            logger.warning("No client configured for order submission")
            return None

        stop = self.positions.get(symbol)
        if not stop:
            logger.warning("No trailing stop for %s", symbol)
            return None

        try:
            # Submit stop order
            side = "sell" if stop.side == "long" else "buy"
            order = self.client.api.submit_order(
                symbol=symbol,
                qty=stop.qty,
                side=side,
                type="stop",
                stop_price=str(round(stop.current_stop, 2)),
                time_in_force="gtc",  # Good til cancelled
            )

            stop.stop_order_id = order.id
            self._save_state()

            logger.info(
                "Submitted stop order for %s: %s %d @ $%.2f (ID: %s)",
                symbol, side, stop.qty, stop.current_stop, order.id,
            )
            return {
                "id": order.id,
                "symbol": symbol,
                "side": side,
                "qty": stop.qty,
                "stop_price": stop.current_stop,
                "status": order.status,
            }

        except Exception as e:
            logger.error("Failed to submit stop order for %s: %s", symbol, e)
            return None

    def update_stop_order(self, symbol: str) -> dict | None:
        """Update an existing stop order with new stop price.

        Cancels existing order and submits new one.

        Args:
            symbol: Ticker symbol

        Returns:
            New order dict if successful, None otherwise
        """
        if not self.client:
            return None

        stop = self.positions.get(symbol)
        if not stop or not stop.stop_order_id:
            return None

        try:
            # Cancel existing order
            self.client.api.cancel_order(stop.stop_order_id)
            logger.info("Cancelled old stop order %s", stop.stop_order_id)
        except Exception as e:
            logger.warning("Failed to cancel old stop order: %s", e)

        # Submit new order
        return self.submit_stop_order(symbol)

    def cancel_stop_order(self, symbol: str) -> bool:
        """Cancel a stop order for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            True if cancelled successfully
        """
        if not self.client:
            return False

        stop = self.positions.get(symbol)
        if not stop or not stop.stop_order_id:
            return False

        try:
            self.client.api.cancel_order(stop.stop_order_id)
            stop.stop_order_id = ""
            self._save_state()
            logger.info("Cancelled stop order for %s", symbol)
            return True
        except Exception as e:
            logger.error("Failed to cancel stop order for %s: %s", symbol, e)
            return False

    # ── State Persistence ──────────────────────────────────────────

    def _save_state(self) -> None:
        """Save trailing stop state to file."""
        if not self.state_file:
            return

        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "positions": {
                    symbol: asdict(stop) for symbol, stop in self.positions.items()
                },
                "last_save": datetime.utcnow().isoformat(),
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save trailing stop state: %s", e)

    def _load_state(self) -> None:
        """Load trailing stop state from file."""
        if not self.state_file or not self.state_file.exists():
            return

        try:
            with open(self.state_file) as f:
                data = json.load(f)

            for symbol, stop_data in data.get("positions", {}).items():
                self.positions[symbol] = TrailingStop(**stop_data)

            logger.info(
                "Loaded %d trailing stops from %s",
                len(self.positions), self.state_file,
            )
        except Exception as e:
            logger.error("Failed to load trailing stop state: %s", e)

    def clear_state(self) -> None:
        """Clear all trailing stop state."""
        self.positions.clear()
        self.update_history.clear()
        self._save_state()
        logger.info("Cleared all trailing stop state")

    # ── Analysis & Reporting ───────────────────────────────────────

    def get_status(self, symbol: str) -> dict | None:
        """Get detailed status for a position's trailing stop."""
        stop = self.positions.get(symbol)
        if not stop:
            return None

        return {
            "symbol": stop.symbol,
            "side": stop.side,
            "entry_price": stop.entry_price,
            "current_stop": stop.current_stop,
            "initial_stop": stop.initial_stop,
            "high_watermark": stop.high_watermark,
            "low_watermark": stop.low_watermark,
            "mode": stop.mode,
            "trail_value": stop.trail_value,
            "atr": stop.atr,
            "stop_order_id": stop.stop_order_id,
            "activated": stop.activated,
            "profit_locked": self._calculate_locked_profit(stop),
            "entry_time": stop.entry_time,
            "last_update": stop.last_update_time,
        }

    def _calculate_locked_profit(self, stop: TrailingStop) -> float:
        """Calculate the profit locked in by the current stop level."""
        if stop.side == "long":
            return max(0, (stop.current_stop - stop.entry_price) * stop.qty)
        else:
            return max(0, (stop.entry_price - stop.current_stop) * stop.qty)

    @staticmethod
    def format_status(positions: list[dict]) -> str:
        """Format trailing stop status as a readable table."""
        if not positions:
            return "\n  No active trailing stops.\n"

        lines = [
            f"\n{'=' * 78}",
            f"  TRAILING STOP STATUS",
            f"{'=' * 78}",
            f"  {'Symbol':<8} {'Side':<6} {'Entry':>10} {'Stop':>10} {'High':>10} {'Locked':>12} {'Mode':<10}",
            f"  {'-' * 8} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 10}",
        ]

        for pos in positions:
            lines.append(
                f"  {pos['symbol']:<8} "
                f"{pos['side']:<6} "
                f"${pos['entry_price']:>9.2f} "
                f"${pos['current_stop']:>9.2f} "
                f"${pos['high_watermark']:>9.2f} "
                f"${pos['profit_locked']:>11.2f} "
                f"{pos['mode']:<10}"
            )

        lines.append(f"{'=' * 78}\n")
        return "\n".join(lines)

    def format_updates(self, updates: list[StopUpdate]) -> str:
        """Format stop updates as readable output."""
        if not updates:
            return "  No stop updates.\n"

        lines = [
            f"\n{'─' * 62}",
            f"  TRAILING STOP UPDATES",
            f"{'─' * 62}",
        ]

        for u in updates:
            lines.append(
                f"  {u.symbol}: Stop moved ${u.old_stop:.2f} → ${u.new_stop:.2f} "
                f"(price=${u.trigger_price:.2f}, high=${u.high_watermark:.2f})"
            )

        lines.append(f"{'─' * 62}\n")
        return "\n".join(lines)
