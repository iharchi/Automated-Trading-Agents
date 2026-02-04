"""Unit tests for the Trailing Stop Manager.

Tests cover:
    - TrailingStop dataclass
    - Stop calculation for different modes (percentage, atr, fixed, stepped)
    - Stop updates on price movements
    - Long and short position handling
    - State persistence
    - Broker integration
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from utils.trailing_stop_manager import (
    TrailingStopManager,
    TrailingStop,
    StopUpdate,
)


# ── TrailingStop dataclass tests ────────────────────────────────────


class TestTrailingStopDataclass(unittest.TestCase):
    """Tests for TrailingStop dataclass."""

    def test_default_values(self):
        stop = TrailingStop(symbol="AAPL", entry_price=150.0)
        self.assertEqual(stop.symbol, "AAPL")
        self.assertEqual(stop.entry_price, 150.0)
        self.assertEqual(stop.side, "long")
        self.assertEqual(stop.mode, "percentage")
        self.assertEqual(stop.trail_value, 5.0)
        self.assertEqual(stop.high_watermark, 150.0)  # Set to entry price
        self.assertEqual(stop.low_watermark, 150.0)

    def test_custom_values(self):
        stop = TrailingStop(
            symbol="TSLA",
            entry_price=200.0,
            qty=10,
            side="short",
            mode="atr",
            trail_value=2.0,
            atr=8.0,
        )
        self.assertEqual(stop.symbol, "TSLA")
        self.assertEqual(stop.side, "short")
        self.assertEqual(stop.mode, "atr")
        self.assertEqual(stop.atr, 8.0)

    def test_entry_time_auto_set(self):
        stop = TrailingStop(symbol="AAPL", entry_price=150.0)
        self.assertIsNotNone(stop.entry_time)
        self.assertTrue(len(stop.entry_time) > 0)


# ── StopUpdate dataclass tests ──────────────────────────────────────


class TestStopUpdateDataclass(unittest.TestCase):
    """Tests for StopUpdate dataclass."""

    def test_creation(self):
        update = StopUpdate(
            symbol="AAPL",
            old_stop=140.0,
            new_stop=145.0,
            trigger_price=155.0,
            high_watermark=155.0,
        )
        self.assertEqual(update.symbol, "AAPL")
        self.assertEqual(update.old_stop, 140.0)
        self.assertEqual(update.new_stop, 145.0)
        self.assertEqual(update.trigger_price, 155.0)

    def test_timestamp_auto_set(self):
        update = StopUpdate(
            symbol="AAPL",
            old_stop=140.0,
            new_stop=145.0,
            trigger_price=155.0,
            high_watermark=155.0,
        )
        self.assertIsNotNone(update.timestamp)


# ── Stop calculation tests ──────────────────────────────────────────


class TestStopCalculation(unittest.TestCase):
    """Tests for stop price calculation."""

    def setUp(self):
        self.manager = TrailingStopManager(client=None, state_file=None, auto_sync=False)

    def test_percentage_mode_long(self):
        # 5% below 100 = 95
        stop = self.manager._calculate_stop(
            reference_price=100.0,
            watermark=100.0,
            mode="percentage",
            trail_value=5.0,
            atr=0,
            side="long",
        )
        self.assertEqual(stop, 95.0)

    def test_percentage_mode_short(self):
        # 5% above 100 = 105
        stop = self.manager._calculate_stop(
            reference_price=100.0,
            watermark=100.0,
            mode="percentage",
            trail_value=5.0,
            atr=0,
            side="short",
        )
        self.assertEqual(stop, 105.0)

    def test_atr_mode_long(self):
        # 2 * ATR(5) below 100 = 90
        stop = self.manager._calculate_stop(
            reference_price=100.0,
            watermark=100.0,
            mode="atr",
            trail_value=2.0,
            atr=5.0,
            side="long",
        )
        self.assertEqual(stop, 90.0)

    def test_atr_mode_short(self):
        # 2 * ATR(5) above 100 = 110
        stop = self.manager._calculate_stop(
            reference_price=100.0,
            watermark=100.0,
            mode="atr",
            trail_value=2.0,
            atr=5.0,
            side="short",
        )
        self.assertEqual(stop, 110.0)

    def test_fixed_mode_long(self):
        # $10 below 100 = 90
        stop = self.manager._calculate_stop(
            reference_price=100.0,
            watermark=100.0,
            mode="fixed",
            trail_value=10.0,
            atr=0,
            side="long",
        )
        self.assertEqual(stop, 90.0)

    def test_fixed_mode_short(self):
        # $10 above 100 = 110
        stop = self.manager._calculate_stop(
            reference_price=100.0,
            watermark=100.0,
            mode="fixed",
            trail_value=10.0,
            atr=0,
            side="short",
        )
        self.assertEqual(stop, 110.0)


# ── Position management tests ───────────────────────────────────────


class TestPositionManagement(unittest.TestCase):
    """Tests for adding, removing, and listing positions."""

    def setUp(self):
        self.manager = TrailingStopManager(client=None, state_file=None, auto_sync=False)

    def test_add_position(self):
        stop = self.manager.add_position(
            symbol="AAPL",
            entry_price=150.0,
            qty=10,
            mode="percentage",
            trail_value=5.0,
        )
        self.assertEqual(stop.symbol, "AAPL")
        self.assertEqual(stop.entry_price, 150.0)
        self.assertEqual(stop.qty, 10)
        # Initial stop = 150 * 0.95 = 142.5
        self.assertAlmostEqual(stop.current_stop, 142.5, places=2)

    def test_add_position_with_atr(self):
        stop = self.manager.add_position(
            symbol="TSLA",
            entry_price=200.0,
            mode="atr",
            trail_value=2.0,
            atr=8.0,
        )
        # Initial stop = 200 - 2*8 = 184
        self.assertEqual(stop.current_stop, 184.0)

    def test_remove_position(self):
        self.manager.add_position("AAPL", entry_price=150.0)
        removed = self.manager.remove_position("AAPL")
        self.assertIsNotNone(removed)
        self.assertEqual(removed.symbol, "AAPL")
        self.assertIsNone(self.manager.get_position("AAPL"))

    def test_remove_nonexistent_position(self):
        removed = self.manager.remove_position("FAKE")
        self.assertIsNone(removed)

    def test_list_positions(self):
        self.manager.add_position("AAPL", entry_price=150.0)
        self.manager.add_position("MSFT", entry_price=300.0)
        positions = self.manager.list_positions()
        self.assertEqual(len(positions), 2)
        symbols = [p.symbol for p in positions]
        self.assertIn("AAPL", symbols)
        self.assertIn("MSFT", symbols)

    def test_get_position(self):
        self.manager.add_position("AAPL", entry_price=150.0)
        stop = self.manager.get_position("AAPL")
        self.assertIsNotNone(stop)
        self.assertEqual(stop.symbol, "AAPL")

    def test_get_nonexistent_position(self):
        stop = self.manager.get_position("FAKE")
        self.assertIsNone(stop)


# ── Stop update tests ───────────────────────────────────────────────


class TestStopUpdates(unittest.TestCase):
    """Tests for trailing stop updates."""

    def setUp(self):
        self.manager = TrailingStopManager(client=None, state_file=None, auto_sync=False)

    def test_stop_moves_up_on_price_increase_long(self):
        self.manager.add_position(
            "AAPL",
            entry_price=100.0,
            mode="percentage",
            trail_value=5.0,
        )
        # Initial stop = 95

        # Price increases to 110
        updates = self.manager.update_stops({"AAPL": 110.0})

        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].symbol, "AAPL")
        self.assertEqual(updates[0].old_stop, 95.0)
        # New stop = 110 * 0.95 = 104.5
        self.assertAlmostEqual(updates[0].new_stop, 104.5, places=2)

    def test_stop_does_not_move_down_long(self):
        self.manager.add_position(
            "AAPL",
            entry_price=100.0,
            mode="percentage",
            trail_value=5.0,
        )
        # Move price up first
        self.manager.update_stops({"AAPL": 110.0})
        stop = self.manager.get_position("AAPL")
        high_stop = stop.current_stop

        # Price decreases - stop should not move down
        updates = self.manager.update_stops({"AAPL": 105.0})

        self.assertEqual(len(updates), 0)
        self.assertEqual(stop.current_stop, high_stop)

    def test_stop_moves_down_on_price_decrease_short(self):
        self.manager.add_position(
            "AAPL",
            entry_price=100.0,
            side="short",
            mode="percentage",
            trail_value=5.0,
        )
        # Initial stop = 105

        # Price decreases to 90
        updates = self.manager.update_stops({"AAPL": 90.0})

        self.assertEqual(len(updates), 1)
        # New stop = 90 * 1.05 = 94.5
        self.assertAlmostEqual(updates[0].new_stop, 94.5, places=2)

    def test_stop_does_not_move_up_short(self):
        self.manager.add_position(
            "AAPL",
            entry_price=100.0,
            side="short",
            mode="percentage",
            trail_value=5.0,
        )
        # Move price down first
        self.manager.update_stops({"AAPL": 90.0})
        stop = self.manager.get_position("AAPL")
        low_stop = stop.current_stop

        # Price increases - stop should not move up
        updates = self.manager.update_stops({"AAPL": 95.0})

        self.assertEqual(len(updates), 0)
        self.assertEqual(stop.current_stop, low_stop)

    def test_high_watermark_updates(self):
        self.manager.add_position("AAPL", entry_price=100.0)

        # Price increases
        self.manager.update_stops({"AAPL": 110.0})
        stop = self.manager.get_position("AAPL")
        self.assertEqual(stop.high_watermark, 110.0)

        # Price increases more
        self.manager.update_stops({"AAPL": 120.0})
        stop = self.manager.get_position("AAPL")
        self.assertEqual(stop.high_watermark, 120.0)

        # Price decreases - watermark should stay
        self.manager.update_stops({"AAPL": 115.0})
        stop = self.manager.get_position("AAPL")
        self.assertEqual(stop.high_watermark, 120.0)

    def test_no_update_for_inactive_position(self):
        stop = self.manager.add_position("AAPL", entry_price=100.0)
        stop.activated = False

        updates = self.manager.update_stops({"AAPL": 110.0})
        self.assertEqual(len(updates), 0)

    def test_no_update_for_unknown_symbol(self):
        self.manager.add_position("AAPL", entry_price=100.0)
        updates = self.manager.update_stops({"UNKNOWN": 110.0})
        self.assertEqual(len(updates), 0)


# ── Stop trigger tests ──────────────────────────────────────────────


class TestStopTrigger(unittest.TestCase):
    """Tests for checking if stops are triggered."""

    def setUp(self):
        self.manager = TrailingStopManager(client=None, state_file=None, auto_sync=False)

    def test_stop_triggered_long(self):
        self.manager.add_position("AAPL", entry_price=100.0, mode="percentage", trail_value=5.0)
        # Stop is at 95

        # Price at stop level
        self.assertTrue(self.manager.check_stop_triggered("AAPL", 95.0))

        # Price below stop
        self.assertTrue(self.manager.check_stop_triggered("AAPL", 90.0))

        # Price above stop
        self.assertFalse(self.manager.check_stop_triggered("AAPL", 96.0))

    def test_stop_triggered_short(self):
        self.manager.add_position(
            "AAPL",
            entry_price=100.0,
            side="short",
            mode="percentage",
            trail_value=5.0,
        )
        # Stop is at 105

        # Price at stop level
        self.assertTrue(self.manager.check_stop_triggered("AAPL", 105.0))

        # Price above stop
        self.assertTrue(self.manager.check_stop_triggered("AAPL", 110.0))

        # Price below stop
        self.assertFalse(self.manager.check_stop_triggered("AAPL", 104.0))

    def test_stop_trigger_unknown_symbol(self):
        self.assertFalse(self.manager.check_stop_triggered("UNKNOWN", 100.0))


# ── State persistence tests ─────────────────────────────────────────


class TestStatePersistence(unittest.TestCase):
    """Tests for saving and loading state."""

    def test_save_and_load_state(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = Path(f.name)

        try:
            # Create manager and add positions
            manager1 = TrailingStopManager(client=None, state_file=state_file, auto_sync=False)
            manager1.add_position("AAPL", entry_price=150.0, qty=10)
            manager1.add_position("MSFT", entry_price=300.0, qty=5)

            # Create new manager and verify it loaded state
            manager2 = TrailingStopManager(client=None, state_file=state_file, auto_sync=False)
            positions = manager2.list_positions()

            self.assertEqual(len(positions), 2)
            aapl = manager2.get_position("AAPL")
            self.assertIsNotNone(aapl)
            self.assertEqual(aapl.entry_price, 150.0)
            self.assertEqual(aapl.qty, 10)

        finally:
            state_file.unlink()

    def test_clear_state(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = Path(f.name)

        try:
            manager = TrailingStopManager(client=None, state_file=state_file, auto_sync=False)
            manager.add_position("AAPL", entry_price=150.0)
            manager.clear_state()

            self.assertEqual(len(manager.list_positions()), 0)

            # Verify file is cleared too
            manager2 = TrailingStopManager(client=None, state_file=state_file, auto_sync=False)
            self.assertEqual(len(manager2.list_positions()), 0)

        finally:
            state_file.unlink()


# ── Status and formatting tests ─────────────────────────────────────


class TestStatusAndFormatting(unittest.TestCase):
    """Tests for status retrieval and formatting."""

    def setUp(self):
        self.manager = TrailingStopManager(client=None, state_file=None, auto_sync=False)

    def test_get_status(self):
        self.manager.add_position(
            "AAPL",
            entry_price=150.0,
            qty=10,
            mode="percentage",
            trail_value=5.0,
        )

        status = self.manager.get_status("AAPL")

        self.assertIsNotNone(status)
        self.assertEqual(status["symbol"], "AAPL")
        self.assertEqual(status["entry_price"], 150.0)
        self.assertEqual(status["mode"], "percentage")
        self.assertIn("profit_locked", status)

    def test_get_status_nonexistent(self):
        status = self.manager.get_status("FAKE")
        self.assertIsNone(status)

    def test_format_status_empty(self):
        output = TrailingStopManager.format_status([])
        self.assertIn("No active", output)

    def test_format_status_with_positions(self):
        self.manager.add_position("AAPL", entry_price=150.0, qty=10)
        positions = [self.manager.get_status("AAPL")]
        output = TrailingStopManager.format_status(positions)

        self.assertIn("TRAILING STOP STATUS", output)
        self.assertIn("AAPL", output)
        self.assertIn("150.0", output)

    def test_format_updates_empty(self):
        output = self.manager.format_updates([])
        self.assertIn("No stop updates", output)

    def test_format_updates_with_data(self):
        updates = [
            StopUpdate(
                symbol="AAPL",
                old_stop=140.0,
                new_stop=145.0,
                trigger_price=155.0,
                high_watermark=155.0,
            )
        ]
        output = self.manager.format_updates(updates)

        self.assertIn("STOP UPDATES", output)
        self.assertIn("AAPL", output)
        self.assertIn("140.0", output)
        self.assertIn("145.0", output)


# ── Locked profit calculation tests ─────────────────────────────────


class TestLockedProfit(unittest.TestCase):
    """Tests for locked profit calculation."""

    def setUp(self):
        self.manager = TrailingStopManager(client=None, state_file=None, auto_sync=False)

    def test_locked_profit_long_position(self):
        stop = self.manager.add_position(
            "AAPL",
            entry_price=100.0,
            qty=10,
            mode="percentage",
            trail_value=5.0,
        )
        # Initial stop = 95, so no profit locked

        profit = self.manager._calculate_locked_profit(stop)
        self.assertEqual(profit, 0.0)  # Stop below entry = no locked profit

        # Price rises, stop moves up
        self.manager.update_stops({"AAPL": 120.0})
        # New stop = 120 * 0.95 = 114

        profit = self.manager._calculate_locked_profit(stop)
        # Locked profit = (114 - 100) * 10 = 140
        self.assertEqual(profit, 140.0)

    def test_locked_profit_short_position(self):
        stop = self.manager.add_position(
            "AAPL",
            entry_price=100.0,
            qty=10,
            side="short",
            mode="percentage",
            trail_value=5.0,
        )
        # Initial stop = 105, so no profit locked

        profit = self.manager._calculate_locked_profit(stop)
        self.assertEqual(profit, 0.0)

        # Price falls, stop moves down
        self.manager.update_stops({"AAPL": 80.0})
        # New stop = 80 * 1.05 = 84

        profit = self.manager._calculate_locked_profit(stop)
        # Locked profit = (100 - 84) * 10 = 160
        self.assertEqual(profit, 160.0)


# ── Broker integration tests (mocked) ───────────────────────────────


class TestBrokerIntegration(unittest.TestCase):
    """Tests for broker integration with mocked client."""

    def test_sync_with_broker_adds_new_positions(self):
        mock_client = MagicMock()
        mock_client.get_positions.return_value = [
            {"symbol": "AAPL", "qty": 10, "side": "long", "current_price": 150.0},
            {"symbol": "MSFT", "qty": 5, "side": "long", "current_price": 300.0},
        ]

        manager = TrailingStopManager(client=mock_client, state_file=None, auto_sync=False)
        result = manager.sync_with_broker()

        self.assertEqual(len(result["added"]), 2)
        self.assertIn("AAPL", result["added"])
        self.assertIn("MSFT", result["added"])
        self.assertEqual(len(manager.list_positions()), 2)

    def test_sync_with_broker_removes_closed_positions(self):
        mock_client = MagicMock()

        manager = TrailingStopManager(client=mock_client, state_file=None, auto_sync=False)
        manager.add_position("AAPL", entry_price=150.0)
        manager.add_position("MSFT", entry_price=300.0)

        # Broker now only has AAPL
        mock_client.get_positions.return_value = [
            {"symbol": "AAPL", "qty": 10, "side": "long", "current_price": 150.0},
        ]

        result = manager.sync_with_broker()

        self.assertEqual(len(result["removed"]), 1)
        self.assertIn("MSFT", result["removed"])
        self.assertEqual(len(manager.list_positions()), 1)

    def test_sync_with_no_client(self):
        manager = TrailingStopManager(client=None, state_file=None, auto_sync=False)
        result = manager.sync_with_broker()

        self.assertEqual(result["added"], [])
        self.assertEqual(result["removed"], [])


if __name__ == "__main__":
    unittest.main()
