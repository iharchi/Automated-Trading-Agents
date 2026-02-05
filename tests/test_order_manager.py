"""Tests for utils/order_manager.py"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from utils.order_manager import Order, OrderManager, OrderStatus


# ── OrderStatus enum tests ──────────────────────────────────────

class TestOrderStatus:
    def test_all_values_present(self):
        expected = {
            "created", "submitted", "partial_fill", "filled",
            "cancelled", "rejected", "failed", "expired",
        }
        actual = {s.value for s in OrderStatus}
        assert actual == expected

    def test_is_str_enum(self):
        assert isinstance(OrderStatus.CREATED, str)
        assert OrderStatus.FILLED == "filled"


# ── Order dataclass tests ───────────────────────────────────────

class TestOrder:
    def test_post_init_sets_created_at(self):
        before = time.time()
        order = Order(order_id="t-1", symbol="AAPL", side="buy", qty=10)
        after = time.time()
        assert before <= order.created_at <= after

    def test_post_init_preserves_explicit_created_at(self):
        order = Order(order_id="t-1", symbol="AAPL", side="buy", qty=10, created_at=99999.0)
        assert order.created_at == 99999.0

    def test_is_terminal_for_filled(self):
        order = Order(order_id="t-1", symbol="X", side="buy", qty=1, status=OrderStatus.FILLED)
        assert order.is_terminal is True

    def test_is_terminal_for_cancelled(self):
        order = Order(order_id="t-1", symbol="X", side="buy", qty=1, status=OrderStatus.CANCELLED)
        assert order.is_terminal is True

    def test_is_terminal_for_rejected(self):
        order = Order(order_id="t-1", symbol="X", side="buy", qty=1, status=OrderStatus.REJECTED)
        assert order.is_terminal is True

    def test_is_terminal_for_failed(self):
        order = Order(order_id="t-1", symbol="X", side="buy", qty=1, status=OrderStatus.FAILED)
        assert order.is_terminal is True

    def test_is_terminal_for_expired(self):
        order = Order(order_id="t-1", symbol="X", side="buy", qty=1, status=OrderStatus.EXPIRED)
        assert order.is_terminal is True

    def test_is_not_terminal_for_created(self):
        order = Order(order_id="t-1", symbol="X", side="buy", qty=1, status=OrderStatus.CREATED)
        assert order.is_terminal is False

    def test_is_not_terminal_for_submitted(self):
        order = Order(order_id="t-1", symbol="X", side="buy", qty=1, status=OrderStatus.SUBMITTED)
        assert order.is_terminal is False

    def test_is_not_terminal_for_partial(self):
        order = Order(order_id="t-1", symbol="X", side="buy", qty=1, status=OrderStatus.PARTIAL)
        assert order.is_terminal is False

    def test_remaining_qty(self):
        order = Order(order_id="t-1", symbol="X", side="buy", qty=100, filled_qty=30)
        assert order.remaining_qty == 70

    def test_remaining_qty_fully_filled(self):
        order = Order(order_id="t-1", symbol="X", side="buy", qty=50, filled_qty=50)
        assert order.remaining_qty == 0

    def test_fill_pct(self):
        order = Order(order_id="t-1", symbol="X", side="buy", qty=200, filled_qty=50)
        assert order.fill_pct == pytest.approx(0.25)

    def test_fill_pct_zero_qty(self):
        order = Order(order_id="t-1", symbol="X", side="buy", qty=0)
        assert order.fill_pct == 0.0


# ── OrderManager.__init__ tests ─────────────────────────────────

class TestOrderManagerInit:
    def test_init_without_client(self):
        oms = OrderManager()
        assert oms.client is None
        assert oms.total_orders == 0
        assert oms._next_id == 1

    def test_init_with_client(self):
        mock_client = MagicMock()
        oms = OrderManager(client=mock_client)
        assert oms.client is mock_client


# ── create_order tests ──────────────────────────────────────────

class TestCreateOrder:
    def test_creates_order_with_correct_attributes(self):
        oms = OrderManager()
        oid = oms.create_order("AAPL", "buy", 50, order_type="limit", limit_price=150.0)
        order = oms.get_order(oid)
        assert order is not None
        assert order.symbol == "AAPL"
        assert order.side == "buy"
        assert order.qty == 50
        assert order.order_type == "limit"
        assert order.limit_price == 150.0
        assert order.status == OrderStatus.CREATED

    def test_increments_order_ids(self):
        oms = OrderManager()
        oid1 = oms.create_order("AAPL", "buy", 10)
        oid2 = oms.create_order("MSFT", "sell", 20)
        oid3 = oms.create_order("GOOG", "buy", 30)
        assert oid1 == "oms-000001"
        assert oid2 == "oms-000002"
        assert oid3 == "oms-000003"

    def test_all_keyword_args_propagated(self):
        oms = OrderManager()
        oid = oms.create_order(
            "TSLA", "sell", 100,
            order_type="stop_limit",
            time_in_force="gtc",
            limit_price=200.0,
            stop_price=195.0,
            reason="Take profit",
            source="StrategyA",
        )
        order = oms.get_order(oid)
        assert order.order_type == "stop_limit"
        assert order.time_in_force == "gtc"
        assert order.limit_price == 200.0
        assert order.stop_price == 195.0
        assert order.reason == "Take profit"
        assert order.source == "StrategyA"

    def test_default_keyword_args(self):
        oms = OrderManager()
        oid = oms.create_order("SPY", "buy", 5)
        order = oms.get_order(oid)
        assert order.order_type == "market"
        assert order.time_in_force == "day"
        assert order.limit_price is None
        assert order.stop_price is None
        assert order.reason == ""
        assert order.source == ""


# ── submit tests ────────────────────────────────────────────────

class TestSubmit:
    def test_order_not_found_returns_false(self):
        oms = OrderManager()
        assert oms.submit("nonexistent-id") is False

    def test_non_created_status_returns_false(self):
        oms = OrderManager()
        oid = oms.create_order("AAPL", "buy", 10)
        order = oms.get_order(oid)
        order.status = OrderStatus.SUBMITTED
        assert oms.submit(oid) is False

    def test_dry_run_immediate_fill(self):
        oms = OrderManager()  # No client -> dry-run
        oid = oms.create_order("AAPL", "buy", 50, limit_price=150.0)
        result = oms.submit(oid)
        assert result is True
        order = oms.get_order(oid)
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 50
        assert order.filled_avg_price == 150.0
        assert order.broker_order_id == f"dry-{oid}"
        assert order.submitted_at > 0
        assert order.filled_at > 0

    def test_dry_run_no_limit_price(self):
        oms = OrderManager()
        oid = oms.create_order("AAPL", "buy", 50)
        oms.submit(oid)
        order = oms.get_order(oid)
        assert order.filled_avg_price == 0.0

    def test_with_mock_client_successful_submission(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {
            "id": "broker-abc-123",
            "status": "submitted",
        }
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 50)
        result = oms.submit(oid)
        assert result is True
        order = oms.get_order(oid)
        assert order.status == OrderStatus.SUBMITTED
        assert order.broker_order_id == "broker-abc-123"
        assert order.submitted_at > 0
        mock_client.submit_order.assert_called_once_with(
            symbol="AAPL",
            qty=50,
            side="buy",
            order_type="market",
            time_in_force="day",
        )

    def test_client_returns_filled_immediately(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {
            "id": "broker-xyz",
            "status": "filled",
        }
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 25)
        oms.submit(oid)
        order = oms.get_order(oid)
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 25
        assert order.filled_at > 0

    def test_client_returns_partially_filled(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {
            "id": "broker-pf",
            "status": "partially_filled",
        }
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 100)
        oms.submit(oid)
        order = oms.get_order(oid)
        assert order.status == OrderStatus.PARTIAL

    def test_client_raises_exception_sets_failed(self):
        mock_client = MagicMock()
        mock_client.submit_order.side_effect = RuntimeError("Connection lost")
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 10)
        result = oms.submit(oid)
        assert result is False
        order = oms.get_order(oid)
        assert order.status == OrderStatus.FAILED
        assert "Connection lost" in order.error


# ── cancel tests ────────────────────────────────────────────────

class TestCancel:
    def test_order_not_found_returns_false(self):
        oms = OrderManager()
        assert oms.cancel("nonexistent") is False

    def test_terminal_order_returns_false(self):
        oms = OrderManager()
        oid = oms.create_order("AAPL", "buy", 10)
        oms.submit(oid)  # dry-run fills immediately
        assert oms.get_order(oid).status == OrderStatus.FILLED
        assert oms.cancel(oid) is False

    def test_successful_cancel_no_client(self):
        oms = OrderManager()
        oid = oms.create_order("AAPL", "buy", 10)
        # Order is CREATED, not terminal -> can cancel
        result = oms.cancel(oid)
        assert result is True
        order = oms.get_order(oid)
        assert order.status == OrderStatus.CANCELLED
        assert order.cancelled_at > 0

    def test_cancel_with_broker_client(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {"id": "broker-1", "status": "submitted"}
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 10)
        oms.submit(oid)
        result = oms.cancel(oid)
        assert result is True
        mock_client.api.cancel_order.assert_called_once_with("broker-1")
        order = oms.get_order(oid)
        assert order.status == OrderStatus.CANCELLED

    def test_broker_cancel_fails_but_local_cancel_succeeds(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {"id": "broker-2", "status": "submitted"}
        mock_client.api.cancel_order.side_effect = RuntimeError("Broker unreachable")
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 10)
        oms.submit(oid)
        result = oms.cancel(oid)
        assert result is True
        order = oms.get_order(oid)
        assert order.status == OrderStatus.CANCELLED

    def test_cancel_created_order_with_client_no_broker_id(self):
        mock_client = MagicMock()
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 10)
        # Not submitted, so no broker_order_id -> skip broker cancel
        result = oms.cancel(oid)
        assert result is True
        mock_client.api.cancel_order.assert_not_called()


# ── check_fills tests ──────────────────────────────────────────

class TestCheckFills:
    def _make_broker_order(self, status, filled_qty=None, filled_avg_price=None):
        return SimpleNamespace(
            status=status,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
        )

    def test_no_open_orders_returns_empty(self):
        oms = OrderManager()
        assert oms.check_fills() == []

    def test_no_client_returns_empty(self):
        oms = OrderManager()
        oid = oms.create_order("AAPL", "buy", 10)
        # Manually set to SUBMITTED with a broker_order_id to test client=None path
        order = oms.get_order(oid)
        order.status = OrderStatus.SUBMITTED
        order.broker_order_id = "broker-x"
        assert oms.check_fills() == []

    def test_broker_returns_filled(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {"id": "b-1", "status": "submitted"}
        mock_client.api.get_order.return_value = self._make_broker_order(
            "filled", filled_qty=50, filled_avg_price=152.30
        )
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 50)
        oms.submit(oid)
        updated = oms.check_fills()
        assert oid in updated
        order = oms.get_order(oid)
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 50
        assert order.filled_avg_price == pytest.approx(152.30)
        assert order.filled_at > 0

    def test_broker_returns_partially_filled(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {"id": "b-2", "status": "submitted"}
        mock_client.api.get_order.return_value = self._make_broker_order(
            "partially_filled", filled_qty=20, filled_avg_price=100.0
        )
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 100)
        oms.submit(oid)
        updated = oms.check_fills()
        assert oid in updated
        order = oms.get_order(oid)
        assert order.status == OrderStatus.PARTIAL
        assert order.filled_qty == 20
        assert order.filled_avg_price == pytest.approx(100.0)

    def test_broker_returns_cancelled(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {"id": "b-3", "status": "submitted"}
        mock_client.api.get_order.return_value = self._make_broker_order("cancelled")
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 10)
        oms.submit(oid)
        updated = oms.check_fills()
        assert oid in updated
        assert oms.get_order(oid).status == OrderStatus.CANCELLED

    def test_broker_returns_canceled_american_spelling(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {"id": "b-3b", "status": "submitted"}
        mock_client.api.get_order.return_value = self._make_broker_order("canceled")
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 10)
        oms.submit(oid)
        updated = oms.check_fills()
        assert oid in updated
        assert oms.get_order(oid).status == OrderStatus.CANCELLED

    def test_broker_returns_expired(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {"id": "b-4", "status": "submitted"}
        mock_client.api.get_order.return_value = self._make_broker_order("expired")
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 10)
        oms.submit(oid)
        updated = oms.check_fills()
        assert oid in updated
        assert oms.get_order(oid).status == OrderStatus.EXPIRED

    def test_broker_returns_rejected(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {"id": "b-5", "status": "submitted"}
        mock_client.api.get_order.return_value = self._make_broker_order("rejected")
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 10)
        oms.submit(oid)
        updated = oms.check_fills()
        assert oid in updated
        order = oms.get_order(oid)
        assert order.status == OrderStatus.REJECTED
        assert order.error == "Rejected by broker"

    def test_broker_raises_exception_skips_order(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {"id": "b-err", "status": "submitted"}
        mock_client.api.get_order.side_effect = RuntimeError("timeout")
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 10)
        oms.submit(oid)
        updated = oms.check_fills()
        assert updated == []
        # Order remains in its previous state (SUBMITTED)
        assert oms.get_order(oid).status == OrderStatus.SUBMITTED

    def test_terminal_orders_are_skipped(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {"id": "b-t", "status": "filled"}
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 10)
        oms.submit(oid)  # fills immediately
        assert oms.get_order(oid).status == OrderStatus.FILLED
        updated = oms.check_fills()
        assert updated == []
        mock_client.api.get_order.assert_not_called()

    def test_orders_without_broker_id_are_skipped(self):
        mock_client = MagicMock()
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 10)
        # Still CREATED, no broker_order_id
        updated = oms.check_fills()
        assert updated == []
        mock_client.api.get_order.assert_not_called()

    def test_filled_qty_defaults_to_order_qty_when_none(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {"id": "b-def", "status": "submitted"}
        mock_client.api.get_order.return_value = self._make_broker_order(
            "filled", filled_qty=None, filled_avg_price=None
        )
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 75)
        oms.submit(oid)
        oms.check_fills()
        order = oms.get_order(oid)
        assert order.filled_qty == 75
        assert order.filled_avg_price == 0.0


# ── reconcile tests ────────────────────────────────────────────

class TestReconcile:
    def test_no_client_returns_zeros(self):
        oms = OrderManager()
        result = oms.reconcile()
        assert result == {"reconciled": 0, "mismatches": 0}

    def test_mismatch_gets_corrected(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {"id": "b-rec", "status": "submitted"}
        broker_order = SimpleNamespace(
            status="filled",
            filled_qty=50,
            filled_avg_price=155.0,
        )
        mock_client.api.get_order.return_value = broker_order
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 50)
        oms.submit(oid)
        # Order is SUBMITTED locally, but broker says filled -> mismatch
        result = oms.reconcile()
        assert result["reconciled"] == 1
        assert result["mismatches"] == 1
        order = oms.get_order(oid)
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 50
        assert order.filled_avg_price == pytest.approx(155.0)
        assert order.filled_at > 0

    def test_reconcile_no_mismatch(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {"id": "b-ok", "status": "submitted"}
        broker_order = SimpleNamespace(
            status="submitted",
            filled_qty=0,
            filled_avg_price=0,
        )
        mock_client.api.get_order.return_value = broker_order
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 50)
        oms.submit(oid)
        result = oms.reconcile()
        assert result["reconciled"] == 1
        assert result["mismatches"] == 0

    def test_reconcile_skips_terminal_orders(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {"id": "b-term", "status": "filled"}
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 10)
        oms.submit(oid)  # Broker says filled -> terminal
        result = oms.reconcile()
        assert result["reconciled"] == 0
        mock_client.api.get_order.assert_not_called()

    def test_reconcile_skips_orders_without_broker_id(self):
        mock_client = MagicMock()
        oms = OrderManager(client=mock_client)
        oms.create_order("AAPL", "buy", 10)  # CREATED, no broker id
        result = oms.reconcile()
        assert result["reconciled"] == 0

    def test_reconcile_handles_exception(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = {"id": "b-exc", "status": "submitted"}
        mock_client.api.get_order.side_effect = RuntimeError("API error")
        oms = OrderManager(client=mock_client)
        oid = oms.create_order("AAPL", "buy", 10)
        oms.submit(oid)
        result = oms.reconcile()
        # Exception prevents reconciliation count from incrementing
        assert result["reconciled"] == 0
        assert result["mismatches"] == 0


# ── Query method tests ──────────────────────────────────────────

class TestQueryMethods:
    def test_get_order_found(self):
        oms = OrderManager()
        oid = oms.create_order("AAPL", "buy", 10)
        order = oms.get_order(oid)
        assert order is not None
        assert order.order_id == oid

    def test_get_order_not_found(self):
        oms = OrderManager()
        assert oms.get_order("nonexistent") is None

    def test_get_open_orders(self):
        oms = OrderManager()
        oid1 = oms.create_order("AAPL", "buy", 10)
        oid2 = oms.create_order("MSFT", "sell", 20)
        oms.submit(oid1)  # dry-run -> filled (terminal)
        open_orders = oms.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].order_id == oid2

    def test_get_filled_orders(self):
        oms = OrderManager()
        oid1 = oms.create_order("AAPL", "buy", 10)
        oid2 = oms.create_order("MSFT", "sell", 20)
        oms.submit(oid1)  # filled
        oms.submit(oid2)  # filled
        filled = oms.get_filled_orders()
        assert len(filled) == 2

    def test_get_orders_by_symbol(self):
        oms = OrderManager()
        oms.create_order("AAPL", "buy", 10)
        oms.create_order("MSFT", "sell", 20)
        oms.create_order("AAPL", "sell", 5)
        aapl_orders = oms.get_orders_by_symbol("AAPL")
        assert len(aapl_orders) == 2
        assert all(o.symbol == "AAPL" for o in aapl_orders)

    def test_get_orders_by_symbol_none_found(self):
        oms = OrderManager()
        oms.create_order("AAPL", "buy", 10)
        assert oms.get_orders_by_symbol("TSLA") == []

    def test_get_blotter_ordering(self):
        oms = OrderManager()
        oid1 = oms.create_order("AAPL", "buy", 10)
        oid2 = oms.create_order("MSFT", "sell", 20)
        oid3 = oms.create_order("GOOG", "buy", 30)
        blotter = oms.get_blotter()
        # Newest first (oid3 was created last)
        assert blotter[0].order_id == oid3
        assert blotter[-1].order_id == oid1

    def test_get_blotter_limit(self):
        oms = OrderManager()
        for i in range(10):
            oms.create_order("AAPL", "buy", 1)
        blotter = oms.get_blotter(limit=3)
        assert len(blotter) == 3

    def test_get_blotter_default_limit(self):
        oms = OrderManager()
        for i in range(5):
            oms.create_order("AAPL", "buy", 1)
        blotter = oms.get_blotter()
        assert len(blotter) == 5  # fewer than 50 -> returns all


# ── Property tests ──────────────────────────────────────────────

class TestProperties:
    def test_total_orders(self):
        oms = OrderManager()
        assert oms.total_orders == 0
        oms.create_order("AAPL", "buy", 10)
        assert oms.total_orders == 1
        oms.create_order("MSFT", "sell", 20)
        assert oms.total_orders == 2

    def test_open_count(self):
        oms = OrderManager()
        oms.create_order("AAPL", "buy", 10)
        oms.create_order("MSFT", "sell", 20)
        assert oms.open_count == 2
        # Submit one (dry-run fills it)
        oms.submit("oms-000001")
        assert oms.open_count == 1

    def test_filled_count(self):
        oms = OrderManager()
        oid1 = oms.create_order("AAPL", "buy", 10)
        oid2 = oms.create_order("MSFT", "sell", 20)
        assert oms.filled_count == 0
        oms.submit(oid1)
        assert oms.filled_count == 1
        oms.submit(oid2)
        assert oms.filled_count == 2


# ── Formatting tests ────────────────────────────────────────────

class TestFormatting:
    def test_format_order_basic(self):
        order = Order(
            order_id="oms-000001",
            symbol="AAPL",
            side="buy",
            qty=50,
            order_type="market",
            status=OrderStatus.CREATED,
            created_at=1700000000.0,
        )
        text = OrderManager.format_order(order)
        assert "oms-000001" in text
        assert "BUY" in text
        assert "50" in text
        assert "AAPL" in text
        assert "market" in text

    def test_format_order_with_fill_info(self):
        order = Order(
            order_id="oms-000002",
            symbol="MSFT",
            side="sell",
            qty=100,
            order_type="limit",
            status=OrderStatus.FILLED,
            filled_qty=100,
            filled_avg_price=350.50,
            created_at=1700000000.0,
        )
        text = OrderManager.format_order(order)
        assert "filled=100/100" in text
        assert "@$350.50" in text

    def test_format_order_no_fill_no_price(self):
        order = Order(
            order_id="oms-000003",
            symbol="GOOG",
            side="buy",
            qty=10,
            order_type="market",
            status=OrderStatus.SUBMITTED,
            filled_qty=0,
            filled_avg_price=0.0,
            created_at=1700000000.0,
        )
        text = OrderManager.format_order(order)
        assert "filled=" not in text
        assert "@$" not in text

    def test_format_blotter_empty(self):
        text = OrderManager.format_blotter([])
        assert "No orders." in text

    def test_format_blotter_with_orders(self):
        orders = [
            Order(
                order_id="oms-000001",
                symbol="AAPL",
                side="buy",
                qty=50,
                created_at=1700000000.0,
            ),
            Order(
                order_id="oms-000002",
                symbol="MSFT",
                side="sell",
                qty=100,
                created_at=1700000001.0,
            ),
        ]
        text = OrderManager.format_blotter(orders)
        assert "ORDER BLOTTER" in text
        assert "2 orders" in text
        assert "oms-000001" in text
        assert "oms-000002" in text
        assert "=" * 70 in text
