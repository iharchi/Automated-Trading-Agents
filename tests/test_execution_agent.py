"""Unit tests for the Execution Agent.

Tests cover:
    - Pre-flight checks (risk approval, position size, signal, market open)
    - Order ticket building from portfolio decisions
    - Dry-run mode (no API calls)
    - Retry logic on transient failures
    - Fill verification
    - Formatting
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from agents.execution_agent import (
    ExecutionAgent,
    OrderTicket,
    ExecutionResult,
)


# ── Helpers ──────────────────────────────────────────────────────

def _make_decision(
    symbol="AAPL",
    signal="BUY",
    risk_approved=True,
    position_size=10,
    current_price=150.0,
    stop_loss=145.0,
    take_profit=160.0,
    atr=3.5,
):
    return {
        "symbol": symbol,
        "signal": signal,
        "risk_approved": risk_approved,
        "position_size": position_size,
        "current_price": current_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "atr": atr,
        "combined_score": 0.45,
        "confidence": 0.45,
        "agent_signals": [],
    }


def _make_agent(dry_run=False, market_open=True):
    """Create an ExecutionAgent with a mocked AlpacaClient."""
    mock_client = MagicMock()
    mock_client.is_market_open.return_value = market_open
    agent = ExecutionAgent(client=mock_client, dry_run=dry_run, max_retries=1, retry_delay=0.01)
    return agent


# ── Pre-flight check tests ───────────────────────────────────────

class TestPreflightChecks(unittest.TestCase):

    def test_all_checks_pass(self):
        agent = _make_agent(market_open=True)
        decision = _make_decision()
        result = agent.analyze("AAPL", decision=decision)
        self.assertTrue(result["all_passed"])
        self.assertEqual(len(result["checks"]), 4)
        for chk in result["checks"]:
            self.assertTrue(chk["passed"], f"{chk['rule']} failed unexpectedly")

    def test_hold_signal_fails(self):
        agent = _make_agent()
        decision = _make_decision(signal="HOLD")
        result = agent.analyze("AAPL", decision=decision)
        self.assertFalse(result["all_passed"])
        signal_check = [c for c in result["checks"] if c["rule"] == "signal_actionable"][0]
        self.assertFalse(signal_check["passed"])

    def test_risk_not_approved_fails(self):
        agent = _make_agent()
        decision = _make_decision(risk_approved=False)
        result = agent.analyze("AAPL", decision=decision)
        self.assertFalse(result["all_passed"])

    def test_zero_position_size_fails(self):
        agent = _make_agent()
        decision = _make_decision(position_size=0)
        result = agent.analyze("AAPL", decision=decision)
        self.assertFalse(result["all_passed"])

    def test_market_closed_fails(self):
        agent = _make_agent(market_open=False)
        decision = _make_decision()
        result = agent.analyze("AAPL", decision=decision)
        self.assertFalse(result["all_passed"])
        market_check = [c for c in result["checks"] if c["rule"] == "market_open"][0]
        self.assertFalse(market_check["passed"])

    def test_market_check_exception(self):
        agent = _make_agent()
        agent.client.is_market_open.side_effect = Exception("network error")
        decision = _make_decision()
        result = agent.analyze("AAPL", decision=decision)
        self.assertFalse(result["all_passed"])


# ── Order ticket building ─────────────────────────────────────────

class TestOrderTicket(unittest.TestCase):

    def test_buy_ticket(self):
        decision = _make_decision(signal="BUY", position_size=15)
        ticket = ExecutionAgent._build_ticket(decision)
        self.assertEqual(ticket.symbol, "AAPL")
        self.assertEqual(ticket.side, "buy")
        self.assertEqual(ticket.qty, 15)
        self.assertEqual(ticket.order_type, "market")
        self.assertEqual(ticket.stop_price, 145.0)
        self.assertEqual(ticket.take_profit_price, 160.0)

    def test_sell_ticket(self):
        decision = _make_decision(signal="SELL", position_size=5)
        ticket = ExecutionAgent._build_ticket(decision)
        self.assertEqual(ticket.side, "sell")
        self.assertEqual(ticket.qty, 5)

    def test_no_stop_loss(self):
        decision = _make_decision(stop_loss=0)
        ticket = ExecutionAgent._build_ticket(decision)
        self.assertIsNone(ticket.stop_price)

    def test_no_take_profit(self):
        decision = _make_decision(take_profit=0)
        ticket = ExecutionAgent._build_ticket(decision)
        self.assertIsNone(ticket.take_profit_price)


# ── Dry-run mode ──────────────────────────────────────────────────

class TestDryRun(unittest.TestCase):

    def test_dry_run_does_not_submit(self):
        agent = _make_agent(dry_run=True)
        decision = _make_decision()
        analysis = agent.analyze("AAPL", decision=decision)
        result = agent.execute("AAPL", analysis)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["qty"], 10)
        self.assertEqual(result["side"], "buy")
        agent.client.submit_order.assert_not_called()

    def test_dry_run_includes_stop_and_target(self):
        agent = _make_agent(dry_run=True)
        decision = _make_decision(stop_loss=145.0, take_profit=160.0)
        analysis = agent.analyze("AAPL", decision=decision)
        result = agent.execute("AAPL", analysis)
        self.assertAlmostEqual(result["stop_loss"], 145.0)
        self.assertAlmostEqual(result["take_profit"], 160.0)


# ── Execution (mocked submission) ─────────────────────────────────

class TestExecution(unittest.TestCase):

    def test_successful_order(self):
        agent = _make_agent()
        agent.client.submit_order.return_value = {
            "id": "order-123",
            "symbol": "AAPL",
            "qty": 10,
            "side": "buy",
            "type": "market",
            "status": "accepted",
        }
        # Mock fill check
        mock_order = MagicMock()
        mock_order.status = "filled"
        mock_order.filled_qty = 10
        mock_order.filled_avg_price = 150.25
        agent.client.api.get_order.return_value = mock_order

        decision = _make_decision()
        analysis = agent.analyze("AAPL", decision=decision)
        result = agent.execute("AAPL", analysis)

        self.assertEqual(result["order_id"], "order-123")
        self.assertEqual(result["status"], "filled")
        self.assertEqual(result["filled_qty"], 10)
        self.assertAlmostEqual(result["filled_avg_price"], 150.25)

    def test_failed_order_with_retry(self):
        agent = _make_agent()
        agent.client.submit_order.side_effect = Exception("API timeout")

        decision = _make_decision()
        analysis = agent.analyze("AAPL", decision=decision)
        result = agent.execute("AAPL", analysis)

        self.assertEqual(result["status"], "failed")
        self.assertIn("API timeout", result["error"])
        self.assertEqual(result["retries"], 1)  # max_retries=1

    def test_skipped_when_checks_fail(self):
        agent = _make_agent()
        decision = _make_decision(signal="HOLD")
        analysis = agent.analyze("AAPL", decision=decision)
        result = agent.execute("AAPL", analysis)
        self.assertEqual(result["status"], "skipped")
        agent.client.submit_order.assert_not_called()

    def test_sell_order_no_bracket(self):
        """Sell orders should not use bracket order class."""
        agent = _make_agent()
        agent.client.submit_order.return_value = {
            "id": "sell-456",
            "symbol": "AAPL",
            "qty": 5,
            "side": "sell",
            "type": "market",
            "status": "accepted",
        }
        mock_order = MagicMock()
        mock_order.status = "filled"
        mock_order.filled_qty = 5
        mock_order.filled_avg_price = 149.0
        agent.client.api.get_order.return_value = mock_order

        decision = _make_decision(signal="SELL", position_size=5)
        analysis = agent.analyze("AAPL", decision=decision)
        result = agent.execute("AAPL", analysis)

        self.assertEqual(result["side"], "sell")
        # submit_order should have been called without bracket params
        call_kwargs = agent.client.submit_order.call_args
        self.assertNotIn("order_class", call_kwargs.kwargs if call_kwargs.kwargs else {})


# ── Fill verification ─────────────────────────────────────────────

class TestFillCheck(unittest.TestCase):

    def test_fill_returns_none_on_exception(self):
        agent = _make_agent()
        agent.client.api.get_order.side_effect = Exception("not found")
        result = agent._check_fill("bad-id")
        self.assertIsNone(result)

    def test_fill_returns_status(self):
        agent = _make_agent()
        mock_order = MagicMock()
        mock_order.status = "filled"
        mock_order.filled_qty = 10
        mock_order.filled_avg_price = 150.0
        agent.client.api.get_order.return_value = mock_order

        result = agent._check_fill("order-123")
        self.assertEqual(result["status"], "filled")
        self.assertEqual(result["filled_qty"], 10)


# ── Formatting ────────────────────────────────────────────────────

class TestFormatting(unittest.TestCase):

    def test_format_produces_string(self):
        result = ExecutionResult(
            symbol="AAPL",
            side="buy",
            qty=10,
            order_type="market",
            status="filled",
            order_id="test-123",
            filled_qty=10,
            filled_avg_price=150.0,
            checks=[{"rule": "risk_approved", "passed": True, "detail": "ok"}],
        )
        output = ExecutionAgent.format_analysis(result.__dict__)
        self.assertIn("AAPL", output)
        self.assertIn("FILLED", output)
        self.assertIn("test-123", output)

    def test_format_skipped(self):
        result = ExecutionResult(
            symbol="TSLA",
            status="skipped",
            error="Pre-flight checks failed",
            checks=[{"rule": "signal_actionable", "passed": False, "detail": "HOLD"}],
        )
        output = ExecutionAgent.format_analysis(result.__dict__)
        self.assertIn("TSLA", output)
        self.assertIn("SKIPPED", output)


if __name__ == "__main__":
    unittest.main()
