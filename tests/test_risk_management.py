"""Tests for Risk Management Agent position sizing and rule checks.

All tests use mock portfolio data — no Alpaca API keys required.
"""

import unittest

from agents.risk_management_agent import RiskManagementAgent


class TestPositionSizing(unittest.TestCase):
    """Test ATR-based position sizing calculations."""

    def _make_agent(self, **kwargs) -> RiskManagementAgent:
        """Create agent without connecting to Alpaca."""
        agent = object.__new__(RiskManagementAgent)
        agent.max_position_pct = kwargs.get("max_position_pct", 0.10)
        agent.max_portfolio_exposure = kwargs.get("max_portfolio_exposure", 0.90)
        agent.risk_per_trade_pct = kwargs.get("risk_per_trade", 0.02)
        agent.atr_stop_multiplier = kwargs.get("atr_stop_mult", 1.5)
        agent.take_profit_ratio = kwargs.get("tp_ratio", 2.0)
        return agent

    def test_basic_sizing(self):
        """With $100k equity, $150 price, ATR=3, expect sensible share count."""
        agent = self._make_agent()
        shares, stop, risk = agent._calculate_position_size(
            equity=100_000, price=150.0, atr=3.0
        )
        # risk_budget = 100k * 0.02 = 2000
        # stop_distance = 3.0 * 1.5 = 4.5
        # shares = int(2000 / 4.5) = 444
        self.assertEqual(shares, 444)
        self.assertAlmostEqual(stop, 150.0 - 4.5, places=2)
        self.assertGreater(risk, 0)

    def test_zero_atr_returns_zero(self):
        """ATR of 0 should produce 0 shares (can't size without volatility)."""
        agent = self._make_agent()
        shares, stop, risk = agent._calculate_position_size(
            equity=100_000, price=150.0, atr=0.0
        )
        self.assertEqual(shares, 0)

    def test_zero_price_returns_zero(self):
        agent = self._make_agent()
        shares, stop, risk = agent._calculate_position_size(
            equity=100_000, price=0.0, atr=3.0
        )
        self.assertEqual(shares, 0)

    def test_small_equity_limits_shares(self):
        """Small account should produce fewer shares."""
        agent = self._make_agent()
        shares, _, _ = agent._calculate_position_size(
            equity=5_000, price=150.0, atr=3.0
        )
        # risk_budget = 5k * 0.02 = 100
        # shares = int(100 / 4.5) = 22
        self.assertEqual(shares, 22)

    def test_high_atr_reduces_shares(self):
        """Higher ATR (more volatile) should reduce position size."""
        agent = self._make_agent()
        shares_low_vol, _, _ = agent._calculate_position_size(
            equity=100_000, price=150.0, atr=2.0
        )
        shares_high_vol, _, _ = agent._calculate_position_size(
            equity=100_000, price=150.0, atr=10.0
        )
        self.assertGreater(shares_low_vol, shares_high_vol)


class TestRiskChecks(unittest.TestCase):
    """Test individual risk rule checks."""

    def _make_agent(self) -> RiskManagementAgent:
        agent = object.__new__(RiskManagementAgent)
        agent.max_position_pct = 0.10
        agent.max_portfolio_exposure = 0.90
        return agent

    def test_concentration_pass(self):
        """Small position relative to equity should pass."""
        agent = self._make_agent()
        check = agent._check_concentration(
            symbol="AAPL", price=150.0, shares=10,
            equity=100_000, existing_value=0,
        )
        self.assertTrue(check.passed)

    def test_concentration_fail(self):
        """Large position relative to equity should fail."""
        agent = self._make_agent()
        check = agent._check_concentration(
            symbol="AAPL", price=150.0, shares=100,
            equity=100_000, existing_value=5_000,
        )
        # (5000 + 100*150) / 100000 = 20000/100000 = 20% > 10%
        self.assertFalse(check.passed)

    def test_exposure_pass(self):
        """Total exposure within limit should pass."""
        agent = self._make_agent()
        check = agent._check_exposure(
            total_market_value=50_000, new_trade_value=10_000, equity=100_000
        )
        # (50000 + 10000) / 100000 = 60% < 90%
        self.assertTrue(check.passed)

    def test_exposure_fail(self):
        """Total exposure exceeding limit should fail."""
        agent = self._make_agent()
        check = agent._check_exposure(
            total_market_value=85_000, new_trade_value=10_000, equity=100_000
        )
        # (85000 + 10000) / 100000 = 95% > 90%
        self.assertFalse(check.passed)

    def test_buying_power_pass(self):
        agent = self._make_agent()
        check = agent._check_buying_power(
            buying_power=50_000, price=150.0, shares=10
        )
        self.assertTrue(check.passed)

    def test_buying_power_fail(self):
        agent = self._make_agent()
        check = agent._check_buying_power(
            buying_power=1_000, price=150.0, shares=10
        )
        self.assertFalse(check.passed)

    def test_sell_position_pass(self):
        agent = self._make_agent()
        positions = {"AAPL": {"qty": 50, "market_value": 7500}}
        check = agent._check_position_for_sell("AAPL", positions)
        self.assertTrue(check.passed)

    def test_sell_position_fail(self):
        agent = self._make_agent()
        check = agent._check_position_for_sell("AAPL", {})
        self.assertFalse(check.passed)


if __name__ == "__main__":
    unittest.main()
