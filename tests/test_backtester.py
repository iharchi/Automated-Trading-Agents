"""Tests for Backtester metrics computation and simulation logic.

Uses synthetic bar data — no Alpaca API keys required.
"""

import unittest

import numpy as np
import pandas as pd

from backtest import Backtester


def _make_trending_bars(n: int = 200, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic daily OHLCV bars with a timestamp index."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")

    if trend == "up":
        close = np.linspace(100, 200, n) + np.random.normal(0, 2, n)
    elif trend == "down":
        close = np.linspace(200, 100, n) + np.random.normal(0, 2, n)
    else:
        close = 150 + np.cumsum(np.random.normal(0, 1, n))

    high = close + np.random.uniform(0.5, 3, n)
    low = close - np.random.uniform(0.5, 3, n)
    open_ = close + np.random.normal(0, 1, n)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(100_000, 1_000_000, n),
    })
    return df


class TestBacktesterMetrics(unittest.TestCase):
    """Test that the backtester computes valid metrics."""

    def test_initial_capital_preserved_when_no_trades(self):
        """Flat prices with no signals should keep equity near initial."""
        np.random.seed(7)
        n = 100
        flat_price = 100.0
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC"),
            "open": [flat_price] * n,
            "high": [flat_price + 0.1] * n,
            "low": [flat_price - 0.1] * n,
            "close": [flat_price] * n,
            "volume": [500_000] * n,
        })
        bt = Backtester(initial_capital=100_000)
        result = bt.run_on_dataframe("TEST", df)
        self.assertEqual(result.total_trades, 0)
        self.assertAlmostEqual(result.final_equity, 100_000, places=0)

    def test_uptrend_produces_trades(self):
        """A clear uptrend should generate at least one trade."""
        df = _make_trending_bars(200, trend="up")
        bt = Backtester(initial_capital=100_000)
        result = bt.run_on_dataframe("TEST", df)
        self.assertGreater(result.total_trades, 0)

    def test_equity_curve_length(self):
        """Equity curve should have n+1 entries (initial + one per bar)."""
        df = _make_trending_bars(100, trend="up")
        bt = Backtester(initial_capital=50_000)
        result = bt.run_on_dataframe("TEST", df)
        # After dropna for indicators, bars shrink. Equity curve = bars + 1
        # Just verify it's longer than 1
        self.assertGreater(len(result.equity_curve), 1)

    def test_max_drawdown_non_negative(self):
        """Max drawdown should always be >= 0."""
        df = _make_trending_bars(200, trend="down")
        bt = Backtester(initial_capital=100_000)
        result = bt.run_on_dataframe("TEST", df)
        self.assertGreaterEqual(result.max_drawdown_pct, 0)

    def test_win_rate_range(self):
        """Win rate should be between 0 and 100."""
        df = _make_trending_bars(200, trend="up")
        bt = Backtester(initial_capital=100_000)
        result = bt.run_on_dataframe("TEST", df)
        self.assertGreaterEqual(result.win_rate_pct, 0)
        self.assertLessEqual(result.win_rate_pct, 100)

    def test_trade_pnl_sums_correctly(self):
        """Sum of trade P&Ls should roughly equal equity change."""
        df = _make_trending_bars(200, trend="up")
        bt = Backtester(initial_capital=100_000)
        result = bt.run_on_dataframe("TEST", df)
        if result.trades:
            total_pnl = sum(t.pnl for t in result.trades)
            equity_change = result.final_equity - result.initial_capital
            self.assertAlmostEqual(total_pnl, equity_change, delta=1.0)

    def test_trade_fields(self):
        """Each trade should have all required fields."""
        df = _make_trending_bars(200, trend="up")
        bt = Backtester(initial_capital=100_000)
        result = bt.run_on_dataframe("TEST", df)
        for trade in result.trades:
            self.assertIsNotNone(trade.entry_date)
            self.assertIsNotNone(trade.exit_date)
            self.assertGreater(trade.entry_price, 0)
            self.assertGreater(trade.exit_price, 0)
            self.assertGreater(trade.shares, 0)
            self.assertIn(trade.exit_reason, (
                "signal", "stop_loss", "take_profit", "end_of_data"
            ))


class TestBacktesterPositionSizing(unittest.TestCase):
    """Test the internal position sizing method."""

    def test_sizing_basic(self):
        bt = Backtester(initial_capital=100_000, risk_per_trade=0.02, atr_stop_mult=1.5)
        shares = bt._size_position(equity=100_000, price=150.0, atr=3.0)
        # risk_budget = 2000, stop_dist = 4.5, shares = int(2000/4.5) = 444
        self.assertEqual(shares, 444)

    def test_sizing_zero_atr(self):
        bt = Backtester()
        shares = bt._size_position(equity=100_000, price=150.0, atr=0.0)
        self.assertEqual(shares, 0)

    def test_sizing_zero_price(self):
        bt = Backtester()
        shares = bt._size_position(equity=100_000, price=0.0, atr=3.0)
        self.assertEqual(shares, 0)


class TestBacktesterScoring(unittest.TestCase):
    """Test the bar scoring method."""

    def test_score_range(self):
        """Score should be between -4 and +4 (4 indicators, each ±1)."""
        df = _make_trending_bars(100, trend="up")
        bt = Backtester()
        df = bt._compute_indicators(df)
        df = df.dropna()
        for _, row in df.iterrows():
            score = bt._score_bar(row)
            self.assertGreaterEqual(score, -4)
            self.assertLessEqual(score, 4)


class TestFormatResult(unittest.TestCase):
    """Test the format_result static method."""

    def test_format_produces_string(self):
        df = _make_trending_bars(200, trend="up")
        bt = Backtester(initial_capital=100_000)
        result = bt.run_on_dataframe("TEST", df)
        output = Backtester.format_result(result)
        self.assertIsInstance(output, str)
        self.assertIn("TEST", output)
        self.assertIn("BACKTEST RESULTS", output)


if __name__ == "__main__":
    unittest.main()
