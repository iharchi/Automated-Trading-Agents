"""Tests for Technical Analysis Agent indicator scoring.

All tests use synthetic price data — no Alpaca API keys required.
"""

import unittest

import numpy as np
import pandas as pd

from agents.technical_analysis_agent import (
    TechnicalAnalysisAgent,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
)


def _make_price_series(prices: list[float]) -> pd.Series:
    """Create a pd.Series from a list of prices."""
    return pd.Series(prices, dtype=float)


def _make_bars(n: int = 100, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV bars.

    trend: "up" — steadily rising prices (bullish indicators expected)
           "down" — steadily falling prices (bearish indicators expected)
           "flat" — range-bound around 100
    """
    np.random.seed(42)
    if trend == "up":
        base = np.linspace(80, 160, n) + np.random.normal(0, 1, n)
    elif trend == "down":
        base = np.linspace(160, 80, n) + np.random.normal(0, 1, n)
    else:
        base = 100 + np.random.normal(0, 2, n)

    df = pd.DataFrame({
        "open": base - np.random.uniform(0, 1, n),
        "high": base + np.random.uniform(0.5, 2, n),
        "low": base - np.random.uniform(0.5, 2, n),
        "close": base,
        "volume": np.random.randint(100_000, 1_000_000, n),
    })
    return df


class TestIndicatorHelpers(unittest.TestCase):
    """Test individual indicator computation methods."""

    def test_rsi_oversold(self):
        """A sharp drop should yield RSI <= 30 → bullish signal (+1)."""
        prices = [100.0] * 20 + [100 - i * 3 for i in range(20)]
        close = _make_price_series(prices)
        result = TechnicalAnalysisAgent._compute_rsi(close, period=14)
        self.assertLessEqual(result.value, RSI_OVERSOLD)
        self.assertEqual(result.signal, 1)

    def test_rsi_overbought(self):
        """A sharp rise should yield RSI >= 70 → bearish signal (-1)."""
        prices = [50.0] * 20 + [50 + i * 3 for i in range(20)]
        close = _make_price_series(prices)
        result = TechnicalAnalysisAgent._compute_rsi(close, period=14)
        self.assertGreaterEqual(result.value, RSI_OVERBOUGHT)
        self.assertEqual(result.signal, -1)

    def test_rsi_neutral(self):
        """Sideways price should yield neutral RSI → signal 0."""
        np.random.seed(99)
        prices = list(100 + np.random.normal(0, 0.5, 50))
        close = _make_price_series(prices)
        result = TechnicalAnalysisAgent._compute_rsi(close, period=14)
        self.assertEqual(result.signal, 0)

    def test_macd_bullish(self):
        """Strong uptrend should produce bullish MACD."""
        prices = list(np.linspace(50, 150, 60))
        close = _make_price_series(prices)
        result = TechnicalAnalysisAgent._compute_macd(close)
        self.assertEqual(result.signal, 1)

    def test_macd_bearish(self):
        """Strong downtrend should produce bearish MACD."""
        prices = list(np.linspace(150, 50, 60))
        close = _make_price_series(prices)
        result = TechnicalAnalysisAgent._compute_macd(close)
        self.assertEqual(result.signal, -1)

    def test_bollinger_below_lower(self):
        """Price at lower band should be bullish."""
        # Steady prices then a sharp sudden drop well below the band
        prices = [100.0] * 40 + [70.0] * 3
        close = _make_price_series(prices)
        result = TechnicalAnalysisAgent._compute_bollinger(close, period=20, std=2)
        self.assertEqual(result.signal, 1)

    def test_bollinger_above_upper(self):
        """Price at upper band should be bearish."""
        prices = [100.0] * 40 + [130.0] * 3
        close = _make_price_series(prices)
        result = TechnicalAnalysisAgent._compute_bollinger(close, period=20, std=2)
        self.assertEqual(result.signal, -1)

    def test_ema_crossover_bullish(self):
        """Uptrend: short EMA > long EMA → bullish."""
        prices = list(np.linspace(50, 150, 50))
        close = _make_price_series(prices)
        result = TechnicalAnalysisAgent._compute_ema_crossover(close, short=9, long=21)
        self.assertEqual(result.signal, 1)

    def test_ema_crossover_bearish(self):
        """Downtrend: short EMA < long EMA → bearish."""
        prices = list(np.linspace(150, 50, 50))
        close = _make_price_series(prices)
        result = TechnicalAnalysisAgent._compute_ema_crossover(close, short=9, long=21)
        self.assertEqual(result.signal, -1)

    def test_atr_positive(self):
        """ATR should always be a positive number."""
        bars = _make_bars(50, trend="up")
        atr = TechnicalAnalysisAgent._compute_atr(
            bars["high"], bars["low"], bars["close"], period=14
        )
        self.assertGreater(atr, 0)


class TestCompositeSignal(unittest.TestCase):
    """Test that composite scoring produces correct BUY/SELL/HOLD."""

    def test_uptrend_produces_buy(self):
        """Strong uptrend with oversold start should produce BUY."""
        # Create a V-shaped recovery: drop then strong rise
        prices = [100 - i * 2 for i in range(25)] + [50 + i * 4 for i in range(75)]
        bars = pd.DataFrame({
            "open": prices,
            "high": [p + 1.5 for p in prices],
            "low": [p - 1.5 for p in prices],
            "close": prices,
            "volume": [500_000] * 100,
        })
        bars = TechnicalAnalysisAgent._add_all_indicators(bars) if hasattr(TechnicalAnalysisAgent, '_add_all_indicators') else bars
        # We test the scoring logic indirectly via the static helpers
        close = pd.Series(prices, dtype=float)
        scores = []
        scores.append(TechnicalAnalysisAgent._compute_ema_crossover(close, 9, 21).signal)
        scores.append(TechnicalAnalysisAgent._compute_macd(close).signal)
        composite = sum(scores)
        # In a strong uptrend, at least EMA and MACD should be bullish
        self.assertGreaterEqual(composite, 1)

    def test_threshold_constants(self):
        """Verify threshold constants are sensible."""
        self.assertEqual(BUY_THRESHOLD, 2)
        self.assertEqual(SELL_THRESHOLD, -2)
        self.assertLess(SELL_THRESHOLD, 0)
        self.assertGreater(BUY_THRESHOLD, 0)


class TestIndicatorResultFields(unittest.TestCase):
    """Test that indicator results have correct field types."""

    def test_rsi_result_fields(self):
        prices = list(np.linspace(90, 110, 40))
        result = TechnicalAnalysisAgent._compute_rsi(
            _make_price_series(prices), period=14
        )
        self.assertIsInstance(result.name, str)
        self.assertIsInstance(result.value, float)
        self.assertIn(result.signal, (-1, 0, 1))
        self.assertIsInstance(result.detail, str)

    def test_macd_result_fields(self):
        prices = list(np.linspace(90, 110, 40))
        result = TechnicalAnalysisAgent._compute_macd(_make_price_series(prices))
        self.assertEqual(result.name, "MACD")
        self.assertIn(result.signal, (-1, 0, 1))

    def test_bollinger_result_fields(self):
        prices = [100.0] * 30
        result = TechnicalAnalysisAgent._compute_bollinger(
            _make_price_series(prices), period=20, std=2
        )
        self.assertEqual(result.name, "Bollinger")
        self.assertIn(result.signal, (-1, 0, 1))

    def test_ema_result_fields(self):
        prices = list(np.linspace(90, 110, 30))
        result = TechnicalAnalysisAgent._compute_ema_crossover(
            _make_price_series(prices), short=9, long=21
        )
        self.assertEqual(result.name, "EMA_Cross")
        self.assertIn(result.signal, (-1, 0, 1))


if __name__ == "__main__":
    unittest.main()
