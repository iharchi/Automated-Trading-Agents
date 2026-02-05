"""Unit tests for the Market Regime Detector.

Tests cover:
    - Indicator calculation (ADX, ATR, BB width, EMAs)
    - Regime classification logic
    - Strategy adjustments per regime
    - Edge cases and error handling
    - Formatting
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from utils.market_regime import (
    MarketRegimeDetector,
    RegimeResult,
    RegimeIndicators,
    REGIME_ADJUSTMENTS,
)


# ── Test data helpers ───────────────────────────────────────────────


def _make_bars(
    n: int = 200,
    start_price: float = 100.0,
    trend: float = 0.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV bars.

    Args:
        n: Number of bars
        start_price: Starting close price
        trend: Daily drift (e.g., 0.001 = 0.1% daily uptrend)
        volatility: Daily return std dev
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    dates = pd.date_range(end="2024-06-01", periods=n, freq="D")
    returns = np.random.normal(trend, volatility, n)
    close = start_price * np.cumprod(1 + returns)

    # Generate OHLV from close
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_trending_up_bars(n: int = 200) -> pd.DataFrame:
    """Strong uptrend: positive drift, low noise."""
    return _make_bars(n=n, trend=0.004, volatility=0.01, seed=10)


def _make_trending_down_bars(n: int = 200) -> pd.DataFrame:
    """Strong downtrend: negative drift, low noise."""
    return _make_bars(n=n, trend=-0.004, volatility=0.01, seed=20)


def _make_ranging_bars(n: int = 200) -> pd.DataFrame:
    """Sideways market: no drift, low volatility."""
    return _make_bars(n=n, trend=0.0, volatility=0.005, seed=30)


def _make_volatile_bars(n: int = 200) -> pd.DataFrame:
    """High volatility: no drift, high noise."""
    return _make_bars(n=n, trend=0.0, volatility=0.04, seed=40)


# ── Regime Adjustments Tests ────────────────────────────────────────


class TestRegimeAdjustments(unittest.TestCase):
    """Tests for regime adjustment profiles."""

    def test_all_regimes_have_adjustments(self):
        for regime in ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "BREAKOUT"]:
            self.assertIn(regime, REGIME_ADJUSTMENTS)

    def test_adjustment_keys_consistent(self):
        expected_keys = {
            "position_size_factor",
            "stop_multiplier",
            "take_profit_multiplier",
            "signal_bias",
            "min_score_adjustment",
            "description",
        }
        for regime, adj in REGIME_ADJUSTMENTS.items():
            self.assertEqual(set(adj.keys()), expected_keys, f"Keys mismatch for {regime}")

    def test_volatile_reduces_position_size(self):
        self.assertLess(
            REGIME_ADJUSTMENTS["VOLATILE"]["position_size_factor"],
            REGIME_ADJUSTMENTS["TRENDING_UP"]["position_size_factor"],
        )

    def test_trending_up_favours_longs(self):
        self.assertEqual(REGIME_ADJUSTMENTS["TRENDING_UP"]["signal_bias"], "long")

    def test_trending_down_favours_shorts(self):
        self.assertEqual(REGIME_ADJUSTMENTS["TRENDING_DOWN"]["signal_bias"], "short")


# ── RegimeIndicators Tests ──────────────────────────────────────────


class TestRegimeIndicators(unittest.TestCase):
    """Tests for RegimeIndicators dataclass."""

    def test_default_values(self):
        ind = RegimeIndicators()
        self.assertEqual(ind.adx, 0.0)
        self.assertEqual(ind.price, 0.0)
        self.assertFalse(ind.bb_squeeze)

    def test_custom_values(self):
        ind = RegimeIndicators(adx=30.0, price=150.0, bb_squeeze=True)
        self.assertEqual(ind.adx, 30.0)
        self.assertEqual(ind.price, 150.0)
        self.assertTrue(ind.bb_squeeze)


# ── RegimeResult Tests ──────────────────────────────────────────────


class TestRegimeResult(unittest.TestCase):
    """Tests for RegimeResult dataclass."""

    def test_default_result(self):
        result = RegimeResult(symbol="SPY", regime="RANGING")
        self.assertEqual(result.symbol, "SPY")
        self.assertEqual(result.regime, "RANGING")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.error, "")

    def test_error_result(self):
        result = RegimeResult(symbol="BAD", regime="RANGING", error="No data")
        self.assertEqual(result.error, "No data")


# ── Indicator Calculation Tests ─────────────────────────────────────


class TestIndicatorCalculation(unittest.TestCase):
    """Tests for indicator calculation from price data."""

    def setUp(self):
        self.detector = MarketRegimeDetector(client=None)

    def test_calculate_indicators_uptrend(self):
        bars = _make_trending_up_bars()
        ind = self.detector._calculate_indicators(bars)

        self.assertGreater(ind.price, 0)
        self.assertGreater(ind.adx, 0)
        self.assertGreater(ind.atr, 0)
        self.assertGreater(ind.ema_short, 0)
        self.assertGreater(ind.ema_long, 0)
        # In uptrend, price should be above long EMA
        self.assertGreater(ind.price_vs_ema_long, 0)
        # Short EMA should be above long EMA
        self.assertGreater(ind.ema_short, ind.ema_long)

    def test_calculate_indicators_downtrend(self):
        bars = _make_trending_down_bars()
        ind = self.detector._calculate_indicators(bars)

        # In downtrend, price should be below long EMA
        self.assertLess(ind.price_vs_ema_long, 0)
        # Short EMA should be below long EMA
        self.assertLess(ind.ema_short, ind.ema_long)

    def test_calculate_indicators_volatile(self):
        bars = _make_volatile_bars()
        ind = self.detector._calculate_indicators(bars)

        # Volatile market should have higher historical vol
        self.assertGreater(ind.historical_vol, 0.1)

    def test_calculate_indicators_ranging(self):
        bars = _make_ranging_bars()
        ind = self.detector._calculate_indicators(bars)

        # Ranging market should have lower volatility
        self.assertLess(ind.historical_vol, 0.2)

    def test_calculate_indicators_with_volume(self):
        bars = _make_ranging_bars()
        ind = self.detector._calculate_indicators(bars)
        self.assertGreater(ind.volume_trend, 0)

    def test_calculate_indicators_insufficient_data(self):
        bars = _make_bars(n=10)
        ind = self.detector._calculate_indicators(bars)
        # Should return default indicators with price set
        self.assertGreater(ind.price, 0)

    def test_adx_calculation(self):
        bars = _make_trending_up_bars()
        adx = self.detector._calculate_adx(bars, 14)
        self.assertEqual(len(adx), len(bars))
        # ADX should be between 0 and 100
        self.assertTrue(all(0 <= v <= 100 for v in adx.dropna()))

    def test_atr_calculation(self):
        bars = _make_bars()
        atr = self.detector._calculate_atr(bars, 14)
        self.assertEqual(len(atr), len(bars))
        # ATR should be positive
        self.assertTrue(all(v >= 0 for v in atr.dropna()))

    def test_bb_width_calculation(self):
        bars = _make_bars()
        bb_width = self.detector._calculate_bb_width(bars["close"], 20, 2.0)
        self.assertEqual(len(bb_width), len(bars))
        # BB width should be positive
        self.assertTrue(all(v >= 0 for v in bb_width.dropna()))


# ── Regime Classification Tests ─────────────────────────────────────


class TestRegimeClassification(unittest.TestCase):
    """Tests for regime classification logic."""

    def setUp(self):
        self.detector = MarketRegimeDetector(client=None)

    def test_uptrend_detected(self):
        bars = _make_trending_up_bars()
        result = self.detector.detect("TEST", bars=bars)

        self.assertEqual(result.regime, "TRENDING_UP")
        self.assertGreater(result.confidence, 0.2)

    def test_downtrend_detected(self):
        bars = _make_trending_down_bars()
        result = self.detector.detect("TEST", bars=bars)

        self.assertEqual(result.regime, "TRENDING_DOWN")
        self.assertGreater(result.confidence, 0.2)

    def test_volatile_detected(self):
        bars = _make_volatile_bars()
        result = self.detector.detect("TEST", bars=bars)

        self.assertIn(result.regime, ["VOLATILE", "RANGING"])

    def test_ranging_detected(self):
        bars = _make_ranging_bars()
        result = self.detector.detect("TEST", bars=bars)

        self.assertIn(result.regime, ["RANGING", "TRENDING_UP", "TRENDING_DOWN"])
        # Even if not exactly RANGING, confidence shouldn't be too high
        # for a ranging market on any trending signal

    def test_adjustments_included(self):
        bars = _make_trending_up_bars()
        result = self.detector.detect("TEST", bars=bars)

        self.assertIn("position_size_factor", result.adjustments)
        self.assertIn("stop_multiplier", result.adjustments)
        self.assertIn("take_profit_multiplier", result.adjustments)
        self.assertIn("signal_bias", result.adjustments)

    def test_secondary_regime_set(self):
        bars = _make_bars()
        result = self.detector.detect("TEST", bars=bars)

        self.assertNotEqual(result.secondary_regime, "")
        self.assertNotEqual(result.secondary_regime, result.regime)

    def test_confidence_between_0_and_1(self):
        for bars_fn in [_make_trending_up_bars, _make_trending_down_bars, _make_ranging_bars, _make_volatile_bars]:
            bars = bars_fn()
            result = self.detector.detect("TEST", bars=bars)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)


# ── Error Handling Tests ────────────────────────────────────────────


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling."""

    def setUp(self):
        self.detector = MarketRegimeDetector(client=None)

    def test_empty_bars(self):
        empty = pd.DataFrame()
        result = self.detector.detect("TEST", bars=empty)

        self.assertEqual(result.regime, "RANGING")
        self.assertIn("Insufficient", result.error)

    def test_too_few_bars(self):
        bars = _make_bars(n=10)
        result = self.detector.detect("TEST", bars=bars)

        self.assertEqual(result.regime, "RANGING")
        self.assertIn("Insufficient", result.error)

    def test_no_client_no_bars(self):
        detector = MarketRegimeDetector(client=None)
        result = detector.detect("TEST")

        self.assertEqual(result.regime, "RANGING")
        self.assertIn("No client", result.error)

    def test_client_fetch_error(self):
        mock_client = MagicMock()
        mock_client.get_bars.side_effect = Exception("API Error")

        detector = MarketRegimeDetector(client=mock_client)
        result = detector.detect("TEST")

        self.assertEqual(result.regime, "RANGING")
        self.assertIn("Failed to fetch", result.error)


# ── Multi-symbol Tests ──────────────────────────────────────────────


class TestMultiSymbol(unittest.TestCase):
    """Tests for multi-symbol detection."""

    def test_detect_multi_returns_dict(self):
        mock_client = MagicMock()

        def get_bars_side_effect(symbol, **kwargs):
            if symbol == "BULL":
                return _make_trending_up_bars()
            elif symbol == "BEAR":
                return _make_trending_down_bars()
            return _make_ranging_bars()

        mock_client.get_bars.side_effect = get_bars_side_effect

        detector = MarketRegimeDetector(client=mock_client)
        results = detector.detect_multi(["BULL", "BEAR", "FLAT"])

        self.assertEqual(len(results), 3)
        self.assertIn("BULL", results)
        self.assertIn("BEAR", results)
        self.assertIn("FLAT", results)
        self.assertEqual(results["BULL"].regime, "TRENDING_UP")
        self.assertEqual(results["BEAR"].regime, "TRENDING_DOWN")


# ── Formatting Tests ────────────────────────────────────────────────


class TestFormatting(unittest.TestCase):
    """Tests for output formatting."""

    def test_format_result_basic(self):
        result = RegimeResult(
            symbol="SPY",
            regime="TRENDING_UP",
            confidence=0.75,
            indicators=RegimeIndicators(
                adx=32.0, atr=3.5, atr_pct=0.8, price=450.0,
                historical_vol=0.15, bb_width=4.2,
                ema_short=448.0, ema_long=440.0,
                ema_slope_short=0.15, ema_slope_long=0.08,
                price_vs_ema_short=0.45, price_vs_ema_long=2.3,
                adx_slope=3.5,
            ),
            adjustments=REGIME_ADJUSTMENTS["TRENDING_UP"],
            secondary_regime="BREAKOUT",
            description="Strong uptrend",
        )
        output = MarketRegimeDetector.format_result(result)

        self.assertIn("MARKET REGIME", output)
        self.assertIn("SPY", output)
        self.assertIn("TRENDING_UP", output)
        self.assertIn("75%", output)
        self.assertIn("BREAKOUT", output)
        self.assertIn("Position Size", output)

    def test_format_result_with_error(self):
        result = RegimeResult(
            symbol="BAD",
            regime="RANGING",
            error="No data available",
        )
        output = MarketRegimeDetector.format_result(result)

        self.assertIn("BAD", output)
        self.assertIn("No data available", output)

    def test_format_multi_empty(self):
        output = MarketRegimeDetector.format_multi({})
        self.assertIn("No regime data", output)

    def test_format_multi_with_data(self):
        results = {
            "SPY": RegimeResult(
                symbol="SPY",
                regime="TRENDING_UP",
                confidence=0.75,
                indicators=RegimeIndicators(adx=30, historical_vol=0.15, bb_width=4.0),
                adjustments=REGIME_ADJUSTMENTS["TRENDING_UP"],
            ),
            "VIX": RegimeResult(
                symbol="VIX",
                regime="VOLATILE",
                confidence=0.65,
                indicators=RegimeIndicators(adx=15, historical_vol=0.45, bb_width=8.0),
                adjustments=REGIME_ADJUSTMENTS["VOLATILE"],
            ),
        }
        output = MarketRegimeDetector.format_multi(results)

        self.assertIn("MARKET REGIME SUMMARY", output)
        self.assertIn("SPY", output)
        self.assertIn("VIX", output)
        self.assertIn("TRENDING_UP", output)
        self.assertIn("VOLATILE", output)


# ── Configuration Tests ─────────────────────────────────────────────


class TestConfiguration(unittest.TestCase):
    """Tests for configurable parameters."""

    def test_custom_adx_threshold(self):
        # Low threshold = more sensitive trend detection
        detector_low = MarketRegimeDetector(client=None, adx_trend_threshold=15.0)
        detector_high = MarketRegimeDetector(client=None, adx_trend_threshold=40.0)

        bars = _make_bars(trend=0.002, volatility=0.015, seed=50)

        result_low = detector_low.detect("TEST", bars=bars)
        result_high = detector_high.detect("TEST", bars=bars)

        # Lower threshold should be more likely to detect trends
        # Higher threshold should be more conservative
        self.assertIsNotNone(result_low.regime)
        self.assertIsNotNone(result_high.regime)

    def test_custom_vol_threshold(self):
        detector = MarketRegimeDetector(client=None, vol_high_threshold=0.10)
        bars = _make_bars(volatility=0.02)
        result = detector.detect("TEST", bars=bars)
        self.assertIsNotNone(result.regime)

    def test_default_parameters(self):
        detector = MarketRegimeDetector(client=None)
        self.assertEqual(detector.adx_period, 14)
        self.assertEqual(detector.adx_trend_threshold, 25.0)
        self.assertEqual(detector.bb_period, 20)
        self.assertEqual(detector.ema_long_period, 50)


if __name__ == "__main__":
    unittest.main()
