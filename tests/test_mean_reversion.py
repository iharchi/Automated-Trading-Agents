"""Unit tests for the Mean Reversion Agent.

Tests cover:
    - MeanReversionResult dataclass defaults
    - MeanReversionAgent.__init__ with default and custom parameters
    - analyze() with oversold, overbought, neutral, trending, insufficient data
    - execute() with HOLD, auto_trade off, BUY, SELL with/without position
    - _z_score static method
    - format_analysis() output
    - Edge cases: empty DataFrame, NaN values
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from agents.mean_reversion_agent import MeanReversionAgent, MeanReversionResult


# ── Synthetic data helpers ─────────────────────────────────────────


def _make_bars(
    n: int = 100,
    start_price: float = 100.0,
    trend: float = 0.0,
    volatility: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate generic OHLCV bars."""
    np.random.seed(seed)
    dates = pd.date_range(end="2024-06-01", periods=n, freq="D")
    returns = np.random.normal(trend, volatility, n)
    close = start_price * np.cumprod(1 + returns)

    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    opn = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(1_000, 10_000, n).astype(float)

    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_oversold_bars(n: int = 100) -> pd.DataFrame:
    """Create bars with noise then a sharp drop at the end -> oversold.

    We add noise to the flat section to keep ADX from becoming extreme,
    and we make the drop steep enough to breach lower BB, RSI, z-score
    and stochastic thresholds.
    """
    np.random.seed(42)
    # Noisy flat section keeps ADX moderate
    noise = np.random.normal(0, 0.3, n - 12)
    flat = 100 + noise
    # Sharp drop at end
    drop = np.linspace(flat[-1], 85, 12)
    close = np.concatenate([flat, drop])

    high = close + np.abs(np.random.normal(0.5, 0.2, n))
    low = close - np.abs(np.random.normal(0.5, 0.2, n))
    opn = close + np.random.normal(0, 0.1, n)
    volume = np.full(n, 5000.0)

    dates = pd.date_range(end="2024-06-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_overbought_bars(n: int = 100) -> pd.DataFrame:
    """Create bars with noise then a sharp surge at the end -> overbought."""
    np.random.seed(42)
    noise = np.random.normal(0, 0.3, n - 12)
    flat = 100 + noise
    surge = np.linspace(flat[-1], 115, 12)
    close = np.concatenate([flat, surge])

    high = close + np.abs(np.random.normal(0.5, 0.2, n))
    low = close - np.abs(np.random.normal(0.5, 0.2, n))
    opn = close - np.random.normal(0, 0.1, n)
    volume = np.full(n, 5000.0)

    dates = pd.date_range(end="2024-06-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_neutral_bars(n: int = 100) -> pd.DataFrame:
    """Oscillating price around the mean -- no extremes in any indicator.

    We use a sine wave with small noise so price stays near the SMA, RSI
    stays mid-range, z-score stays near zero, and stochastic stays
    between 20 and 80.
    """
    np.random.seed(99)
    t = np.arange(n, dtype=float)
    # Gentle sine oscillation: amplitude 1 around 100
    close = 100.0 + 1.0 * np.sin(2 * np.pi * t / 25) + np.random.normal(0, 0.15, n)
    high = close + np.abs(np.random.normal(0.3, 0.1, n))
    low = close - np.abs(np.random.normal(0.3, 0.1, n))
    opn = close + np.random.normal(0, 0.05, n)
    volume = np.full(n, 5000.0)

    dates = pd.date_range(end="2024-06-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_trending_bars(n: int = 100) -> pd.DataFrame:
    """Strong directional trend -> high ADX."""
    np.random.seed(10)
    close = np.linspace(100, 150, n) + np.random.normal(0, 0.3, n)
    high = close + np.abs(np.random.normal(0.5, 0.3, n))
    low = close - np.abs(np.random.normal(0.5, 0.3, n))
    opn = close + np.random.normal(0, 0.2, n)
    volume = np.full(n, 5000.0)

    dates = pd.date_range(end="2024-06-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _agent(bars_df: pd.DataFrame | None = None, **kwargs) -> MeanReversionAgent:
    """Build agent with a mocked client returning *bars_df*."""
    mock_client = MagicMock()
    if bars_df is not None:
        mock_client.get_bars.return_value = bars_df
    else:
        mock_client.get_bars.return_value = pd.DataFrame()
    return MeanReversionAgent(client=mock_client, **kwargs)


# ── 1. MeanReversionResult dataclass ──────────────────────────────


class TestMeanReversionResult(unittest.TestCase):
    """Tests for the MeanReversionResult dataclass defaults."""

    def test_default_symbol(self):
        r = MeanReversionResult()
        self.assertEqual(r.symbol, "")

    def test_default_signal_is_hold(self):
        r = MeanReversionResult()
        self.assertEqual(r.signal, "HOLD")

    def test_default_score_zero(self):
        r = MeanReversionResult()
        self.assertEqual(r.score, 0.0)

    def test_default_numeric_fields_zero(self):
        r = MeanReversionResult()
        for attr in (
            "current_price", "sma", "upper_band", "lower_band",
            "bb_width", "z_score", "rsi", "stochastic_k", "atr", "adx",
            "stop_loss", "take_profit",
        ):
            self.assertEqual(getattr(r, attr), 0.0, f"{attr} should default to 0.0")

    def test_default_string_fields_empty(self):
        r = MeanReversionResult()
        self.assertEqual(r.regime, "")
        self.assertEqual(r.detail, "")

    def test_custom_values(self):
        r = MeanReversionResult(
            symbol="AAPL", signal="BUY", score=0.75,
            current_price=150.0, rsi=25.0, z_score=-2.1,
        )
        self.assertEqual(r.symbol, "AAPL")
        self.assertEqual(r.signal, "BUY")
        self.assertAlmostEqual(r.score, 0.75)
        self.assertAlmostEqual(r.rsi, 25.0)

    def test_dict_conversion(self):
        r = MeanReversionResult(symbol="SPY", signal="SELL")
        d = r.__dict__
        self.assertIsInstance(d, dict)
        self.assertEqual(d["symbol"], "SPY")
        self.assertEqual(d["signal"], "SELL")


# ── 2. __init__ tests ─────────────────────────────────────────────


class TestMeanReversionAgentInit(unittest.TestCase):
    """Tests for MeanReversionAgent constructor."""

    def test_default_parameters(self):
        agent = _agent(_make_bars())
        self.assertEqual(agent.bb_period, 20)
        self.assertEqual(agent.bb_std, 2.0)
        self.assertEqual(agent.rsi_period, 14)
        self.assertEqual(agent.rsi_oversold, 30)
        self.assertEqual(agent.rsi_overbought, 70)
        self.assertEqual(agent.z_entry_threshold, 1.5)
        self.assertEqual(agent.z_exit_threshold, 0.0)
        self.assertEqual(agent.stoch_period, 14)
        self.assertEqual(agent.stoch_smooth, 3)
        self.assertEqual(agent.stoch_oversold, 20)
        self.assertEqual(agent.stoch_overbought, 80)
        self.assertEqual(agent.adx_period, 14)
        self.assertAlmostEqual(agent.adx_max_threshold, 25.0)
        self.assertEqual(agent.atr_period, 14)
        self.assertAlmostEqual(agent.atr_stop_mult, 1.5)
        self.assertAlmostEqual(agent.take_profit_ratio, 2.0)
        self.assertFalse(agent.auto_trade)
        self.assertEqual(agent.default_qty, 1)
        self.assertEqual(agent.min_confirmations, 2)

    def test_custom_parameters(self):
        agent = _agent(
            _make_bars(),
            bb_period=30,
            bb_std=2.5,
            rsi_period=10,
            rsi_oversold=25,
            rsi_overbought=75,
            z_entry_threshold=2.0,
            stoch_oversold=15,
            stoch_overbought=85,
            adx_max_threshold=30.0,
            atr_stop_mult=2.0,
            take_profit_ratio=3.0,
            auto_trade=True,
            default_qty=5,
            min_confirmations=3,
        )
        self.assertEqual(agent.bb_period, 30)
        self.assertEqual(agent.bb_std, 2.5)
        self.assertEqual(agent.rsi_period, 10)
        self.assertEqual(agent.rsi_oversold, 25)
        self.assertEqual(agent.rsi_overbought, 75)
        self.assertAlmostEqual(agent.z_entry_threshold, 2.0)
        self.assertEqual(agent.stoch_oversold, 15)
        self.assertEqual(agent.stoch_overbought, 85)
        self.assertAlmostEqual(agent.adx_max_threshold, 30.0)
        self.assertAlmostEqual(agent.atr_stop_mult, 2.0)
        self.assertAlmostEqual(agent.take_profit_ratio, 3.0)
        self.assertTrue(agent.auto_trade)
        self.assertEqual(agent.default_qty, 5)
        self.assertEqual(agent.min_confirmations, 3)

    def test_name_set_by_base(self):
        agent = _agent(_make_bars())
        self.assertEqual(agent.name, "MeanReversion")

    def test_client_is_mock(self):
        mock_client = MagicMock()
        agent = MeanReversionAgent(client=mock_client)
        self.assertIs(agent.client, mock_client)


# ── 3. analyze() tests ────────────────────────────────────────────


class TestAnalyzeInsufficientData(unittest.TestCase):
    """Insufficient bar count returns default HOLD."""

    def test_empty_dataframe(self):
        agent = _agent(pd.DataFrame())
        result = agent.analyze("SPY")
        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["symbol"], "SPY")

    def test_few_bars(self):
        # Only 10 bars, below bb_period(20) + 5
        agent = _agent(_make_bars(n=10))
        result = agent.analyze("AAPL")
        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["symbol"], "AAPL")

    def test_barely_insufficient(self):
        # Exactly bb_period + 4 = 24 bars (needs > bb_period + 5 = 25)
        agent = _agent(_make_bars(n=24))
        result = agent.analyze("QQQ")
        self.assertEqual(result["signal"], "HOLD")


class TestAnalyzeOversold(unittest.TestCase):
    """Sharp drop should produce BUY signal (oversold).

    We set adx_max_threshold=99 to disable the trending filter because
    the sharp price move inherently raises ADX.  The trending filter
    is verified in its own test class.
    """

    def _oversold_agent(self, **kw):
        kw.setdefault("adx_max_threshold", 99.0)
        return _agent(_make_oversold_bars(), **kw)

    def test_oversold_returns_buy(self):
        result = self._oversold_agent().analyze("SPY")
        self.assertEqual(result["signal"], "BUY")

    def test_oversold_positive_score(self):
        result = self._oversold_agent().analyze("SPY")
        self.assertGreater(result["score"], 0.0)

    def test_oversold_score_capped_at_1(self):
        result = self._oversold_agent().analyze("SPY")
        self.assertLessEqual(result["score"], 1.0)

    def test_oversold_stop_loss_below_price(self):
        result = self._oversold_agent().analyze("SPY")
        self.assertGreater(result["current_price"], result["stop_loss"])

    def test_oversold_take_profit_above_price(self):
        result = self._oversold_agent().analyze("SPY")
        self.assertGreater(result["take_profit"], result["current_price"])

    def test_oversold_regime_is_ranging(self):
        result = self._oversold_agent().analyze("SPY")
        self.assertEqual(result["regime"], "ranging")


class TestAnalyzeOverbought(unittest.TestCase):
    """Sharp surge should produce SELL signal (overbought).

    adx_max_threshold=99 disables trending filter (tested separately).
    """

    def _overbought_agent(self, **kw):
        kw.setdefault("adx_max_threshold", 99.0)
        return _agent(_make_overbought_bars(), **kw)

    def test_overbought_returns_sell(self):
        result = self._overbought_agent().analyze("SPY")
        self.assertEqual(result["signal"], "SELL")

    def test_overbought_negative_score(self):
        result = self._overbought_agent().analyze("SPY")
        self.assertLess(result["score"], 0.0)

    def test_overbought_score_lower_bound(self):
        result = self._overbought_agent().analyze("SPY")
        self.assertGreaterEqual(result["score"], -1.0)

    def test_overbought_stop_loss_above_price(self):
        result = self._overbought_agent().analyze("SPY")
        self.assertGreater(result["stop_loss"], result["current_price"])

    def test_overbought_take_profit_below_price(self):
        result = self._overbought_agent().analyze("SPY")
        self.assertLess(result["take_profit"], result["current_price"])


class TestAnalyzeNeutral(unittest.TestCase):
    """Sideways / neutral bars should produce HOLD."""

    def test_neutral_hold(self):
        agent = _agent(_make_neutral_bars())
        result = agent.analyze("SPY")
        self.assertEqual(result["signal"], "HOLD")

    def test_neutral_zero_score(self):
        agent = _agent(_make_neutral_bars())
        result = agent.analyze("SPY")
        self.assertEqual(result["score"], 0.0)

    def test_neutral_no_stop_loss(self):
        agent = _agent(_make_neutral_bars())
        result = agent.analyze("SPY")
        self.assertEqual(result["stop_loss"], 0.0)

    def test_neutral_no_take_profit(self):
        agent = _agent(_make_neutral_bars())
        result = agent.analyze("SPY")
        self.assertEqual(result["take_profit"], 0.0)


class TestAnalyzeTrending(unittest.TestCase):
    """High ADX (trending) should suppress mean-reversion signals."""

    def test_trending_produces_hold(self):
        # Even with oversold data, if ADX is high the agent should HOLD.
        # Use a very low adx_max_threshold so the trending bars trigger it.
        agent = _agent(_make_trending_bars(), adx_max_threshold=10.0)
        result = agent.analyze("SPY")
        self.assertEqual(result["signal"], "HOLD")

    def test_trending_regime_label(self):
        agent = _agent(_make_trending_bars(), adx_max_threshold=10.0)
        result = agent.analyze("SPY")
        self.assertEqual(result["regime"], "trending")

    def test_trending_detail_mentions_adx(self):
        agent = _agent(_make_trending_bars(), adx_max_threshold=10.0)
        result = agent.analyze("SPY")
        self.assertIn("ADX", result["detail"])
        self.assertIn("TRENDING", result["detail"])


class TestAnalyzeScoreBounds(unittest.TestCase):
    """Score must always be in [-1.0, 1.0]."""

    def test_oversold_score_in_range(self):
        agent = _agent(_make_oversold_bars(), adx_max_threshold=99.0)
        result = agent.analyze("SPY")
        self.assertGreaterEqual(result["score"], -1.0)
        self.assertLessEqual(result["score"], 1.0)

    def test_overbought_score_in_range(self):
        agent = _agent(_make_overbought_bars(), adx_max_threshold=99.0)
        result = agent.analyze("SPY")
        self.assertGreaterEqual(result["score"], -1.0)
        self.assertLessEqual(result["score"], 1.0)

    def test_neutral_score_in_range(self):
        agent = _agent(_make_neutral_bars())
        result = agent.analyze("SPY")
        self.assertGreaterEqual(result["score"], -1.0)
        self.assertLessEqual(result["score"], 1.0)

    def test_generic_bars_score_in_range(self):
        agent = _agent(_make_bars())
        result = agent.analyze("SPY")
        self.assertGreaterEqual(result["score"], -1.0)
        self.assertLessEqual(result["score"], 1.0)


class TestAnalyzeIndicatorValues(unittest.TestCase):
    """Verify that returned indicator fields are sensible."""

    def test_rsi_in_0_100(self):
        agent = _agent(_make_bars())
        result = agent.analyze("SPY")
        self.assertGreaterEqual(result["rsi"], 0.0)
        self.assertLessEqual(result["rsi"], 100.0)

    def test_stochastic_k_in_0_100(self):
        agent = _agent(_make_bars())
        result = agent.analyze("SPY")
        self.assertGreaterEqual(result["stochastic_k"], 0.0)
        self.assertLessEqual(result["stochastic_k"], 100.0)

    def test_atr_positive(self):
        agent = _agent(_make_bars())
        result = agent.analyze("SPY")
        self.assertGreater(result["atr"], 0.0)

    def test_adx_non_negative(self):
        agent = _agent(_make_bars())
        result = agent.analyze("SPY")
        self.assertGreaterEqual(result["adx"], 0.0)

    def test_sma_positive(self):
        agent = _agent(_make_bars())
        result = agent.analyze("SPY")
        self.assertGreater(result["sma"], 0.0)

    def test_upper_band_above_lower(self):
        agent = _agent(_make_bars())
        result = agent.analyze("SPY")
        self.assertGreater(result["upper_band"], result["lower_band"])

    def test_bb_width_positive(self):
        agent = _agent(_make_bars())
        result = agent.analyze("SPY")
        self.assertGreater(result["bb_width"], 0.0)

    def test_result_is_dict(self):
        agent = _agent(_make_bars())
        result = agent.analyze("SPY")
        self.assertIsInstance(result, dict)


class TestAnalyzeReturnsAllKeys(unittest.TestCase):
    """analyze() dict must contain every MeanReversionResult field."""

    def test_keys_present(self):
        agent = _agent(_make_bars())
        result = agent.analyze("SPY")
        expected = {
            "symbol", "signal", "score", "current_price", "sma",
            "upper_band", "lower_band", "bb_width", "z_score", "rsi",
            "stochastic_k", "atr", "adx", "stop_loss", "take_profit",
            "regime", "detail",
        }
        self.assertTrue(expected.issubset(result.keys()))


class TestAnalyzeMinConfirmations(unittest.TestCase):
    """Varying min_confirmations affects whether a signal fires.

    adx_max_threshold=99 disables trending filter so we isolate the
    confirmation-count logic.
    """

    def test_high_min_confirmations_forces_hold(self):
        # With min_confirmations=5, out of 4 possible indicators,
        # signal must remain HOLD.
        agent = _agent(
            _make_oversold_bars(),
            min_confirmations=5,
            adx_max_threshold=99.0,
        )
        result = agent.analyze("SPY")
        self.assertEqual(result["signal"], "HOLD")

    def test_low_min_confirmations_easier_signal(self):
        agent = _agent(
            _make_oversold_bars(),
            min_confirmations=1,
            adx_max_threshold=99.0,
        )
        result = agent.analyze("SPY")
        # At least one bullish indicator fires on oversold data
        self.assertEqual(result["signal"], "BUY")


# ── 4. execute() tests ────────────────────────────────────────────


class TestExecuteHold(unittest.TestCase):
    """HOLD signal should never place an order."""

    def test_hold_no_order(self):
        agent = _agent(_make_bars(), auto_trade=True)
        analysis = {"signal": "HOLD", "score": 0.0, "current_price": 100,
                     "stop_loss": 0, "take_profit": 0, "regime": "ranging", "atr": 1.0}
        result = agent.execute("SPY", analysis)
        self.assertIsNone(result["order"])
        agent.client.submit_order.assert_not_called()

    def test_hold_returns_expected_keys(self):
        agent = _agent(_make_bars())
        analysis = {"signal": "HOLD", "score": 0.0, "current_price": 100,
                     "stop_loss": 0, "take_profit": 0, "regime": "ranging", "atr": 1.0}
        result = agent.execute("SPY", analysis)
        self.assertIn("symbol", result)
        self.assertIn("signal", result)
        self.assertIn("order", result)


class TestExecuteAutoTradeOff(unittest.TestCase):
    """auto_trade=False should skip order placement even on BUY/SELL."""

    def test_buy_no_auto_trade(self):
        agent = _agent(_make_bars(), auto_trade=False)
        analysis = {"signal": "BUY", "score": 0.5, "current_price": 100,
                     "stop_loss": 95, "take_profit": 110, "regime": "ranging", "atr": 1.5}
        result = agent.execute("SPY", analysis)
        self.assertIsNone(result["order"])
        agent.client.submit_order.assert_not_called()

    def test_sell_no_auto_trade(self):
        agent = _agent(_make_bars(), auto_trade=False)
        analysis = {"signal": "SELL", "score": -0.5, "current_price": 100,
                     "stop_loss": 105, "take_profit": 90, "regime": "ranging", "atr": 1.5}
        result = agent.execute("SPY", analysis)
        self.assertIsNone(result["order"])
        agent.client.submit_order.assert_not_called()


class TestExecuteBuy(unittest.TestCase):
    """BUY + auto_trade -> submit_order called."""

    def test_buy_order_placed(self):
        agent = _agent(_make_bars(), auto_trade=True, default_qty=3)
        agent.client.submit_order.return_value = {"id": "order123"}
        analysis = {"signal": "BUY", "score": 0.5, "current_price": 100,
                     "stop_loss": 95, "take_profit": 110, "regime": "ranging", "atr": 1.5}
        result = agent.execute("SPY", analysis)
        agent.client.submit_order.assert_called_once_with(symbol="SPY", qty=3, side="buy")
        self.assertEqual(result["order"], {"id": "order123"})

    def test_buy_custom_qty(self):
        agent = _agent(_make_bars(), auto_trade=True, default_qty=1)
        agent.client.submit_order.return_value = {"id": "order456"}
        analysis = {"signal": "BUY", "score": 0.5, "current_price": 100,
                     "stop_loss": 95, "take_profit": 110, "regime": "ranging", "atr": 1.5}
        result = agent.execute("SPY", analysis, qty=10)
        agent.client.submit_order.assert_called_once_with(symbol="SPY", qty=10, side="buy")

    def test_buy_result_fields(self):
        agent = _agent(_make_bars(), auto_trade=True)
        agent.client.submit_order.return_value = {"id": "o1"}
        analysis = {"signal": "BUY", "score": 0.6, "current_price": 50,
                     "stop_loss": 48, "take_profit": 55, "regime": "ranging", "atr": 0.8}
        result = agent.execute("TEST", analysis)
        self.assertEqual(result["symbol"], "TEST")
        self.assertEqual(result["signal"], "BUY")
        self.assertAlmostEqual(result["score"], 0.6)
        self.assertEqual(result["current_price"], 50)
        self.assertEqual(result["stop_loss"], 48)
        self.assertEqual(result["take_profit"], 55)
        self.assertEqual(result["regime"], "ranging")


class TestExecuteSellNoPosition(unittest.TestCase):
    """SELL + auto_trade but no existing position -> skip."""

    def test_sell_no_position_skipped(self):
        agent = _agent(_make_bars(), auto_trade=True)
        agent.client.get_positions.return_value = []
        analysis = {"signal": "SELL", "score": -0.5, "current_price": 100,
                     "stop_loss": 105, "take_profit": 90, "regime": "ranging", "atr": 1.5}
        result = agent.execute("SPY", analysis)
        self.assertEqual(result["order"], "skipped_no_position")
        agent.client.submit_order.assert_not_called()

    def test_sell_other_position_not_matching(self):
        agent = _agent(_make_bars(), auto_trade=True)
        agent.client.get_positions.return_value = [{"symbol": "AAPL"}]
        analysis = {"signal": "SELL", "score": -0.5, "current_price": 100,
                     "stop_loss": 105, "take_profit": 90, "regime": "ranging", "atr": 1.5}
        result = agent.execute("SPY", analysis)
        self.assertEqual(result["order"], "skipped_no_position")


class TestExecuteSellWithPosition(unittest.TestCase):
    """SELL + auto_trade + existing position -> order placed."""

    def test_sell_order_placed(self):
        agent = _agent(_make_bars(), auto_trade=True, default_qty=2)
        agent.client.get_positions.return_value = [{"symbol": "SPY"}]
        agent.client.submit_order.return_value = {"id": "sell_order_1"}
        analysis = {"signal": "SELL", "score": -0.75, "current_price": 100,
                     "stop_loss": 105, "take_profit": 90, "regime": "ranging", "atr": 1.5}
        result = agent.execute("SPY", analysis)
        agent.client.submit_order.assert_called_once_with(symbol="SPY", qty=2, side="sell")
        self.assertEqual(result["order"], {"id": "sell_order_1"})

    def test_sell_custom_qty(self):
        agent = _agent(_make_bars(), auto_trade=True, default_qty=1)
        agent.client.get_positions.return_value = [{"symbol": "SPY"}]
        agent.client.submit_order.return_value = {"id": "sell_order_2"}
        analysis = {"signal": "SELL", "score": -0.5, "current_price": 100,
                     "stop_loss": 105, "take_profit": 90, "regime": "ranging", "atr": 1.5}
        result = agent.execute("SPY", analysis, qty=7)
        agent.client.submit_order.assert_called_once_with(symbol="SPY", qty=7, side="sell")


# ── 5. _z_score static method ─────────────────────────────────────


class TestZScore(unittest.TestCase):
    """Tests for the static _z_score helper."""

    def test_z_score_returns_series(self):
        close = pd.Series(np.linspace(100, 110, 50))
        z = MeanReversionAgent._z_score(close, period=20)
        self.assertIsInstance(z, pd.Series)
        self.assertEqual(len(z), 50)

    def test_z_score_initial_nan(self):
        close = pd.Series(np.linspace(100, 110, 50))
        z = MeanReversionAgent._z_score(close, period=20)
        # The first (period - 1) values should be NaN
        self.assertTrue(z.iloc[:19].isna().all())

    def test_z_score_constant_series(self):
        close = pd.Series(np.full(50, 100.0))
        z = MeanReversionAgent._z_score(close, period=20)
        # Standard deviation is 0 -> replaced with NaN -> z-score is NaN
        valid = z.dropna()
        # All non-NaN should actually be NaN because std=0
        # The method replaces 0 std with NaN, so (close - sma) / NaN = NaN
        self.assertTrue(z.iloc[19:].isna().all())

    def test_z_score_positive_for_above_mean(self):
        # Linearly increasing: last values well above the rolling mean
        close = pd.Series(np.linspace(50, 150, 50))
        z = MeanReversionAgent._z_score(close, period=20)
        self.assertGreater(z.iloc[-1], 0)

    def test_z_score_negative_for_below_mean(self):
        # Linearly decreasing: last values well below the rolling mean
        close = pd.Series(np.linspace(150, 50, 50))
        z = MeanReversionAgent._z_score(close, period=20)
        self.assertLess(z.iloc[-1], 0)

    def test_z_score_mean_reverting_near_zero(self):
        # Random walk should have z near 0 over time
        np.random.seed(7)
        close = pd.Series(100 + np.cumsum(np.random.normal(0, 0.1, 200)))
        z = MeanReversionAgent._z_score(close, period=20)
        # The average absolute z should be moderate
        mean_abs_z = z.dropna().abs().mean()
        self.assertLess(mean_abs_z, 3.0)


# ── 6. format_analysis() ──────────────────────────────────────────


class TestFormatAnalysis(unittest.TestCase):
    """Tests for the static format_analysis method."""

    def _sample_analysis(self, signal="BUY"):
        return {
            "symbol": "AAPL",
            "signal": signal,
            "score": 0.5 if signal == "BUY" else (-0.5 if signal == "SELL" else 0.0),
            "current_price": 150.0,
            "sma": 155.0,
            "upper_band": 165.0,
            "lower_band": 145.0,
            "bb_width": 0.1290,
            "z_score": -1.8,
            "rsi": 28.5,
            "stochastic_k": 15.0,
            "atr": 3.2,
            "adx": 18.5,
            "stop_loss": 145.2,
            "take_profit": 155.0,
            "regime": "ranging",
            "detail": "Price $150.00 <= lower BB $145.00; RSI 28.5 <= 30 (oversold)",
        }

    def test_format_contains_symbol(self):
        out = MeanReversionAgent.format_analysis(self._sample_analysis())
        self.assertIn("AAPL", out)

    def test_format_contains_price(self):
        out = MeanReversionAgent.format_analysis(self._sample_analysis())
        self.assertIn("150.00", out)

    def test_format_contains_regime(self):
        out = MeanReversionAgent.format_analysis(self._sample_analysis())
        self.assertIn("ranging", out)

    def test_format_contains_indicator_labels(self):
        out = MeanReversionAgent.format_analysis(self._sample_analysis())
        for label in ("SMA", "Upper BB", "Lower BB", "BB Width",
                       "Z-Score", "RSI", "Stoch %K", "ADX", "ATR"):
            self.assertIn(label, out, f"Missing label: {label}")

    def test_format_contains_signal(self):
        out = MeanReversionAgent.format_analysis(self._sample_analysis("BUY"))
        self.assertIn("BUY", out)

    def test_format_buy_shows_stop_take(self):
        out = MeanReversionAgent.format_analysis(self._sample_analysis("BUY"))
        self.assertIn("Stop Loss", out)
        self.assertIn("Take Profit", out)

    def test_format_hold_omits_stop_take(self):
        out = MeanReversionAgent.format_analysis(self._sample_analysis("HOLD"))
        self.assertNotIn("Stop Loss", out)
        self.assertNotIn("Take Profit", out)

    def test_format_contains_detail(self):
        out = MeanReversionAgent.format_analysis(self._sample_analysis())
        self.assertIn("Reasons", out)
        self.assertIn("oversold", out)

    def test_format_returns_string(self):
        out = MeanReversionAgent.format_analysis(self._sample_analysis())
        self.assertIsInstance(out, str)

    def test_format_sell_signal(self):
        out = MeanReversionAgent.format_analysis(self._sample_analysis("SELL"))
        self.assertIn("SELL", out)
        self.assertIn("Stop Loss", out)


# ── 7. Edge cases ─────────────────────────────────────────────────


class TestEdgeCases(unittest.TestCase):
    """Edge cases: empty data, NaN-contaminated data, etc."""

    def test_empty_dataframe_returns_hold(self):
        agent = _agent(pd.DataFrame())
        result = agent.analyze("XYZ")
        self.assertEqual(result["signal"], "HOLD")

    def test_dataframe_with_nan_close(self):
        """If close column has some NaN, the ta library and z_score should cope."""
        bars = _make_bars(n=100)
        bars.loc[bars.index[50], "close"] = np.nan
        agent = _agent(bars)
        # Should not raise
        result = agent.analyze("SPY")
        self.assertIn("signal", result)

    def test_single_bar(self):
        bars = _make_bars(n=1)
        agent = _agent(bars)
        result = agent.analyze("SPY")
        self.assertEqual(result["signal"], "HOLD")

    def test_analyze_passes_timeframe_and_limit(self):
        agent = _agent(_make_bars())
        agent.analyze("SPY", timeframe="1Hour", limit=200)
        agent.client.get_bars.assert_called_once_with("SPY", timeframe="1Hour", limit=200)

    def test_analyze_default_timeframe_and_limit(self):
        agent = _agent(_make_bars())
        agent.analyze("SPY")
        agent.client.get_bars.assert_called_once_with("SPY", timeframe="1Day", limit=100)


# ── 8. run() lifecycle (inherited from BaseAgent) ─────────────────


class TestRunLifecycle(unittest.TestCase):
    """BaseAgent.run() calls analyze then execute."""

    def test_run_calls_analyze_and_execute(self):
        agent = _agent(_make_oversold_bars(), auto_trade=False)
        result = agent.run("SPY")
        # Should return execute result dict
        self.assertIn("signal", result)
        self.assertIn("order", result)
        # get_bars was called by analyze
        agent.client.get_bars.assert_called_once()

    def test_run_hold_signal_no_order(self):
        # Neutral bars produce HOLD; auto_trade is on but HOLD means no order
        agent = _agent(_make_neutral_bars(), auto_trade=True)
        result = agent.run("SPY")
        self.assertEqual(result["signal"], "HOLD")
        self.assertIsNone(result["order"])


if __name__ == "__main__":
    unittest.main()
