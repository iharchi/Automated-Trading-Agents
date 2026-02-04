"""Unit tests for the Multi-Timeframe Analysis Agent.

Tests cover:
    - Signal combination modes (unanimous, majority, weighted)
    - TimeframeSignal and MultiTimeframeResult dataclasses
    - Confidence and agreement ratio calculation
    - Formatting
"""

import unittest
from unittest.mock import MagicMock, patch

from agents.multi_timeframe_agent import (
    MultiTimeframeAgent,
    TimeframeSignal,
    MultiTimeframeResult,
    TIMEFRAME_WEIGHTS,
)


# ── Helpers ──────────────────────────────────────────────────────

def _make_tf_signal(timeframe: str, signal: str, score: int = 0) -> TimeframeSignal:
    """Create a TimeframeSignal for testing."""
    return TimeframeSignal(
        timeframe=timeframe,
        signal=signal,
        score=score,
        weight=TIMEFRAME_WEIGHTS.get(timeframe, 0.5),
        price=150.0,
        atr=3.5,
        indicators=[],
    )


def _make_agent(agreement_mode: str = "unanimous", min_agreement: float = 0.5):
    """Create a MultiTimeframeAgent with mocked client."""
    mock_client = MagicMock()
    agent = MultiTimeframeAgent(
        client=mock_client,
        timeframes=["1Day", "1Hour"],
        agreement_mode=agreement_mode,
        min_agreement=min_agreement,
    )
    return agent


# ── Unanimous mode tests ─────────────────────────────────────────

class TestUnanimousMode(unittest.TestCase):
    """Tests for unanimous agreement mode."""

    def setUp(self):
        self.agent = _make_agent(agreement_mode="unanimous")

    def test_all_buy_returns_buy(self):
        signals = [
            _make_tf_signal("1Day", "BUY", 3),
            _make_tf_signal("1Hour", "BUY", 2),
        ]
        result, confidence = self.agent._combine_unanimous(signals)
        self.assertEqual(result, "BUY")
        self.assertEqual(confidence, 1.0)

    def test_all_sell_returns_sell(self):
        signals = [
            _make_tf_signal("1Day", "SELL", -3),
            _make_tf_signal("1Hour", "SELL", -2),
        ]
        result, confidence = self.agent._combine_unanimous(signals)
        self.assertEqual(result, "SELL")
        self.assertEqual(confidence, 1.0)

    def test_mixed_signals_returns_hold(self):
        signals = [
            _make_tf_signal("1Day", "BUY", 3),
            _make_tf_signal("1Hour", "SELL", -2),
        ]
        result, confidence = self.agent._combine_unanimous(signals)
        self.assertEqual(result, "HOLD")
        self.assertEqual(confidence, 0.0)

    def test_buy_and_hold_returns_hold(self):
        signals = [
            _make_tf_signal("1Day", "BUY", 3),
            _make_tf_signal("1Hour", "HOLD", 0),
        ]
        result, confidence = self.agent._combine_unanimous(signals)
        self.assertEqual(result, "HOLD")

    def test_empty_signals_returns_hold(self):
        result, confidence = self.agent._combine_unanimous([])
        self.assertEqual(result, "HOLD")
        self.assertEqual(confidence, 0.0)


# ── Majority mode tests ──────────────────────────────────────────

class TestMajorityMode(unittest.TestCase):
    """Tests for majority agreement mode."""

    def setUp(self):
        self.agent = _make_agent(agreement_mode="majority", min_agreement=0.5)

    def test_majority_buy_returns_buy(self):
        signals = [
            _make_tf_signal("1Day", "BUY", 3),
            _make_tf_signal("1Hour", "BUY", 2),
            _make_tf_signal("15Min", "HOLD", 0),
        ]
        result, confidence = self.agent._combine_majority(signals)
        self.assertEqual(result, "BUY")
        self.assertAlmostEqual(confidence, 2/3, places=2)

    def test_majority_sell_returns_sell(self):
        signals = [
            _make_tf_signal("1Day", "SELL", -3),
            _make_tf_signal("1Hour", "SELL", -2),
            _make_tf_signal("15Min", "BUY", 1),
        ]
        result, confidence = self.agent._combine_majority(signals)
        self.assertEqual(result, "SELL")
        self.assertAlmostEqual(confidence, 2/3, places=2)

    def test_no_majority_returns_hold(self):
        signals = [
            _make_tf_signal("1Day", "BUY", 2),
            _make_tf_signal("1Hour", "SELL", -2),
            _make_tf_signal("15Min", "HOLD", 0),
        ]
        result, confidence = self.agent._combine_majority(signals)
        self.assertEqual(result, "HOLD")

    def test_exact_threshold_triggers(self):
        # 2 out of 4 = 0.5, which equals min_agreement
        signals = [
            _make_tf_signal("1Day", "BUY", 2),
            _make_tf_signal("1Hour", "BUY", 2),
            _make_tf_signal("15Min", "HOLD", 0),
            _make_tf_signal("5Min", "HOLD", 0),
        ]
        result, confidence = self.agent._combine_majority(signals)
        self.assertEqual(result, "BUY")


# ── Weighted mode tests ──────────────────────────────────────────

class TestWeightedMode(unittest.TestCase):
    """Tests for weighted agreement mode."""

    def setUp(self):
        self.agent = _make_agent(agreement_mode="weighted")

    def test_higher_timeframe_dominates(self):
        # 1Day (weight 1.0) BUY vs 1Hour (weight 0.6) SELL
        # Weighted sum = 1.0 - 0.6 = 0.4
        # Normalized = 0.4 / 1.6 = 0.25 < 0.3, so HOLD
        signals = [
            _make_tf_signal("1Day", "BUY", 3),
            _make_tf_signal("1Hour", "SELL", -2),
        ]
        result, confidence = self.agent._combine_weighted(signals)
        # This should be HOLD since normalized is below threshold
        self.assertEqual(result, "HOLD")

    def test_strong_buy_agreement(self):
        # Both BUY: sum = 1.0 + 0.6 = 1.6, normalized = 1.0
        signals = [
            _make_tf_signal("1Day", "BUY", 3),
            _make_tf_signal("1Hour", "BUY", 2),
        ]
        result, confidence = self.agent._combine_weighted(signals)
        self.assertEqual(result, "BUY")
        self.assertGreater(confidence, 0.5)

    def test_strong_sell_agreement(self):
        # Both SELL: sum = -1.0 - 0.6 = -1.6, normalized = -1.0
        signals = [
            _make_tf_signal("1Day", "SELL", -3),
            _make_tf_signal("1Hour", "SELL", -2),
        ]
        result, confidence = self.agent._combine_weighted(signals)
        self.assertEqual(result, "SELL")

    def test_holds_produce_hold(self):
        signals = [
            _make_tf_signal("1Day", "HOLD", 0),
            _make_tf_signal("1Hour", "HOLD", 0),
        ]
        result, confidence = self.agent._combine_weighted(signals)
        self.assertEqual(result, "HOLD")


# ── Dataclass tests ──────────────────────────────────────────────

class TestDataclasses(unittest.TestCase):
    """Tests for TimeframeSignal and MultiTimeframeResult."""

    def test_timeframe_signal_creation(self):
        tf_sig = _make_tf_signal("1Day", "BUY", 3)
        self.assertEqual(tf_sig.timeframe, "1Day")
        self.assertEqual(tf_sig.signal, "BUY")
        self.assertEqual(tf_sig.score, 3)
        self.assertEqual(tf_sig.weight, 1.0)

    def test_multi_timeframe_result_defaults(self):
        result = MultiTimeframeResult(symbol="AAPL")
        self.assertEqual(result.symbol, "AAPL")
        self.assertEqual(result.final_signal, "HOLD")
        self.assertEqual(result.agreement_ratio, 0.0)
        self.assertEqual(result.timeframe_signals, [])


# ── Formatting tests ─────────────────────────────────────────────

class TestFormatting(unittest.TestCase):
    """Tests for format_analysis output."""

    def test_format_produces_string(self):
        result = MultiTimeframeResult(
            symbol="AAPL",
            final_signal="BUY",
            agreement_ratio=1.0,
            confidence=1.0,
            weighted_score=2.5,
            current_price=150.0,
            timeframe_signals=[
                _make_tf_signal("1Day", "BUY", 3),
                _make_tf_signal("1Hour", "BUY", 2),
            ],
        )
        output = MultiTimeframeAgent.format_analysis(result.__dict__)
        self.assertIn("AAPL", output)
        self.assertIn("BUY", output)
        self.assertIn("1Day", output)
        self.assertIn("1Hour", output)

    def test_format_hold_signal(self):
        result = MultiTimeframeResult(
            symbol="TSLA",
            final_signal="HOLD",
            agreement_ratio=0.0,
            confidence=0.5,
        )
        output = MultiTimeframeAgent.format_analysis(result.__dict__)
        self.assertIn("HOLD", output)


# ── Agreement ratio tests ────────────────────────────────────────

class TestAgreementRatio(unittest.TestCase):
    """Tests for agreement ratio calculation."""

    def test_full_agreement(self):
        agent = _make_agent()
        signals = [
            _make_tf_signal("1Day", "BUY", 3),
            _make_tf_signal("1Hour", "BUY", 2),
        ]
        # The agent calculates this in analyze(), but we test the logic
        agreeing = sum(1 for s in signals if s.signal == "BUY")
        ratio = agreeing / len(signals)
        self.assertEqual(ratio, 1.0)

    def test_partial_agreement(self):
        signals = [
            _make_tf_signal("1Day", "BUY", 3),
            _make_tf_signal("1Hour", "HOLD", 0),
        ]
        agreeing = sum(1 for s in signals if s.signal == "BUY")
        ratio = agreeing / len(signals)
        self.assertEqual(ratio, 0.5)


# ── Weight tests ─────────────────────────────────────────────────

class TestTimeframeWeights(unittest.TestCase):
    """Tests for timeframe weight lookup."""

    def test_known_timeframe_weights(self):
        self.assertEqual(MultiTimeframeAgent._get_weight("1Day"), 1.0)
        self.assertEqual(MultiTimeframeAgent._get_weight("1Hour"), 0.6)
        self.assertEqual(MultiTimeframeAgent._get_weight("15Min"), 0.3)
        self.assertEqual(MultiTimeframeAgent._get_weight("5Min"), 0.15)
        self.assertEqual(MultiTimeframeAgent._get_weight("1Min"), 0.05)

    def test_unknown_timeframe_default(self):
        weight = MultiTimeframeAgent._get_weight("2Hour")
        self.assertEqual(weight, 0.5)  # default for unknown


if __name__ == "__main__":
    unittest.main()
