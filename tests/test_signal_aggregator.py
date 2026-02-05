"""Tests for utils/signal_aggregator.py"""

import pytest

from utils.signal_aggregator import (
    DEFAULT_WEIGHTS,
    REGIME_WEIGHT_ADJUSTMENTS,
    AggregatedSignal,
    SignalAggregator,
    SourceSignal,
)


# ── Helper factories ──────────────────────────────────────────────

def _ta(score: int = 3, signal: str = "BUY", price: float = 150.0, atr: float = 2.5):
    return {"composite_score": score, "signal": signal, "current_price": price, "atr": atr}


def _sent(score: float = 0.2, signal: str = "BUY"):
    return {"composite_score": score, "signal": signal}


def _mtf(score: float = 2.0, signal: str = "BUY", confidence: float = 0.6):
    return {"weighted_score": score, "final_signal": signal, "confidence": confidence}


def _regime(regime: str = "TRENDING_UP", confidence: float = 0.7, adjustments: dict | None = None):
    from utils.market_regime import REGIME_ADJUSTMENTS
    adj = adjustments if adjustments is not None else REGIME_ADJUSTMENTS.get(regime, {})
    return {"regime": regime, "confidence": confidence, "adjustments": adj}


def _corr(is_correlated: bool = False, correlated_with: list | None = None):
    return {"is_correlated": is_correlated, "correlated_with": correlated_with or []}


# ── Dataclass tests ───────────────────────────────────────────────

class TestSourceSignal:
    def test_creation(self):
        s = SourceSignal("ta", 3.0, 0.5, "BUY", 0.5, 0.4, 0.2)
        assert s.name == "ta"
        assert s.signal == "BUY"
        assert s.weight == 0.4

    def test_weighted_contribution(self):
        s = SourceSignal("ta", 3.0, 0.5, "BUY", 0.5, 0.4, 0.2)
        assert s.weighted_contribution == 0.2


class TestAggregatedSignal:
    def test_defaults(self):
        a = AggregatedSignal(symbol="AAPL")
        assert a.signal == "HOLD"
        assert a.confidence == 0.0
        assert a.source_count == 0
        assert a.sources == []

    def test_error(self):
        a = AggregatedSignal(symbol="AAPL", error="test error")
        assert a.error == "test error"


# ── Normalisation tests ──────────────────────────────────────────

class TestNormalisation:
    def test_ta_positive(self):
        assert SignalAggregator.normalise_ta(3) == pytest.approx(0.5)

    def test_ta_negative(self):
        assert SignalAggregator.normalise_ta(-4) == pytest.approx(-4 / 6)

    def test_ta_clamped_high(self):
        assert SignalAggregator.normalise_ta(10) == 1.0

    def test_ta_clamped_low(self):
        assert SignalAggregator.normalise_ta(-10) == -1.0

    def test_ta_zero_max(self):
        assert SignalAggregator.normalise_ta(3, max_score=0) == 0.0

    def test_sentiment_passthrough(self):
        assert SignalAggregator.normalise_sentiment(0.5) == 0.5

    def test_sentiment_clamped(self):
        assert SignalAggregator.normalise_sentiment(1.5) == 1.0
        assert SignalAggregator.normalise_sentiment(-1.5) == -1.0

    def test_mtf_normalise(self):
        assert SignalAggregator.normalise_mtf(3.0) == pytest.approx(0.5)

    def test_mtf_zero_max(self):
        assert SignalAggregator.normalise_mtf(3.0, max_score=0) == 0.0

    def test_regime_to_score_trending_up(self):
        assert SignalAggregator.regime_to_score("TRENDING_UP") == 0.3

    def test_regime_to_score_trending_down(self):
        assert SignalAggregator.regime_to_score("TRENDING_DOWN") == -0.3

    def test_regime_to_score_ranging(self):
        assert SignalAggregator.regime_to_score("RANGING") == 0.0

    def test_regime_to_score_unknown(self):
        assert SignalAggregator.regime_to_score("UNKNOWN") == 0.0


# ── Weight adjustment tests ──────────────────────────────────────

class TestWeightAdjustment:
    def test_no_adjustment_when_disabled(self):
        agg = SignalAggregator(regime_adaptive=False)
        w = {"ta": 0.5, "sentiment": 0.5}
        result = agg._adjust_weights_for_regime(w, "TRENDING_UP")
        assert result == w

    def test_no_adjustment_unknown_regime(self):
        agg = SignalAggregator()
        w = {"ta": 0.5, "sentiment": 0.5}
        result = agg._adjust_weights_for_regime(w, "UNKNOWN_REGIME")
        assert result == w

    def test_adjustment_renormalises(self):
        agg = SignalAggregator()
        w = {"ta": 0.5, "sentiment": 0.3, "mtf": 0.2}
        result = agg._adjust_weights_for_regime(w, "TRENDING_UP")
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)

    def test_volatile_boosts_sentiment(self):
        agg = SignalAggregator()
        w = {"ta": 0.5, "sentiment": 0.5}
        result = agg._adjust_weights_for_regime(w, "VOLATILE")
        # sentiment factor is 1.2, ta factor is 0.8
        assert result["sentiment"] > result["ta"]


# ── Aggregation tests ────────────────────────────────────────────

class TestAggregation:
    def test_no_sources_returns_error(self):
        agg = SignalAggregator()
        result = agg.aggregate("AAPL")
        assert result.error == "No signal sources provided"
        assert result.signal == "HOLD"

    def test_ta_only_buy(self):
        agg = SignalAggregator()
        result = agg.aggregate("AAPL", ta_result=_ta(score=4, signal="BUY"))
        assert result.source_count == 1
        assert result.signal == "BUY"
        assert result.combined_score > 0

    def test_ta_only_sell(self):
        agg = SignalAggregator()
        result = agg.aggregate("AAPL", ta_result=_ta(score=-4, signal="SELL"))
        assert result.signal == "SELL"
        assert result.combined_score < 0

    def test_ta_only_hold(self):
        agg = SignalAggregator()
        result = agg.aggregate("AAPL", ta_result=_ta(score=0, signal="HOLD"))
        assert result.signal == "HOLD"
        assert result.combined_score == pytest.approx(0.0)

    def test_ta_plus_sentiment_buy(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=3, signal="BUY"),
            sentiment_result=_sent(score=0.3, signal="BUY"),
        )
        assert result.source_count == 2
        assert result.signal == "BUY"

    def test_ta_plus_sentiment_mixed(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=1, signal="HOLD"),
            sentiment_result=_sent(score=-0.3, signal="SELL"),
        )
        # TA norm = 1/6 ~= 0.167, sent norm = -0.3
        # Weights normalised: ta ~= 0.667, sent ~= 0.333
        # combined ~= 0.667*0.167 + 0.333*(-0.3) ~= 0.111 - 0.1 = 0.011
        assert result.signal == "HOLD"

    def test_all_four_sources(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=3, signal="BUY"),
            sentiment_result=_sent(score=0.2, signal="BUY"),
            mtf_result=_mtf(score=2.0, signal="BUY"),
            regime_result=_regime("TRENDING_UP"),
        )
        assert result.source_count == 4
        assert result.signal == "BUY"
        assert result.combined_score > 0

    def test_weights_normalised_to_one(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=3, signal="BUY"),
            sentiment_result=_sent(score=0.2, signal="BUY"),
        )
        total = sum(s.weight for s in result.sources)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_min_sources_gate(self):
        agg = SignalAggregator(min_sources=3)
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=3, signal="BUY"),
        )
        assert "Insufficient sources" in result.error

    def test_custom_thresholds(self):
        agg = SignalAggregator(buy_threshold=0.5)
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=2, signal="BUY"),
        )
        # norm = 2/6 = 0.333, which is < 0.5 threshold
        assert result.signal == "HOLD"


# ── Regime integration ───────────────────────────────────────────

class TestRegimeIntegration:
    def test_regime_adds_directional_bias(self):
        agg = SignalAggregator()
        # TA alone is on the edge — regime should nudge it
        result_no_regime = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=1, signal="HOLD"),
        )
        result_with_regime = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=1, signal="HOLD"),
            regime_result=_regime("TRENDING_UP"),
        )
        assert result_with_regime.combined_score >= result_no_regime.combined_score

    def test_regime_adjusts_weights(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=3, signal="BUY"),
            sentiment_result=_sent(score=0.2, signal="BUY"),
            regime_result=_regime("VOLATILE"),
        )
        assert result.regime == "VOLATILE"
        assert result.regime_adjusted is True

    def test_regime_adjusts_thresholds(self):
        agg = SignalAggregator(buy_threshold=0.25)
        # VOLATILE regime has min_score_adjustment=2 → raises buy threshold
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=2, signal="BUY"),
            regime_result=_regime("VOLATILE"),
        )
        # Threshold should be raised, making it harder to trigger BUY
        assert result.buy_threshold > 0.25

    def test_trending_up_lowers_threshold(self):
        agg = SignalAggregator(buy_threshold=0.25)
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=2, signal="BUY"),
            regime_result=_regime("TRENDING_UP"),
        )
        # TRENDING_UP has min_score_adjustment=-1 → lowers threshold
        assert result.buy_threshold < 0.25


# ── Correlation filtering ────────────────────────────────────────

class TestCorrelationFiltering:
    def test_no_filter_when_not_correlated(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=4, signal="BUY"),
            correlation_result=_corr(is_correlated=False),
        )
        assert result.signal == "BUY"
        assert result.correlation_filtered is False

    def test_buy_downgraded_when_correlated(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=4, signal="BUY"),
            correlation_result=_corr(is_correlated=True, correlated_with=["MSFT"]),
        )
        assert result.signal == "HOLD"
        assert result.correlation_filtered is True
        assert result.original_signal == "BUY"

    def test_sell_not_filtered(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=-4, signal="SELL"),
            correlation_result=_corr(is_correlated=True),
        )
        assert result.signal == "SELL"
        assert result.correlation_filtered is False

    def test_hold_not_filtered(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=0, signal="HOLD"),
            correlation_result=_corr(is_correlated=True),
        )
        assert result.signal == "HOLD"
        assert result.correlation_filtered is False


# ── Confidence & agreement ───────────────────────────────────────

class TestConfidence:
    def test_unanimous_agreement_bonus(self):
        agg = SignalAggregator(agreement_bonus=0.1)
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=4, signal="BUY"),
            sentiment_result=_sent(score=0.5, signal="BUY"),
        )
        assert result.agreement_ratio == 1.0
        # Confidence should have bonus applied
        assert result.confidence > 0

    def test_disagreement_penalises_confidence(self):
        agg = SignalAggregator()
        result_agree = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=3, signal="BUY"),
            sentiment_result=_sent(score=0.3, signal="BUY"),
        )
        result_disagree = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=3, signal="BUY"),
            sentiment_result=_sent(score=-0.1, signal="SELL"),
        )
        # When sources disagree, confidence should be lower
        assert result_disagree.confidence <= result_agree.confidence

    def test_agreement_ratio_correct(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=4, signal="BUY"),
            sentiment_result=_sent(score=-0.3, signal="SELL"),
        )
        # 1 out of 2 sources agree (BUY signal, only TA agrees)
        assert result.agreement_ratio == 0.5

    def test_single_source_no_bonus(self):
        agg = SignalAggregator(agreement_bonus=0.1)
        result = agg.aggregate("AAPL", ta_result=_ta(score=4, signal="BUY"))
        # Single source: agreement is 100% but no bonus for single source
        assert result.source_count == 1


# ── Formatting ───────────────────────────────────────────────────

class TestFormatting:
    def test_format_result_basic(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=3, signal="BUY"),
        )
        text = SignalAggregator.format_result(result)
        assert "AAPL" in text
        assert "SIGNAL AGGREGATOR" in text
        assert "BUY" in text

    def test_format_result_error(self):
        result = AggregatedSignal(symbol="AAPL", error="test error")
        text = SignalAggregator.format_result(result)
        assert "test error" in text

    def test_format_result_with_regime(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=3, signal="BUY"),
            regime_result=_regime("TRENDING_UP"),
        )
        text = SignalAggregator.format_result(result)
        assert "TRENDING_UP" in text

    def test_format_result_with_correlation(self):
        agg = SignalAggregator()
        result = agg.aggregate(
            "AAPL",
            ta_result=_ta(score=4, signal="BUY"),
            correlation_result=_corr(is_correlated=True),
        )
        text = SignalAggregator.format_result(result)
        assert "FILTERED" in text

    def test_format_dict(self):
        agg = SignalAggregator()
        result = agg.aggregate("AAPL", ta_result=_ta(score=3, signal="BUY"))
        text = SignalAggregator.format_result(result.__dict__)
        assert "AAPL" in text


# ── Default weights ──────────────────────────────────────────────

class TestDefaults:
    def test_default_weights_sum_to_one(self):
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_all_regimes_have_weight_adjustments(self):
        expected = {"TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "BREAKOUT"}
        assert set(REGIME_WEIGHT_ADJUSTMENTS.keys()) == expected

    def test_regime_weight_adjustment_keys(self):
        for regime, factors in REGIME_WEIGHT_ADJUSTMENTS.items():
            for key in DEFAULT_WEIGHTS:
                assert key in factors, f"{regime} missing factor for {key}"
