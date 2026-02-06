"""Signal Aggregator / Meta-Strategy Engine

Combines signals from multiple analysis sources (Technical Analysis,
Sentiment Analysis, Multi-Timeframe, Market Regime) into a single
unified trading decision with confidence scoring.

Features:
    - Configurable per-source weights (auto-normalised)
    - Regime-adaptive weight adjustment (optional)
    - Correlation-based signal filtering
    - Confidence scoring from source agreement
    - Works with any subset of available sources

Usage:
    aggregator = SignalAggregator()
    result = aggregator.aggregate(
        "AAPL",
        ta_result=ta_analysis,
        sentiment_result=sent_analysis,
        regime_result=regime,
    )
    print(SignalAggregator.format_result(result))
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

TA_MAX_SCORE = 6  # 6 voting indicators in TA agent, each +/-1
MTF_MAX_SCORE = 6.0  # Same indicator set, scaled by timeframe weights

# Default weights for each signal source
DEFAULT_WEIGHTS: dict[str, float] = {
    "ta": 0.40,
    "sentiment": 0.20,
    "mtf": 0.30,
    "regime": 0.10,
}

# Regime-specific weight multipliers (applied before re-normalisation)
REGIME_WEIGHT_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "TRENDING_UP": {"ta": 1.1, "sentiment": 0.9, "mtf": 1.1, "regime": 1.0},
    "TRENDING_DOWN": {"ta": 1.1, "sentiment": 1.1, "mtf": 1.0, "regime": 1.0},
    "RANGING": {"ta": 0.9, "sentiment": 1.0, "mtf": 1.1, "regime": 1.0},
    "VOLATILE": {"ta": 0.8, "sentiment": 1.2, "mtf": 0.9, "regime": 1.2},
    "BREAKOUT": {"ta": 1.2, "sentiment": 0.8, "mtf": 1.1, "regime": 1.0},
}


@dataclass
class SourceSignal:
    """Normalised signal contribution from one analysis source."""

    name: str
    raw_score: float
    normalised_score: float  # [-1, +1]
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float  # [0, 1]
    weight: float  # Effective weight (sums to 1 across active sources)
    weighted_contribution: float  # normalised_score * weight


@dataclass
class AggregatedSignal:
    """Unified trading decision produced by the aggregator."""

    symbol: str
    sources: list[SourceSignal] = field(default_factory=list)
    combined_score: float = 0.0
    signal: str = "HOLD"
    confidence: float = 0.0
    agreement_ratio: float = 0.0  # Fraction of sources agreeing with signal
    regime: str = ""
    regime_adjusted: bool = False
    correlation_filtered: bool = False
    original_signal: str = ""  # Signal before correlation gate
    adjustments: dict = field(default_factory=dict)
    source_count: int = 0
    buy_threshold: float = 0.25
    sell_threshold: float = -0.25
    error: str = ""


class SignalAggregator:
    """Combines multi-source signals into a unified trading decision."""

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        buy_threshold: float = 0.10,  # Lowered from 0.25 for more signals
        sell_threshold: float = -0.10,  # Lowered from -0.25 for more signals
        min_sources: int = 1,
        regime_adaptive: bool = True,
        agreement_bonus: float = 0.10,
    ):
        self.base_weights = dict(weights or DEFAULT_WEIGHTS)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_sources = min_sources
        self.regime_adaptive = regime_adaptive
        self.agreement_bonus = agreement_bonus

    # ── Normalisation helpers ──────────────────────────────────────

    @staticmethod
    def normalise_ta(composite_score: int | float, max_score: int = TA_MAX_SCORE) -> float:
        """Normalise TA composite score to [-1, +1]."""
        if max_score == 0:
            return 0.0
        return max(-1.0, min(1.0, composite_score / max_score))

    @staticmethod
    def normalise_sentiment(composite_score: float) -> float:
        """Clamp sentiment score to [-1, +1]."""
        return max(-1.0, min(1.0, composite_score))

    @staticmethod
    def normalise_mtf(weighted_score: float, max_score: float = MTF_MAX_SCORE) -> float:
        """Normalise multi-timeframe weighted score to [-1, +1]."""
        if max_score == 0:
            return 0.0
        return max(-1.0, min(1.0, weighted_score / max_score))

    @staticmethod
    def regime_to_score(regime: str) -> float:
        """Map a market regime to a directional score bias."""
        return {
            "TRENDING_UP": 0.3,
            "TRENDING_DOWN": -0.3,
            "RANGING": 0.0,
            "VOLATILE": 0.0,
            "BREAKOUT": 0.1,
        }.get(regime, 0.0)

    # ── Weight adjustment ─────────────────────────────────────────

    def _adjust_weights_for_regime(
        self, weights: dict[str, float], regime: str,
    ) -> dict[str, float]:
        """Multiply weights by regime-specific factors, then re-normalise."""
        if not self.regime_adaptive or regime not in REGIME_WEIGHT_ADJUSTMENTS:
            return weights

        factors = REGIME_WEIGHT_ADJUSTMENTS[regime]
        adjusted = {k: v * factors.get(k, 1.0) for k, v in weights.items()}

        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    # ── Confidence & agreement ────────────────────────────────────

    def _calculate_confidence(
        self,
        sources: list[SourceSignal],
        combined_score: float,
        signal: str,
    ) -> tuple[float, float]:
        """Return (confidence, agreement_ratio)."""
        if not sources:
            return 0.0, 0.0

        base_confidence = min(abs(combined_score), 1.0)

        agreeing = sum(1 for s in sources if s.signal == signal)
        total = len(sources)
        agreement = agreeing / total if total > 0 else 0.0

        confidence = base_confidence
        if agreement == 1.0 and total > 1:
            confidence = min(confidence + self.agreement_bonus, 1.0)
        elif agreement < 0.5:
            confidence *= 0.8

        return round(confidence, 4), round(agreement, 4)

    # ── Main aggregation ──────────────────────────────────────────

    def aggregate(
        self,
        symbol: str,
        *,
        ta_result: dict | None = None,
        sentiment_result: dict | None = None,
        mtf_result: dict | None = None,
        regime_result: dict | None = None,
        correlation_result: dict | None = None,
    ) -> AggregatedSignal:
        """Aggregate signals from all available sources.

        Args:
            symbol: Ticker symbol.
            ta_result: Dict from TechnicalAnalysisAgent.analyze().
            sentiment_result: Dict from SentimentAnalysisAgent.analyze().
            mtf_result: Dict from MultiTimeframeAgent.analyze().
            regime_result: Dict from MarketRegimeDetector.detect() (__dict__).
            correlation_result: Dict from CorrelationFilter.check_correlation().

        Returns:
            AggregatedSignal with unified decision.
        """
        sources: list[SourceSignal] = []

        # ── Extract regime info (affects weights) ────────────────
        regime_name = ""
        regime_adjustments: dict = {}

        if regime_result:
            if hasattr(regime_result, "regime"):
                regime_name = regime_result.regime
                regime_adjustments = getattr(regime_result, "adjustments", {}) or {}
            else:
                regime_name = regime_result.get("regime", "")
                regime_adjustments = regime_result.get("adjustments", {})

        # ── Determine active weights (only for provided sources) ─
        active_weights: dict[str, float] = {}
        if ta_result is not None:
            active_weights["ta"] = self.base_weights.get("ta", 0.4)
        if sentiment_result is not None:
            active_weights["sentiment"] = self.base_weights.get("sentiment", 0.2)
        if mtf_result is not None:
            active_weights["mtf"] = self.base_weights.get("mtf", 0.3)
        if regime_result is not None:
            active_weights["regime"] = self.base_weights.get("regime", 0.1)

        if not active_weights:
            return AggregatedSignal(
                symbol=symbol,
                error="No signal sources provided",
            )

        # Normalise to sum to 1
        total_w = sum(active_weights.values())
        if total_w > 0:
            active_weights = {k: v / total_w for k, v in active_weights.items()}

        # Apply regime adjustments to weights
        regime_adjusted = False
        if regime_name and self.regime_adaptive:
            adjusted = self._adjust_weights_for_regime(active_weights, regime_name)
            if adjusted != active_weights:
                active_weights = adjusted
                regime_adjusted = True

        # ── Build SourceSignal for each source ───────────────────

        # Technical Analysis
        if ta_result is not None and "ta" in active_weights:
            raw = ta_result.get("composite_score", 0)
            norm = self.normalise_ta(raw)
            w = active_weights["ta"]
            sources.append(SourceSignal(
                name="ta",
                raw_score=raw,
                normalised_score=round(norm, 4),
                signal=ta_result.get("signal", "HOLD"),
                confidence=round(min(abs(norm), 1.0), 4),
                weight=round(w, 4),
                weighted_contribution=round(norm * w, 4),
            ))

        # Sentiment Analysis
        if sentiment_result is not None and "sentiment" in active_weights:
            raw = sentiment_result.get("composite_score", 0.0)
            norm = self.normalise_sentiment(raw)
            w = active_weights["sentiment"]
            sources.append(SourceSignal(
                name="sentiment",
                raw_score=round(raw, 4),
                normalised_score=round(norm, 4),
                signal=sentiment_result.get("signal", "HOLD"),
                confidence=round(min(abs(norm), 1.0), 4),
                weight=round(w, 4),
                weighted_contribution=round(norm * w, 4),
            ))

        # Multi-Timeframe
        if mtf_result is not None and "mtf" in active_weights:
            raw = mtf_result.get("weighted_score", 0.0)
            norm = self.normalise_mtf(raw)
            w = active_weights["mtf"]
            mtf_conf = mtf_result.get("confidence", round(min(abs(norm), 1.0), 4))
            sources.append(SourceSignal(
                name="mtf",
                raw_score=round(raw, 4),
                normalised_score=round(norm, 4),
                signal=mtf_result.get("final_signal", "HOLD"),
                confidence=round(mtf_conf, 4),
                weight=round(w, 4),
                weighted_contribution=round(norm * w, 4),
            ))

        # Market Regime (directional bias)
        if regime_result is not None and "regime" in active_weights:
            regime_score = self.regime_to_score(regime_name)
            w = active_weights["regime"]
            regime_signal = "HOLD"
            if regime_score > 0.1:
                regime_signal = "BUY"
            elif regime_score < -0.1:
                regime_signal = "SELL"

            regime_confidence = 0.0
            if hasattr(regime_result, "confidence"):
                regime_confidence = regime_result.confidence
            elif isinstance(regime_result, dict):
                regime_confidence = regime_result.get("confidence", 0.0)

            sources.append(SourceSignal(
                name="regime",
                raw_score=regime_score,
                normalised_score=round(regime_score, 4),
                signal=regime_signal,
                confidence=round(regime_confidence, 4),
                weight=round(w, 4),
                weighted_contribution=round(regime_score * w, 4),
            ))

        # ── Minimum sources gate ─────────────────────────────────
        if len(sources) < self.min_sources:
            return AggregatedSignal(
                symbol=symbol,
                sources=sources,
                source_count=len(sources),
                error=f"Insufficient sources: {len(sources)} < {self.min_sources}",
            )

        # ── Combined score ───────────────────────────────────────
        combined_score = sum(s.weighted_contribution for s in sources)

        # Adjust thresholds based on regime
        buy_thresh = self.buy_threshold
        sell_thresh = self.sell_threshold
        if regime_adjustments:
            score_adj = regime_adjustments.get("min_score_adjustment", 0)
            thresh_delta = score_adj * 0.05
            buy_thresh += thresh_delta
            sell_thresh -= thresh_delta

        # ── Signal determination ─────────────────────────────────
        if combined_score >= buy_thresh:
            signal = "BUY"
        elif combined_score <= sell_thresh:
            signal = "SELL"
        else:
            signal = "HOLD"

        original_signal = signal

        # ── Confidence & agreement ───────────────────────────────
        confidence, agreement = self._calculate_confidence(
            sources, combined_score, signal,
        )

        # ── Correlation gate ─────────────────────────────────────
        correlation_filtered = False
        if correlation_result and signal == "BUY":
            is_correlated = False
            if hasattr(correlation_result, "is_correlated"):
                is_correlated = correlation_result.is_correlated
            elif isinstance(correlation_result, dict):
                is_correlated = correlation_result.get("is_correlated", False)

            if is_correlated:
                signal = "HOLD"
                correlation_filtered = True
                logger.info(
                    "[%s] BUY downgraded to HOLD (correlated with existing positions)",
                    symbol,
                )

        return AggregatedSignal(
            symbol=symbol,
            sources=sources,
            combined_score=round(combined_score, 4),
            signal=signal,
            confidence=confidence,
            agreement_ratio=agreement,
            regime=regime_name,
            regime_adjusted=regime_adjusted,
            correlation_filtered=correlation_filtered,
            original_signal=original_signal,
            adjustments=regime_adjustments,
            source_count=len(sources),
            buy_threshold=round(buy_thresh, 4),
            sell_threshold=round(sell_thresh, 4),
        )

    # ── Formatting ────────────────────────────────────────────────

    @staticmethod
    def format_result(result: "AggregatedSignal | dict") -> str:
        """Human-readable summary of an aggregated signal."""
        if isinstance(result, dict):
            r = type("_R", (), result)()  # quick attr access
        else:
            r = result

        lines = [
            f"\n{'#' * 66}",
            f"  SIGNAL AGGREGATOR -- {r.symbol}",
            f"  Sources: {r.source_count}  |  "
            f"Confidence: {r.confidence:.0%}  |  "
            f"Agreement: {r.agreement_ratio:.0%}",
            f"{'#' * 66}",
        ]

        if r.error:
            lines.append(f"  Error: {r.error}")
            lines.append(f"{'#' * 66}\n")
            return "\n".join(lines)

        for src in r.sources:
            if isinstance(src, dict):
                src = type("_S", (), src)()

            lines.append(
                f"  {src.name:15s}  raw={src.raw_score:>+8.4f}  "
                f"norm={src.normalised_score:+.4f}  "
                f"x{src.weight:.3f} = {src.weighted_contribution:+.4f}  "
                f"[{src.signal}]"
            )

        lines.append(f"{'~' * 66}")
        lines.append(
            f"  Combined Score : {r.combined_score:+.4f}  "
            f"(buy >= {r.buy_threshold:+.4f} | sell <= {r.sell_threshold:+.4f})"
        )
        lines.append(f"  Signal         : {r.signal}")

        if r.regime:
            tag = " (weights adjusted)" if r.regime_adjusted else ""
            lines.append(f"  Market Regime  : {r.regime}{tag}")

        if r.correlation_filtered:
            lines.append(
                f"  Correlation    : FILTERED (was {r.original_signal})"
            )

        lines.append(f"{'#' * 66}\n")
        return "\n".join(lines)
