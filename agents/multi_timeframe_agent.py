"""Multi-Timeframe Analysis Agent

Runs technical analysis on multiple timeframes and combines signals
to produce higher-confidence trading decisions.

The idea: a signal confirmed across timeframes is more reliable than
a signal on a single timeframe. For example, a BUY on both the daily
and hourly charts is stronger than a BUY on just the daily.

Timeframes (default):
    - Primary: 1Day (captures the overall trend)
    - Confirmation: 1Hour (captures intraday momentum)
    - Optional: 15Min, 5Min for timing

Agreement modes:
    - unanimous: All timeframes must agree (most conservative)
    - majority: Majority of timeframes agree (balanced)
    - weighted: Weighted average, higher timeframes have more weight

The agent outputs a consolidated signal with a confidence score based
on how many timeframes agree.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

from agents.base_agent import BaseAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

AgreementMode = Literal["unanimous", "majority", "weighted"]

# Default timeframes in order of importance (higher = more weight)
DEFAULT_TIMEFRAMES = ["1Day", "1Hour"]

# Weights for weighted mode (higher timeframe = higher weight)
TIMEFRAME_WEIGHTS = {
    "1Day": 1.0,
    "1Hour": 0.6,
    "15Min": 0.3,
    "5Min": 0.15,
    "1Min": 0.05,
}


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""

    timeframe: str
    signal: str  # BUY, SELL, HOLD
    score: int   # composite score from TA agent
    weight: float
    price: float
    atr: float
    indicators: list = field(default_factory=list)


@dataclass
class MultiTimeframeResult:
    """Aggregated result across all timeframes."""

    symbol: str
    timeframe_signals: list[TimeframeSignal] = field(default_factory=list)
    final_signal: str = "HOLD"
    agreement_ratio: float = 0.0  # 0.0 to 1.0
    weighted_score: float = 0.0
    confidence: float = 0.0
    current_price: float = 0.0
    atr: float = 0.0
    primary_timeframe: str = "1Day"
    agreement_mode: str = "unanimous"


class MultiTimeframeAgent(BaseAgent):
    """Agent that combines signals from multiple timeframes."""

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        timeframes: list[str] | None = None,
        agreement_mode: AgreementMode = "unanimous",
        min_agreement: float = 0.5,  # for majority mode
        auto_trade: bool = False,
    ):
        super().__init__(name="MultiTimeframe", client=client)
        self.timeframes = timeframes or DEFAULT_TIMEFRAMES
        self.agreement_mode = agreement_mode
        self.min_agreement = min_agreement
        self.auto_trade = auto_trade

        # Create a shared TA agent
        self.ta_agent = TechnicalAnalysisAgent(client=self.client, auto_trade=False)

    # ── Signal combination logic ──────────────────────────────

    @staticmethod
    def _get_weight(timeframe: str) -> float:
        """Get weight for a timeframe."""
        return TIMEFRAME_WEIGHTS.get(timeframe, 0.5)

    def _combine_unanimous(self, signals: list[TimeframeSignal]) -> tuple[str, float]:
        """All timeframes must agree for a signal to fire.

        Returns (signal, confidence).
        """
        if not signals:
            return "HOLD", 0.0

        directions = [s.signal for s in signals]

        # Check if all agree on BUY
        if all(d == "BUY" for d in directions):
            return "BUY", 1.0

        # Check if all agree on SELL
        if all(d == "SELL" for d in directions):
            return "SELL", 1.0

        # Mixed signals or all HOLD
        return "HOLD", 0.0

    def _combine_majority(self, signals: list[TimeframeSignal]) -> tuple[str, float]:
        """Majority of timeframes must agree.

        Returns (signal, confidence).
        """
        if not signals:
            return "HOLD", 0.0

        buy_count = sum(1 for s in signals if s.signal == "BUY")
        sell_count = sum(1 for s in signals if s.signal == "SELL")
        total = len(signals)

        buy_ratio = buy_count / total
        sell_ratio = sell_count / total

        if buy_ratio >= self.min_agreement:
            return "BUY", buy_ratio
        elif sell_ratio >= self.min_agreement:
            return "SELL", sell_ratio
        else:
            return "HOLD", max(buy_ratio, sell_ratio)

    def _combine_weighted(self, signals: list[TimeframeSignal]) -> tuple[str, float]:
        """Weighted combination of signals.

        Returns (signal, confidence).
        """
        if not signals:
            return "HOLD", 0.0

        total_weight = sum(s.weight for s in signals)
        if total_weight == 0:
            return "HOLD", 0.0

        # Convert signals to numeric: BUY=+1, SELL=-1, HOLD=0
        weighted_sum = 0.0
        for s in signals:
            if s.signal == "BUY":
                weighted_sum += s.weight
            elif s.signal == "SELL":
                weighted_sum -= s.weight

        # Normalize to [-1, 1]
        normalized = weighted_sum / total_weight

        # Convert back to signal
        if normalized >= 0.3:
            return "BUY", abs(normalized)
        elif normalized <= -0.3:
            return "SELL", abs(normalized)
        else:
            return "HOLD", 1 - abs(normalized)

    def _combine_signals(self, signals: list[TimeframeSignal]) -> tuple[str, float]:
        """Combine signals based on agreement mode."""
        if self.agreement_mode == "unanimous":
            return self._combine_unanimous(signals)
        elif self.agreement_mode == "majority":
            return self._combine_majority(signals)
        else:  # weighted
            return self._combine_weighted(signals)

    # ── Core lifecycle ────────────────────────────────────────

    def analyze(self, symbol: str, **kwargs) -> dict:
        """Run TA on all timeframes and combine signals."""
        limit = kwargs.get("limit", 100)

        timeframe_signals: list[TimeframeSignal] = []

        for tf in self.timeframes:
            try:
                result = self.ta_agent.analyze(symbol, timeframe=tf, limit=limit)

                tf_signal = TimeframeSignal(
                    timeframe=tf,
                    signal=result.get("signal", "HOLD"),
                    score=result.get("composite_score", 0),
                    weight=self._get_weight(tf),
                    price=result.get("current_price", 0),
                    atr=result.get("atr", 0),
                    indicators=result.get("indicators", []),
                )
                timeframe_signals.append(tf_signal)

                self.logger.debug(
                    "%s %s: %s (score=%d)",
                    symbol, tf, tf_signal.signal, tf_signal.score,
                )
            except Exception as e:
                self.logger.warning("Failed to analyze %s on %s: %s", symbol, tf, e)

        if not timeframe_signals:
            return MultiTimeframeResult(symbol=symbol).__dict__

        # Combine signals
        final_signal, confidence = self._combine_signals(timeframe_signals)

        # Calculate agreement ratio
        if final_signal != "HOLD":
            agreeing = sum(1 for s in timeframe_signals if s.signal == final_signal)
            agreement_ratio = agreeing / len(timeframe_signals)
        else:
            agreement_ratio = 0.0

        # Calculate weighted score
        total_weight = sum(s.weight for s in timeframe_signals)
        weighted_score = sum(s.score * s.weight for s in timeframe_signals) / total_weight if total_weight else 0

        # Use primary timeframe's price and ATR
        primary = next((s for s in timeframe_signals if s.timeframe == self.timeframes[0]), None)
        current_price = primary.price if primary else 0
        atr = primary.atr if primary else 0

        result = MultiTimeframeResult(
            symbol=symbol,
            timeframe_signals=timeframe_signals,
            final_signal=final_signal,
            agreement_ratio=agreement_ratio,
            weighted_score=weighted_score,
            confidence=confidence,
            current_price=current_price,
            atr=atr,
            primary_timeframe=self.timeframes[0],
            agreement_mode=self.agreement_mode,
        )

        return result.__dict__

    def execute(self, symbol: str, analysis: dict, **kwargs) -> dict:
        """Execute trade based on multi-timeframe analysis."""
        signal = analysis.get("final_signal", "HOLD")

        if signal == "HOLD" or not self.auto_trade:
            return {"order": None, "reason": "No action (HOLD or auto_trade=False)"}

        # Delegate to TA agent's execute with the primary timeframe result
        primary_tf = analysis.get("primary_timeframe", "1Day")
        tf_signals = analysis.get("timeframe_signals", [])

        primary_result = None
        for s in tf_signals:
            tf = s.get("timeframe") if isinstance(s, dict) else s.timeframe
            if tf == primary_tf:
                primary_result = s if isinstance(s, dict) else s.__dict__
                break

        if not primary_result:
            return {"order": None, "reason": "No primary timeframe result"}

        # Build a mock TA result for execution
        ta_result = {
            "signal": signal,
            "current_price": analysis.get("current_price", 0),
            "atr": analysis.get("atr", 0),
        }

        return self.ta_agent.execute(symbol, ta_result)

    # ── Pretty print ──────────────────────────────────────────

    @staticmethod
    def format_analysis(analysis: dict) -> str:
        """Human-readable multi-timeframe analysis."""
        symbol = analysis.get("symbol", "???")
        final_signal = analysis.get("final_signal", "HOLD")
        agreement_ratio = analysis.get("agreement_ratio", 0)
        confidence = analysis.get("confidence", 0)
        weighted_score = analysis.get("weighted_score", 0)
        mode = analysis.get("agreement_mode", "unanimous")
        price = analysis.get("current_price", 0)

        lines = [
            f"\n{'=' * 62}",
            f"  MULTI-TIMEFRAME ANALYSIS — {symbol}",
            f"{'=' * 62}",
        ]

        # Per-timeframe breakdown
        for tf_sig in analysis.get("timeframe_signals", []):
            if isinstance(tf_sig, dict):
                tf = tf_sig.get("timeframe", "?")
                sig = tf_sig.get("signal", "HOLD")
                score = tf_sig.get("score", 0)
                weight = tf_sig.get("weight", 0)
            else:
                tf = tf_sig.timeframe
                sig = tf_sig.signal
                score = tf_sig.score
                weight = tf_sig.weight

            icon = {"BUY": "+", "SELL": "-", "HOLD": "o"}.get(sig, "?")
            lines.append(f"  [{icon}] {tf:8s}  {sig:5s}  score={score:+d}  weight={weight:.2f}")

        lines.append(f"{'─' * 62}")
        lines.append(f"  Mode       : {mode}")
        lines.append(f"  Agreement  : {agreement_ratio:.0%}")
        lines.append(f"  Confidence : {confidence:.0%}")
        lines.append(f"  Wtd. Score : {weighted_score:+.2f}")
        lines.append(f"  Price      : ${price:.2f}")
        lines.append(f"{'─' * 62}")

        signal_icon = {"BUY": "▲ BUY", "SELL": "▼ SELL", "HOLD": "● HOLD"}.get(final_signal, final_signal)
        lines.append(f"  SIGNAL     : {signal_icon}")
        lines.append(f"{'=' * 62}\n")

        return "\n".join(lines)
