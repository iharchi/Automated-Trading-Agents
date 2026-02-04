"""Portfolio Manager Agent (Orchestrator)

Collects signals from the Technical Analysis and Sentiment Analysis
agents, normalises their scores to a common [-1, +1] scale, applies
configurable weights, and produces a single unified trading decision
per symbol.  That decision is then validated through the Risk
Management agent before any order is placed.

Pipeline:
    TA Agent  ──┐
                 ├──> Portfolio Manager ──> Risk Manager ──> Order
    Sentiment ──┘

Scoring:
    1. TA composite_score (int, roughly -4..+4) is normalised to [-1, +1]
       by dividing by the maximum possible score (4 indicators).
    2. Sentiment composite_score (float, already in [-1, +1]) is used as-is.
    3. Weighted sum:  score = w_ta * ta_norm + w_sent * sent_norm
    4. Thresholds map the combined score to BUY / SELL / HOLD.
    5. Confidence = abs(combined score), clamped to [0, 1].
"""

import logging
from dataclasses import dataclass, field

from agents.base_agent import BaseAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.sentiment_analysis_agent import SentimentAnalysisAgent
from agents.risk_management_agent import RiskManagementAgent
from utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────

DEFAULT_TA_WEIGHT = 0.65
DEFAULT_SENTIMENT_WEIGHT = 0.35
BUY_THRESHOLD = 0.25
SELL_THRESHOLD = -0.25
TA_MAX_SCORE = 6  # 6 voting indicators, each ±1


@dataclass
class AgentSignal:
    """Normalised signal from a single agent."""

    agent: str
    raw_score: float
    normalised_score: float
    signal: str
    weight: float
    weighted_contribution: float


@dataclass
class PortfolioDecision:
    """The unified trading decision for one symbol."""

    symbol: str
    agent_signals: list[AgentSignal] = field(default_factory=list)
    combined_score: float = 0.0
    confidence: float = 0.0
    signal: str = "HOLD"
    current_price: float = 0.0
    atr: float = 0.0
    risk_approved: bool | None = None  # None = risk check not run
    position_size: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    order: dict | str | None = None


class PortfolioManagerAgent(BaseAgent):
    """Orchestrator that combines agent signals and manages execution."""

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        ta_weight: float = DEFAULT_TA_WEIGHT,
        sentiment_weight: float = DEFAULT_SENTIMENT_WEIGHT,
        buy_threshold: float = BUY_THRESHOLD,
        sell_threshold: float = SELL_THRESHOLD,
        auto_trade: bool = False,
    ):
        super().__init__(name="PortfolioManager", client=client)
        # Normalise weights so they sum to 1
        total = ta_weight + sentiment_weight
        self.ta_weight = ta_weight / total
        self.sentiment_weight = sentiment_weight / total
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.auto_trade = auto_trade

        # Sub-agents share the same client to reuse the connection
        self.ta_agent = TechnicalAnalysisAgent(client=self.client, auto_trade=False)
        self.sentiment_agent = SentimentAnalysisAgent(client=self.client, auto_trade=False)
        self.risk_agent = RiskManagementAgent(client=self.client, auto_trade=auto_trade)

    # ── Normalisation ────────────────────────────────────────

    @staticmethod
    def _normalise_ta(composite_score: int) -> float:
        """Map TA composite score (roughly -4..+4) into [-1, +1]."""
        return max(-1.0, min(1.0, composite_score / TA_MAX_SCORE))

    @staticmethod
    def _normalise_sentiment(composite_score: float) -> float:
        """Sentiment score is already in [-1, +1]; just clamp."""
        return max(-1.0, min(1.0, composite_score))

    # ── Core lifecycle ───────────────────────────────────────

    def analyze(self, symbol: str, **kwargs) -> dict:
        """Run sub-agents, combine signals, run risk check."""

        # ── 1. Technical Analysis ────────────────────────────
        ta_result = self.ta_agent.analyze(symbol, **kwargs)
        ta_raw = ta_result.get("composite_score", 0)
        ta_norm = self._normalise_ta(ta_raw)

        print(TechnicalAnalysisAgent.format_analysis(ta_result))

        # ── 2. Sentiment Analysis ────────────────────────────
        sent_result = self.sentiment_agent.analyze(symbol, **kwargs)
        sent_raw = sent_result.get("composite_score", 0.0)
        sent_norm = self._normalise_sentiment(sent_raw)

        print(SentimentAnalysisAgent.format_analysis(sent_result))

        # ── 3. Weighted combination ──────────────────────────
        ta_contrib = self.ta_weight * ta_norm
        sent_contrib = self.sentiment_weight * sent_norm
        combined = ta_contrib + sent_contrib

        agent_signals = [
            AgentSignal(
                agent="TechnicalAnalysis",
                raw_score=ta_raw,
                normalised_score=round(ta_norm, 4),
                signal=ta_result.get("signal", "HOLD"),
                weight=round(self.ta_weight, 2),
                weighted_contribution=round(ta_contrib, 4),
            ),
            AgentSignal(
                agent="SentimentAnalysis",
                raw_score=round(sent_raw, 4),
                normalised_score=round(sent_norm, 4),
                signal=sent_result.get("signal", "HOLD"),
                weight=round(self.sentiment_weight, 2),
                weighted_contribution=round(sent_contrib, 4),
            ),
        ]

        if combined >= self.buy_threshold:
            signal = "BUY"
        elif combined <= self.sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        confidence = round(min(abs(combined), 1.0), 4)
        current_price = ta_result.get("current_price", 0.0)
        atr = ta_result.get("atr", 0.0)

        decision = PortfolioDecision(
            symbol=symbol,
            agent_signals=agent_signals,
            combined_score=round(combined, 4),
            confidence=confidence,
            signal=signal,
            current_price=current_price,
            atr=atr,
        )

        # ── 4. Risk Management (for actionable signals) ─────
        if signal != "HOLD":
            proposed_side = "buy" if signal == "BUY" else "sell"
            risk_result = self.risk_agent.analyze(
                symbol,
                proposed_side=proposed_side,
                price=current_price,
                atr=atr,
            )
            print(RiskManagementAgent.format_analysis(risk_result))

            decision.risk_approved = risk_result.get("approved", False)
            decision.position_size = risk_result.get("position_size", 0)
            decision.stop_loss = risk_result.get("stop_loss", 0.0)
            decision.take_profit = risk_result.get("take_profit", 0.0)

        return decision.__dict__

    def execute(self, symbol: str, analysis: dict, **kwargs) -> dict:
        """Place an order if the signal is actionable and risk-approved."""
        result = {
            "symbol": symbol,
            "signal": analysis.get("signal", "HOLD"),
            "combined_score": analysis.get("combined_score"),
            "confidence": analysis.get("confidence"),
            "risk_approved": analysis.get("risk_approved"),
            "position_size": analysis.get("position_size", 0),
            "order": None,
        }

        if not self.auto_trade:
            return result
        if analysis.get("signal") == "HOLD":
            return result
        if not analysis.get("risk_approved"):
            return result

        side = "buy" if analysis["signal"] == "BUY" else "sell"
        qty = analysis["position_size"]

        if qty <= 0:
            return result

        order = self.client.submit_order(symbol=symbol, qty=qty, side=side)
        result["order"] = order
        return result

    # ── Pretty print ─────────────────────────────────────────

    @staticmethod
    def format_analysis(analysis: dict) -> str:
        """Human-readable summary of the portfolio decision."""
        lines = [
            f"\n{'#' * 62}",
            f"  PORTFOLIO DECISION — {analysis['symbol']}",
            f"  Price: ${analysis['current_price']}  |  "
            f"Confidence: {analysis['confidence']:.0%}",
            f"{'#' * 62}",
        ]

        for sig in analysis.get("agent_signals", []):
            if isinstance(sig, dict):
                name = sig["agent"]
                raw = sig["raw_score"]
                norm = sig["normalised_score"]
                w = sig["weight"]
                contrib = sig["weighted_contribution"]
                sub_signal = sig["signal"]
            else:
                name = sig.agent
                raw = sig.raw_score
                norm = sig.normalised_score
                w = sig.weight
                contrib = sig.weighted_contribution
                sub_signal = sig.signal

            lines.append(
                f"  {name:22s}  raw={raw:>+8.4f}  "
                f"norm={norm:+.4f}  "
                f"x{w:.2f} = {contrib:+.4f}  "
                f"[{sub_signal}]"
            )

        lines.append(f"{'─' * 62}")
        lines.append(
            f"  Combined Score: {analysis['combined_score']:+.4f}  →  "
            f"Signal: {analysis['signal']}"
        )

        risk = analysis.get("risk_approved")
        if risk is not None:
            verdict = "APPROVED" if risk else "REJECTED"
            lines.append(f"  Risk Check: {verdict}")
            if risk:
                lines.append(
                    f"  Size: {analysis['position_size']} shares  |  "
                    f"Stop: ${analysis['stop_loss']}  |  "
                    f"Target: ${analysis['take_profit']}"
                )
        else:
            lines.append("  Risk Check: skipped (HOLD signal)")

        lines.append(f"{'#' * 62}\n")
        return "\n".join(lines)
