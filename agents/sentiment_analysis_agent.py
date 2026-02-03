"""Sentiment Analysis Agent

Fetches recent news articles for a symbol via the Alpaca News API,
scores each headline (and summary when available) using VADER sentiment
analysis, and produces a composite BUY / SELL / HOLD signal.

Scoring approach:
    1. Each article gets a VADER compound score in [-1, +1].
    2. Recent articles are weighted more heavily (exponential decay).
    3. The weighted average becomes the composite sentiment score.
    4. Thresholds map the score to a trading signal.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from agents.base_agent import BaseAgent
from utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

# ── Signal thresholds ────────────────────────────────────────────

BULLISH_THRESHOLD = 0.15
BEARISH_THRESHOLD = -0.15

# Half-life for time-decay weighting (in hours).
# Articles older than this are weighted ~50 % less.
DECAY_HALF_LIFE_HOURS = 48


@dataclass
class ArticleScore:
    """Sentiment result for a single news article."""

    headline: str
    source: str
    created_at: str
    compound: float
    weight: float
    weighted_score: float


@dataclass
class SentimentResult:
    """Aggregated sentiment analysis for one symbol."""

    symbol: str
    article_count: int = 0
    articles: list[ArticleScore] = field(default_factory=list)
    composite_score: float = 0.0
    signal: str = "HOLD"
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0


class SentimentAnalysisAgent(BaseAgent):
    """Agent that generates trading signals from news sentiment."""

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        news_limit: int = 20,
        lookback_days: int = 7,
        bullish_threshold: float = BULLISH_THRESHOLD,
        bearish_threshold: float = BEARISH_THRESHOLD,
        auto_trade: bool = False,
        default_qty: int = 1,
    ):
        super().__init__(name="SentimentAnalysis", client=client)
        self.news_limit = news_limit
        self.lookback_days = lookback_days
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.auto_trade = auto_trade
        self.default_qty = default_qty
        self.analyzer = SentimentIntensityAnalyzer()

    # ── Helpers ──────────────────────────────────────────────

    def _score_text(self, text: str) -> float:
        """Return the VADER compound score for a piece of text."""
        return self.analyzer.polarity_scores(text)["compound"]

    @staticmethod
    def _time_weight(created_at: str) -> float:
        """Exponential decay weight based on article age."""
        try:
            # Alpaca timestamps look like "2025-05-20 14:30:00+00:00"
            article_dt = datetime.fromisoformat(created_at)
            now = datetime.now(tz=article_dt.tzinfo)
            age_hours = max((now - article_dt).total_seconds() / 3600, 0)
        except (ValueError, TypeError):
            age_hours = DECAY_HALF_LIFE_HOURS  # fallback: half weight
        return math.exp(-0.693 * age_hours / DECAY_HALF_LIFE_HOURS)

    def _score_article(self, article: dict) -> ArticleScore:
        """Score a single article dict from AlpacaClient.get_news()."""
        headline = article.get("headline", "")
        summary = article.get("summary", "")

        # Combine headline + summary; headline is more important
        text = headline
        if summary:
            text = f"{headline}. {summary}"

        compound = self._score_text(text)
        weight = self._time_weight(article.get("created_at", ""))
        weighted = compound * weight

        return ArticleScore(
            headline=headline,
            source=article.get("source", "unknown"),
            created_at=article.get("created_at", ""),
            compound=round(compound, 4),
            weight=round(weight, 4),
            weighted_score=round(weighted, 4),
        )

    # ── Core lifecycle ───────────────────────────────────────

    def analyze(self, symbol: str, **kwargs) -> dict:
        """Fetch news, score articles, return SentimentResult as dict."""
        lookback = kwargs.get("lookback_days", self.lookback_days)
        limit = kwargs.get("news_limit", self.news_limit)

        start = (datetime.now() - timedelta(days=lookback)).strftime(
            "%Y-%m-%dT00:00:00Z"
        )

        articles_raw = self.client.get_news(symbol, limit=limit, start=start)
        if not articles_raw:
            self.logger.warning("No news found for %s", symbol)
            return SentimentResult(symbol=symbol).__dict__

        scored: list[ArticleScore] = [
            self._score_article(a) for a in articles_raw
        ]

        total_weight = sum(s.weight for s in scored)
        if total_weight > 0:
            composite = sum(s.weighted_score for s in scored) / total_weight
        else:
            composite = 0.0

        bullish = sum(1 for s in scored if s.compound > 0.05)
        bearish = sum(1 for s in scored if s.compound < -0.05)
        neutral = len(scored) - bullish - bearish

        if composite >= self.bullish_threshold:
            signal = "BUY"
        elif composite <= self.bearish_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        result = SentimentResult(
            symbol=symbol,
            article_count=len(scored),
            articles=scored,
            composite_score=round(composite, 4),
            signal=signal,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
        )
        return result.__dict__

    def execute(self, symbol: str, analysis: dict, **kwargs) -> dict:
        """Optionally place a paper trade based on the sentiment signal."""
        signal = analysis.get("signal", "HOLD")
        result = {
            "symbol": symbol,
            "signal": signal,
            "composite_score": analysis.get("composite_score", 0),
            "article_count": analysis.get("article_count", 0),
            "order": None,
        }

        if not self.auto_trade or signal == "HOLD":
            return result

        qty = kwargs.get("qty", self.default_qty)
        side = "buy" if signal == "BUY" else "sell"

        if side == "sell":
            positions = {p["symbol"]: p for p in self.client.get_positions()}
            if symbol not in positions:
                self.logger.info("No position in %s to sell — skipping.", symbol)
                result["order"] = "skipped_no_position"
                return result

        order = self.client.submit_order(symbol=symbol, qty=qty, side=side)
        result["order"] = order
        return result

    # ── Pretty print ─────────────────────────────────────────

    @staticmethod
    def format_analysis(analysis: dict) -> str:
        """Return a human-readable summary of a sentiment analysis dict."""
        lines = [
            f"\n{'=' * 60}",
            f"  Sentiment Analysis — {analysis['symbol']}",
            f"  Articles: {analysis['article_count']}  "
            f"(+{analysis['bullish_count']} / "
            f"~{analysis['neutral_count']} / "
            f"-{analysis['bearish_count']})",
            f"{'=' * 60}",
        ]

        for art in analysis.get("articles", []):
            if isinstance(art, dict):
                score = art["compound"]
                headline = art["headline"]
                source = art["source"]
            else:
                score = art.compound
                headline = art.headline
                source = art.source

            icon = "+" if score > 0.05 else ("-" if score < -0.05 else "~")
            # Truncate long headlines
            short = headline[:60] + "..." if len(headline) > 60 else headline
            lines.append(f"  [{icon}] {score:+.3f}  {short}  ({source})")

        lines.append(f"{'─' * 60}")
        lines.append(
            f"  Composite Score: {analysis['composite_score']:+.4f}  →  "
            f"Signal: {analysis['signal']}"
        )
        lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)
