"""News Analyzer

Fetches and analyzes news for stocks to identify catalysts.

Uses Alpaca's News API to:
    - Check if a stock has recent news (potential catalyst)
    - Analyze news sentiment (bullish/bearish)
    - Filter breakouts with news vs. technical-only moves

Usage:
    analyzer = NewsAnalyzer()

    # Check single stock
    news = analyzer.get_news("AAPL", days=1)

    # Analyze news sentiment
    result = analyzer.analyze_stock("AAPL")

    # Batch analyze multiple stocks
    results = analyzer.analyze_batch(["AAPL", "TSLA", "NVDA"])
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """Single news article."""
    headline: str
    summary: str
    source: str
    url: str
    timestamp: datetime
    symbols: list[str]
    sentiment_score: float = 0.0  # -1 to 1
    sentiment: str = "neutral"  # bullish, bearish, neutral


@dataclass
class NewsAnalysis:
    """News analysis result for a stock."""
    symbol: str
    has_news: bool = False
    news_count: int = 0
    latest_news_age_hours: float = 0.0
    sentiment_score: float = 0.0  # -1 (bearish) to 1 (bullish)
    sentiment: str = "neutral"  # bullish, bearish, neutral
    has_catalyst: bool = False  # Recent impactful news
    catalyst_type: str = ""  # earnings, upgrade, product, etc.
    news_items: list[NewsItem] = field(default_factory=list)
    error: str = ""


# Keywords for sentiment analysis
BULLISH_KEYWORDS = [
    "surge", "soar", "jump", "rally", "gain", "rise", "climb", "spike",
    "beat", "exceed", "outperform", "upgrade", "buy", "strong", "growth",
    "profit", "revenue", "positive", "bullish", "optimistic", "record",
    "breakthrough", "deal", "partnership", "launch", "expansion", "approve",
    "FDA approval", "contract", "win", "success", "boom", "hot", "momentum",
]

BEARISH_KEYWORDS = [
    "fall", "drop", "plunge", "crash", "decline", "sink", "tumble", "slide",
    "miss", "disappoint", "underperform", "downgrade", "sell", "weak",
    "loss", "negative", "bearish", "pessimistic", "cut", "layoff", "recall",
    "investigation", "lawsuit", "fraud", "warning", "concern", "risk",
    "SEC", "probe", "fine", "penalty", "bankruptcy", "default", "debt",
]

CATALYST_KEYWORDS = {
    "earnings": ["earnings", "EPS", "revenue", "quarterly", "Q1", "Q2", "Q3", "Q4", "guidance"],
    "upgrade": ["upgrade", "price target", "analyst", "rating", "buy rating", "outperform"],
    "downgrade": ["downgrade", "sell rating", "underperform", "cut rating"],
    "product": ["launch", "release", "announce", "unveil", "new product", "innovation"],
    "deal": ["acquisition", "merger", "deal", "partnership", "contract", "agreement"],
    "fda": ["FDA", "approval", "clinical trial", "drug", "pharma"],
    "legal": ["lawsuit", "SEC", "investigation", "settlement", "fine"],
}


class NewsAnalyzer:
    """Analyzes news for stocks to identify catalysts and sentiment."""

    def __init__(self, api_client=None):
        """Initialize with optional Alpaca client."""
        self._client = api_client
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 300  # 5 minute cache

    def _get_client(self):
        """Lazy load Alpaca client."""
        if self._client is None:
            from utils.alpaca_client import AlpacaClient
            self._client = AlpacaClient()
        return self._client

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache_time:
            return False
        age = (datetime.now() - self._cache_time[key]).total_seconds()
        return age < self._cache_ttl

    def get_news(
        self,
        symbol: str,
        days: int = 1,
        limit: int = 10,
        use_cache: bool = True,
    ) -> list[NewsItem]:
        """Fetch recent news for a symbol.

        Args:
            symbol: Stock ticker
            days: How many days back to look
            limit: Max number of articles
            use_cache: Whether to use cached results

        Returns:
            List of NewsItem objects
        """
        cache_key = f"{symbol}_{days}_{limit}"

        if use_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            client = self._get_client()

            # Calculate date range
            end = datetime.now()
            start = end - timedelta(days=days)

            # Fetch news from Alpaca
            news_list = client.api.get_news(
                symbol=symbol,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                limit=limit,
            )

            items = []
            for article in news_list:
                # Parse timestamp
                if hasattr(article, 'created_at'):
                    timestamp = article.created_at
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.now()

                # Calculate sentiment
                text = f"{article.headline} {getattr(article, 'summary', '')}"
                sentiment_score = self._calculate_sentiment(text)
                sentiment = self._score_to_sentiment(sentiment_score)

                item = NewsItem(
                    headline=article.headline,
                    summary=getattr(article, 'summary', ''),
                    source=getattr(article, 'source', 'Unknown'),
                    url=getattr(article, 'url', ''),
                    timestamp=timestamp,
                    symbols=getattr(article, 'symbols', [symbol]),
                    sentiment_score=sentiment_score,
                    sentiment=sentiment,
                )
                items.append(item)

            # Cache results
            self._cache[cache_key] = items
            self._cache_time[cache_key] = datetime.now()

            return items

        except Exception as e:
            logger.warning("Failed to fetch news for %s: %s", symbol, e)
            return []

    def analyze_stock(
        self,
        symbol: str,
        days: int = 2,
        min_news_for_catalyst: int = 1,
    ) -> NewsAnalysis:
        """Analyze news for a single stock.

        Args:
            symbol: Stock ticker
            days: How many days to look back
            min_news_for_catalyst: Minimum articles to consider it a catalyst

        Returns:
            NewsAnalysis with sentiment and catalyst info
        """
        analysis = NewsAnalysis(symbol=symbol)

        try:
            news_items = self.get_news(symbol, days=days, limit=20)

            if not news_items:
                return analysis

            analysis.has_news = True
            analysis.news_count = len(news_items)
            analysis.news_items = news_items[:5]  # Keep top 5 for reference

            # Calculate average sentiment
            if news_items:
                total_sentiment = sum(n.sentiment_score for n in news_items)
                analysis.sentiment_score = round(total_sentiment / len(news_items), 2)
                analysis.sentiment = self._score_to_sentiment(analysis.sentiment_score)

            # Calculate latest news age
            latest = max(news_items, key=lambda x: x.timestamp)
            age = datetime.now(latest.timestamp.tzinfo) - latest.timestamp
            analysis.latest_news_age_hours = round(age.total_seconds() / 3600, 1)

            # Check for catalyst
            if len(news_items) >= min_news_for_catalyst:
                catalyst_type = self._detect_catalyst_type(news_items)
                if catalyst_type:
                    analysis.has_catalyst = True
                    analysis.catalyst_type = catalyst_type
                # Also consider significant sentiment as catalyst
                elif abs(analysis.sentiment_score) >= 0.3:
                    analysis.has_catalyst = True
                    analysis.catalyst_type = "sentiment"

            return analysis

        except Exception as e:
            analysis.error = str(e)
            logger.error("News analysis failed for %s: %s", symbol, e)
            return analysis

    def analyze_batch(
        self,
        symbols: list[str],
        days: int = 2,
    ) -> dict[str, NewsAnalysis]:
        """Analyze news for multiple stocks.

        Args:
            symbols: List of stock tickers
            days: How many days to look back

        Returns:
            Dict of symbol -> NewsAnalysis
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.analyze_stock(symbol, days=days)
        return results

    def filter_with_news(
        self,
        symbols: list[str],
        require_catalyst: bool = False,
        min_sentiment: float = -1.0,
        max_news_age_hours: float = 48.0,
    ) -> list[tuple[str, NewsAnalysis]]:
        """Filter stocks to only those with relevant news.

        Args:
            symbols: List of stock tickers
            require_catalyst: Only include stocks with identified catalysts
            min_sentiment: Minimum sentiment score (-1 to 1)
            max_news_age_hours: Maximum age of latest news

        Returns:
            List of (symbol, analysis) tuples for stocks with news
        """
        results = []

        for symbol in symbols:
            analysis = self.analyze_stock(symbol)

            if not analysis.has_news:
                continue

            if require_catalyst and not analysis.has_catalyst:
                continue

            if analysis.sentiment_score < min_sentiment:
                continue

            if analysis.latest_news_age_hours > max_news_age_hours:
                continue

            results.append((symbol, analysis))

        # Sort by sentiment score (most bullish first)
        results.sort(key=lambda x: x[1].sentiment_score, reverse=True)

        return results

    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score from text.

        Returns:
            Score from -1 (bearish) to 1 (bullish)
        """
        if not text:
            return 0.0

        text_lower = text.lower()

        bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw.lower() in text_lower)
        bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw.lower() in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        # Score between -1 and 1
        score = (bullish_count - bearish_count) / total
        return round(score, 2)

    def _score_to_sentiment(self, score: float) -> str:
        """Convert numeric score to sentiment label."""
        if score >= 0.2:
            return "bullish"
        elif score <= -0.2:
            return "bearish"
        return "neutral"

    def _detect_catalyst_type(self, news_items: list[NewsItem]) -> str:
        """Detect the type of catalyst from news items."""
        # Combine all headlines and summaries
        all_text = " ".join(
            f"{n.headline} {n.summary}" for n in news_items
        ).lower()

        # Check each catalyst type
        for catalyst_type, keywords in CATALYST_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in all_text:
                    return catalyst_type

        return ""

    @staticmethod
    def format_analysis(analysis: NewsAnalysis) -> str:
        """Format analysis as readable string."""
        if not analysis.has_news:
            return f"  {analysis.symbol}: No recent news"

        sentiment_icon = {
            "bullish": "+",
            "bearish": "-",
            "neutral": "~"
        }.get(analysis.sentiment, "~")

        catalyst_str = f" [{analysis.catalyst_type}]" if analysis.has_catalyst else ""

        return (
            f"  {analysis.symbol}: {analysis.news_count} articles, "
            f"sentiment: {sentiment_icon}{analysis.sentiment_score:+.2f} ({analysis.sentiment})"
            f"{catalyst_str}, age: {analysis.latest_news_age_hours:.0f}h"
        )

    @staticmethod
    def format_results(results: dict[str, NewsAnalysis]) -> str:
        """Format batch results as table."""
        if not results:
            return "No news analysis results."

        lines = [
            f"\n{'=' * 70}",
            f"  NEWS ANALYSIS ({len(results)} stocks)",
            f"{'=' * 70}",
            f"  {'Symbol':8s} {'Articles':>8s} {'Sentiment':>12s} {'Score':>8s} {'Catalyst':>12s} {'Age':>6s}",
            f"  {'-' * 66}",
        ]

        for symbol, a in results.items():
            if a.has_news:
                catalyst = a.catalyst_type[:10] if a.has_catalyst else "-"
                lines.append(
                    f"  {symbol:8s} {a.news_count:>8d} {a.sentiment:>12s} "
                    f"{a.sentiment_score:>+8.2f} {catalyst:>12s} {a.latest_news_age_hours:>5.0f}h"
                )
            else:
                lines.append(f"  {symbol:8s} {'No news':>8s}")

        lines.append(f"{'=' * 70}\n")
        return "\n".join(lines)


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="News Analyzer")
    parser.add_argument("symbols", nargs="+", help="Stock symbols to analyze")
    parser.add_argument("--days", type=int, default=2, help="Days to look back")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    analyzer = NewsAnalyzer()
    results = analyzer.analyze_batch(args.symbols, days=args.days)
    print(NewsAnalyzer.format_results(results))

    # Print detailed news for each
    for symbol, analysis in results.items():
        if analysis.news_items:
            print(f"\n--- {symbol} Headlines ---")
            for item in analysis.news_items[:3]:
                print(f"  [{item.sentiment}] {item.headline[:70]}...")
