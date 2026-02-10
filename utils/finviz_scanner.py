"""Finviz Scanner Integration

Fetches trading opportunities from Finviz stock screener.

Screens available:
    - TOP_GAINERS: Stocks up most today
    - TOP_LOSERS: Stocks down most today
    - UNUSUAL_VOLUME: Abnormal volume spikes
    - NEW_HIGH: 52-week highs
    - NEW_LOW: 52-week lows
    - OVERSOLD: RSI < 30 (potential bounce)
    - OVERBOUGHT: RSI > 70 (potential pullback)
    - BREAKOUT: Price breaking above resistance
    - EARNINGS_TODAY: Reporting earnings today
    - UPGRADES: Recent analyst upgrades
    - DOWNGRADES: Recent analyst downgrades

Usage:
    scanner = FinvizScanner()

    # Get top gainers
    gainers = scanner.get_screen("TOP_GAINERS", limit=20)

    # Get oversold stocks (potential buys)
    oversold = scanner.get_screen("OVERSOLD")

    # Get combined opportunities
    opportunities = scanner.get_opportunities()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FinvizStock:
    """Stock data from Finviz."""
    symbol: str
    company: str = ""
    sector: str = ""
    industry: str = ""
    country: str = ""
    market_cap: str = ""
    price: float = 0.0
    change_pct: float = 0.0
    volume: int = 0
    avg_volume: int = 0
    relative_volume: float = 0.0
    pe: float = 0.0
    eps: float = 0.0
    rsi: float = 0.0
    sma20: float = 0.0
    sma50: float = 0.0
    sma200: float = 0.0
    signal: str = ""  # BUY, SELL, or HOLD
    screen_source: str = ""  # Which screen found this stock


# Finviz filter presets (using valid finvizfinance filter names)
SCREENS = {
    "TOP_GAINERS": {
        "Change": "Up 5%",
        "Relative Volume": "Over 2",
    },
    "TOP_LOSERS": {
        "Change": "Down 5%",
        "Relative Volume": "Over 2",
    },
    "UNUSUAL_VOLUME": {
        "Relative Volume": "Over 3",
        "Average Volume": "Over 500K",
    },
    "NEW_HIGH": {
        "52-Week High/Low": "New High",
    },
    "NEW_LOW": {
        "52-Week High/Low": "New Low",
    },
    "OVERSOLD": {
        "RSI (14)": "Oversold (30)",
    },
    "OVERBOUGHT": {
        "RSI (14)": "Overbought (70)",
    },
    "BREAKOUT": {
        "20-Day High/Low": "New High",
        "Relative Volume": "Over 1.5",
        "Change": "Up",
    },
    "SMA_CROSS_UP": {
        "20-Day Simple Moving Average": "Price crossed SMA20 above",
    },
    "SMA_CROSS_DOWN": {
        "20-Day Simple Moving Average": "Price crossed SMA20 below",
    },
    "GOLDEN_CROSS": {
        "50-Day Simple Moving Average": "Price crossed SMA50 above",
    },
    "DEATH_CROSS": {
        "50-Day Simple Moving Average": "Price crossed SMA50 below",
    },
    "UPGRADES": {
        "Analyst Recom.": "Strong Buy (1)",
        "Change": "Up",
    },
    "DOWNGRADES": {
        "Analyst Recom.": "Sell",
        "Change": "Down",
    },
}


class FinvizScanner:
    """Scans Finviz for trading opportunities."""

    def __init__(self, min_price: float = 5.0, max_price: float = 500.0):
        """Initialize scanner.

        Args:
            min_price: Minimum stock price filter
            max_price: Maximum stock price filter
        """
        self.min_price = min_price
        self.max_price = max_price
        self._finviz = None
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 300  # 5 minute cache

    def _get_finviz(self):
        """Lazy load finvizfinance."""
        if self._finviz is None:
            try:
                from finvizfinance.screener.overview import Overview
                self._finviz = Overview
                logger.info("Finviz scanner initialized")
            except ImportError:
                logger.error("finvizfinance not installed. Run: pip install finvizfinance")
                raise ImportError("Please install finvizfinance: pip install finvizfinance")
        return self._finviz

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache_time:
            return False
        age = (datetime.now() - self._cache_time[key]).total_seconds()
        return age < self._cache_ttl

    def get_screen(
        self,
        screen_name: str,
        limit: int = 20,
        use_cache: bool = True,
    ) -> list[FinvizStock]:
        """Get stocks from a predefined screen.

        Args:
            screen_name: Name of screen (TOP_GAINERS, OVERSOLD, etc.)
            limit: Maximum stocks to return
            use_cache: Whether to use cached results

        Returns:
            List of FinvizStock objects
        """
        cache_key = f"{screen_name}_{limit}"

        if use_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        if screen_name not in SCREENS:
            logger.error("Unknown screen: %s. Available: %s", screen_name, list(SCREENS.keys()))
            return []

        try:
            Overview = self._get_finviz()
            foverview = Overview()

            # Build filters based on screen
            filters = SCREENS[screen_name].copy()

            # Add price filters
            filters["Price"] = f"Over $5"

            # Set filters
            foverview.set_filter(filters_dict=filters)

            # Get data
            df = foverview.screener_view(limit=limit)

            if df is None or df.empty:
                logger.debug("No results for screen: %s", screen_name)
                return []

            # Convert to FinvizStock objects
            stocks = []
            for _, row in df.iterrows():
                try:
                    price = self._parse_float(row.get("Price", 0))
                    if price < self.min_price:
                        continue

                    stock = FinvizStock(
                        symbol=str(row.get("Ticker", "")),
                        company=str(row.get("Company", "")),
                        sector=str(row.get("Sector", "")),
                        industry=str(row.get("Industry", "")),
                        country=str(row.get("Country", "")),
                        market_cap=str(row.get("Market Cap", "")),
                        price=price,
                        change_pct=self._parse_pct(row.get("Change", "0%")),
                        volume=self._parse_int(row.get("Volume", 0)),
                        avg_volume=self._parse_int(row.get("Avg Volume", 0)),
                        pe=self._parse_float(row.get("P/E", 0)),
                        screen_source=screen_name,
                    )

                    # Determine signal based on screen type
                    if screen_name in ["TOP_GAINERS", "NEW_HIGH", "BREAKOUT", "OVERSOLD",
                                       "SMA_CROSS_UP", "GOLDEN_CROSS", "UPGRADES"]:
                        stock.signal = "BUY"
                    elif screen_name in ["TOP_LOSERS", "NEW_LOW", "OVERBOUGHT",
                                         "SMA_CROSS_DOWN", "DEATH_CROSS", "DOWNGRADES"]:
                        stock.signal = "SELL"
                    else:
                        stock.signal = "HOLD"

                    stocks.append(stock)
                except Exception as e:
                    logger.warning("Failed to parse stock: %s", e)
                    continue

            # Cache results
            self._cache[cache_key] = stocks
            self._cache_time[cache_key] = datetime.now()

            logger.info("Finviz %s: found %d stocks", screen_name, len(stocks))
            return stocks

        except Exception as e:
            logger.error("Finviz screen %s failed: %s", screen_name, e)
            return []

    def get_opportunities(
        self,
        screens: list[str] = None,
        limit_per_screen: int = 10,
    ) -> dict[str, list[FinvizStock]]:
        """Get trading opportunities from multiple screens.

        Args:
            screens: List of screen names to run. Default: key momentum screens
            limit_per_screen: Max stocks per screen

        Returns:
            Dict of screen_name -> list of stocks
        """
        if screens is None:
            screens = [
                "TOP_GAINERS",
                "UNUSUAL_VOLUME",
                "OVERSOLD",
                "BREAKOUT",
                "NEW_HIGH",
            ]

        results = {}
        for screen in screens:
            stocks = self.get_screen(screen, limit=limit_per_screen)
            if stocks:
                results[screen] = stocks

        return results

    def get_buy_candidates(self, limit: int = 20) -> list[FinvizStock]:
        """Get best buy candidates from multiple screens.

        Combines oversold stocks, breakouts, and unusual volume.
        """
        candidates = []
        seen = set()

        # Priority screens for buys
        buy_screens = ["OVERSOLD", "BREAKOUT", "UNUSUAL_VOLUME", "GOLDEN_CROSS", "UPGRADES"]

        for screen in buy_screens:
            stocks = self.get_screen(screen, limit=10)
            for stock in stocks:
                if stock.symbol not in seen:
                    stock.signal = "BUY"
                    candidates.append(stock)
                    seen.add(stock.symbol)

                    if len(candidates) >= limit:
                        return candidates

        return candidates

    def get_sell_candidates(self, limit: int = 20) -> list[FinvizStock]:
        """Get best sell/short candidates from multiple screens.

        Combines overbought stocks, death crosses, and downgrades.
        """
        candidates = []
        seen = set()

        # Priority screens for sells
        sell_screens = ["OVERBOUGHT", "DEATH_CROSS", "DOWNGRADES", "TOP_LOSERS"]

        for screen in sell_screens:
            stocks = self.get_screen(screen, limit=10)
            for stock in stocks:
                if stock.symbol not in seen:
                    stock.signal = "SELL"
                    candidates.append(stock)
                    seen.add(stock.symbol)

                    if len(candidates) >= limit:
                        return candidates

        return candidates

    def get_symbols(self, screen_name: str, limit: int = 20) -> list[str]:
        """Get just the symbols from a screen.

        Useful for feeding into other scanners/pipelines.
        """
        stocks = self.get_screen(screen_name, limit=limit)
        return [s.symbol for s in stocks]

    @staticmethod
    def _parse_float(value) -> float:
        """Parse float from various formats."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.replace(",", "").replace("$", "").replace("%", "")
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    def _parse_int(value) -> int:
        """Parse int from various formats."""
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            value = value.replace(",", "").replace("$", "")
            # Handle K, M, B suffixes
            multiplier = 1
            if value.endswith("K"):
                multiplier = 1000
                value = value[:-1]
            elif value.endswith("M"):
                multiplier = 1000000
                value = value[:-1]
            elif value.endswith("B"):
                multiplier = 1000000000
                value = value[:-1]
            try:
                return int(float(value) * multiplier)
            except ValueError:
                return 0
        return 0

    @staticmethod
    def _parse_pct(value) -> float:
        """Parse percentage value."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "")
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    def format_results(stocks: list[FinvizStock]) -> str:
        """Format stocks as a readable table."""
        if not stocks:
            return "No stocks found."

        lines = [
            f"\n{'=' * 80}",
            f"  FINVIZ SCANNER RESULTS ({len(stocks)} stocks)",
            f"{'=' * 80}",
            f"  {'Symbol':8s} {'Signal':6s} {'Price':>10s} {'Change':>8s} {'Volume':>12s} {'Screen':20s}",
            f"  {'-' * 76}",
        ]

        for s in stocks:
            signal_color = "BUY" if s.signal == "BUY" else "SELL" if s.signal == "SELL" else "HOLD"
            lines.append(
                f"  {s.symbol:8s} {signal_color:6s} ${s.price:>9.2f} {s.change_pct:>+7.2f}% "
                f"{s.volume:>12,d} {s.screen_source:20s}"
            )

        lines.append(f"{'=' * 80}\n")
        return "\n".join(lines)


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Finviz Stock Scanner")
    parser.add_argument(
        "--screen",
        default="TOP_GAINERS",
        choices=list(SCREENS.keys()),
        help="Screen to run",
    )
    parser.add_argument("--limit", type=int, default=20, help="Max stocks")
    parser.add_argument("--buys", action="store_true", help="Get buy candidates")
    parser.add_argument("--sells", action="store_true", help="Get sell candidates")
    parser.add_argument("--all", action="store_true", help="Run all screens")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    scanner = FinvizScanner()

    if args.buys:
        stocks = scanner.get_buy_candidates(limit=args.limit)
        print(FinvizScanner.format_results(stocks))
    elif args.sells:
        stocks = scanner.get_sell_candidates(limit=args.limit)
        print(FinvizScanner.format_results(stocks))
    elif args.all:
        opportunities = scanner.get_opportunities()
        for screen_name, stocks in opportunities.items():
            print(f"\n--- {screen_name} ---")
            print(FinvizScanner.format_results(stocks))
    else:
        stocks = scanner.get_screen(args.screen, limit=args.limit)
        print(FinvizScanner.format_results(stocks))
