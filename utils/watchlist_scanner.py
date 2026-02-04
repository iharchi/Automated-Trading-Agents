"""Watchlist Scanner

Scans a universe of stocks and surfaces the top candidates by signal strength.

Watchlist sources:
    - Built-in lists: SP500_TOP50, NASDAQ100_TOP50, POPULAR_TECH, etc.
    - Custom CSV file with a 'symbol' column
    - Alpaca tradeable assets (filtered by exchange, status)

Usage:
    scanner = WatchlistScanner()
    results = scanner.scan(source="SP500_TOP50", top_n=10)
    print(WatchlistScanner.format_results(results))
"""

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agents.technical_analysis_agent import TechnicalAnalysisAgent
from utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

SignalFilter = Literal["all", "buy", "sell"]

# ── Built-in watchlists ──────────────────────────────────────────

# Top 50 S&P 500 holdings by market cap (approximate)
SP500_TOP50 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
    "JPM", "V", "PG", "XOM", "HD", "MA", "CVX", "MRK", "ABBV", "LLY",
    "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO", "ABT", "DHR",
    "ACN", "CRM", "VZ", "ADBE", "NKE", "CMCSA", "PFE", "TXN", "PM", "INTC",
    "WFC", "NEE", "DIS", "AMD", "RTX", "QCOM", "UPS", "BMY", "COP", "HON",
]

# Top 50 NASDAQ 100 holdings
NASDAQ100_TOP50 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "PEP",
    "COST", "CSCO", "ADBE", "CMCSA", "NFLX", "AMD", "INTC", "TXN", "QCOM", "AMGN",
    "INTU", "TMUS", "HON", "ISRG", "SBUX", "AMAT", "ADP", "BKNG", "MDLZ", "GILD",
    "ADI", "VRTX", "LRCX", "PYPL", "REGN", "CSX", "MU", "PANW", "SNPS", "KLAC",
    "MNST", "CDNS", "ASML", "MELI", "ORLY", "FTNT", "KDP", "CTAS", "MAR", "ABNB",
]

# Popular tech/growth stocks
POPULAR_TECH = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CRM",
    "ADBE", "NFLX", "PYPL", "SQ", "SHOP", "SNOW", "PLTR", "COIN", "UBER", "ABNB",
    "RBLX", "DDOG", "ZS", "CRWD", "NET", "TWLO", "OKTA", "MDB", "DOCU", "ZM",
]

# High dividend stocks
HIGH_DIVIDEND = [
    "VZ", "T", "MO", "PM", "XOM", "CVX", "IBM", "ABBV", "PFE", "KO",
    "PEP", "JNJ", "PG", "MMM", "CAT", "DE", "UPS", "HD", "LOW", "TGT",
]

# ETFs for market sentiment
ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VGT", "XLK", "XLF", "XLE",
    "XLV", "XLP", "XLY", "XLI", "XLB", "XLU", "XLRE", "GLD", "SLV", "TLT",
]

BUILT_IN_WATCHLISTS = {
    "SP500_TOP50": SP500_TOP50,
    "NASDAQ100_TOP50": NASDAQ100_TOP50,
    "POPULAR_TECH": POPULAR_TECH,
    "HIGH_DIVIDEND": HIGH_DIVIDEND,
    "ETFS": ETFS,
}


@dataclass
class ScanResult:
    """Result of scanning a single symbol."""

    symbol: str
    signal: str = "HOLD"
    score: int = 0
    price: float = 0.0
    atr: float = 0.0
    rsi: float = 0.0
    indicators: list = field(default_factory=list)
    error: str = ""


@dataclass
class ScanSummary:
    """Summary of a watchlist scan."""

    source: str
    total_scanned: int
    buy_signals: int
    sell_signals: int
    hold_signals: int
    errors: int
    top_buys: list[ScanResult] = field(default_factory=list)
    top_sells: list[ScanResult] = field(default_factory=list)
    scan_time_seconds: float = 0.0


class WatchlistScanner:
    """Scans watchlists for trading opportunities."""

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        timeframe: str = "1Day",
        rate_limit_delay: float = 0.1,  # seconds between API calls
    ):
        self.client = client or AlpacaClient()
        self.timeframe = timeframe
        self.rate_limit_delay = rate_limit_delay
        self.ta_agent = TechnicalAnalysisAgent(client=self.client, auto_trade=False)

    # ── Watchlist loading ─────────────────────────────────────

    def _load_from_csv(self, path: str) -> list[str]:
        """Load symbols from a CSV file with a 'symbol' column."""
        filepath = Path(path)
        if not filepath.exists():
            logger.error("CSV file not found: %s", path)
            return []

        symbols = []
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try common column names
                symbol = row.get("symbol") or row.get("Symbol") or row.get("ticker") or row.get("Ticker")
                if symbol:
                    symbols.append(symbol.strip().upper())

        logger.info("Loaded %d symbols from %s", len(symbols), path)
        return symbols

    def _load_from_alpaca(
        self,
        exchange: str | None = None,
        min_price: float = 5.0,
        max_price: float = 500.0,
        limit: int = 100,
    ) -> list[str]:
        """Load tradeable symbols from Alpaca assets API."""
        try:
            assets = self.client.api.list_assets(status="active")
            symbols = []

            for asset in assets:
                # Filter by tradeability and exchange
                if not asset.tradable:
                    continue
                if exchange and asset.exchange != exchange:
                    continue
                # Only US stocks
                if asset.asset_class != "us_equity":
                    continue
                # Skip OTC and other non-standard assets
                if asset.exchange in ("OTC", ""):
                    continue

                symbols.append(asset.symbol)

                if len(symbols) >= limit:
                    break

            logger.info("Loaded %d symbols from Alpaca", len(symbols))
            return symbols
        except Exception as e:
            logger.error("Failed to load Alpaca assets: %s", e)
            return []

    def load_watchlist(self, source: str) -> list[str]:
        """Load symbols from the specified source.

        Args:
            source: One of:
                - Built-in list name (e.g., "SP500_TOP50")
                - Path to CSV file (e.g., "/path/to/watchlist.csv")
                - "alpaca" to load from Alpaca assets API
                - Comma-separated symbols (e.g., "AAPL,MSFT,GOOGL")
        """
        # Handle empty source
        if not source or not source.strip():
            logger.warning("Empty watchlist source provided")
            return []

        # Check built-in lists
        if source.upper() in BUILT_IN_WATCHLISTS:
            symbols = BUILT_IN_WATCHLISTS[source.upper()]
            logger.info("Using built-in watchlist %s (%d symbols)", source, len(symbols))
            return symbols

        # Check if it's a file path
        if source.endswith(".csv") or Path(source).exists():
            return self._load_from_csv(source)

        # Check if it's the Alpaca source
        if source.lower() == "alpaca":
            return self._load_from_alpaca()

        # Assume comma-separated symbols
        if "," in source:
            symbols = [s.strip().upper() for s in source.split(",")]
            logger.info("Parsed %d symbols from input", len(symbols))
            return symbols

        # Single symbol
        return [source.upper()]

    # ── Scanning ──────────────────────────────────────────────

    def _scan_symbol(self, symbol: str) -> ScanResult:
        """Scan a single symbol and return the result."""
        try:
            analysis = self.ta_agent.analyze(symbol, timeframe=self.timeframe)

            # Extract RSI from indicators
            rsi = 50.0
            for ind in analysis.get("indicators", []):
                if isinstance(ind, dict):
                    if ind.get("name") == "RSI":
                        rsi = ind.get("value", 50.0)
                else:
                    if ind.name == "RSI":
                        rsi = ind.value

            return ScanResult(
                symbol=symbol,
                signal=analysis.get("signal", "HOLD"),
                score=analysis.get("composite_score", 0),
                price=analysis.get("current_price", 0),
                atr=analysis.get("atr", 0),
                rsi=rsi,
                indicators=analysis.get("indicators", []),
            )
        except Exception as e:
            logger.warning("Error scanning %s: %s", symbol, e)
            return ScanResult(symbol=symbol, error=str(e))

    def scan(
        self,
        source: str = "SP500_TOP50",
        *,
        top_n: int = 10,
        signal_filter: SignalFilter = "all",
        min_score: int = 0,
    ) -> ScanSummary:
        """Scan a watchlist and return the top candidates.

        Args:
            source: Watchlist source (built-in name, CSV path, or symbols)
            top_n: Number of top candidates to return
            signal_filter: "all", "buy", or "sell"
            min_score: Minimum absolute score to include

        Returns:
            ScanSummary with top buys and sells
        """
        start_time = time.time()
        symbols = self.load_watchlist(source)

        if not symbols:
            return ScanSummary(
                source=source,
                total_scanned=0,
                buy_signals=0,
                sell_signals=0,
                hold_signals=0,
                errors=0,
            )

        results: list[ScanResult] = []
        errors = 0

        for i, symbol in enumerate(symbols):
            logger.debug("Scanning %s (%d/%d)", symbol, i + 1, len(symbols))
            result = self._scan_symbol(symbol)
            results.append(result)

            if result.error:
                errors += 1

            # Rate limiting
            if self.rate_limit_delay > 0 and i < len(symbols) - 1:
                time.sleep(self.rate_limit_delay)

        # Count signals
        buy_signals = sum(1 for r in results if r.signal == "BUY")
        sell_signals = sum(1 for r in results if r.signal == "SELL")
        hold_signals = sum(1 for r in results if r.signal == "HOLD")

        # Filter and sort
        buys = [r for r in results if r.signal == "BUY" and abs(r.score) >= min_score]
        sells = [r for r in results if r.signal == "SELL" and abs(r.score) >= min_score]

        # Sort by score (descending for buys, ascending for sells)
        buys.sort(key=lambda r: r.score, reverse=True)
        sells.sort(key=lambda r: r.score)  # Most negative first

        # Determine what to return based on filter
        top_buys = buys[:top_n] if signal_filter in ("all", "buy") else []
        top_sells = sells[:top_n] if signal_filter in ("all", "sell") else []

        scan_time = time.time() - start_time

        return ScanSummary(
            source=source,
            total_scanned=len(symbols),
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            hold_signals=hold_signals,
            errors=errors,
            top_buys=top_buys,
            top_sells=top_sells,
            scan_time_seconds=round(scan_time, 2),
        )

    # ── Formatting ────────────────────────────────────────────

    @staticmethod
    def format_results(summary: ScanSummary) -> str:
        """Format scan results as a readable table."""
        lines = [
            f"\n{'=' * 72}",
            f"  WATCHLIST SCAN — {summary.source}",
            f"{'=' * 72}",
            f"  Scanned    : {summary.total_scanned} symbols in {summary.scan_time_seconds}s",
            f"  BUY signals: {summary.buy_signals}",
            f"  SELL signals: {summary.sell_signals}",
            f"  HOLD signals: {summary.hold_signals}",
            f"  Errors     : {summary.errors}",
        ]

        if summary.top_buys:
            lines.append(f"\n{'─' * 72}")
            lines.append("  TOP BUY CANDIDATES")
            lines.append(f"{'─' * 72}")
            lines.append(f"  {'Rank':<5} {'Symbol':<8} {'Signal':<6} {'Score':>6} {'Price':>10} {'RSI':>6}")
            lines.append(f"  {'-' * 5} {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 10} {'-' * 6}")

            for i, r in enumerate(summary.top_buys, 1):
                lines.append(
                    f"  {i:<5} {r.symbol:<8} {r.signal:<6} {r.score:>+6} ${r.price:>9.2f} {r.rsi:>6.1f}"
                )

        if summary.top_sells:
            lines.append(f"\n{'─' * 72}")
            lines.append("  TOP SELL CANDIDATES")
            lines.append(f"{'─' * 72}")
            lines.append(f"  {'Rank':<5} {'Symbol':<8} {'Signal':<6} {'Score':>6} {'Price':>10} {'RSI':>6}")
            lines.append(f"  {'-' * 5} {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 10} {'-' * 6}")

            for i, r in enumerate(summary.top_sells, 1):
                lines.append(
                    f"  {i:<5} {r.symbol:<8} {r.signal:<6} {r.score:>+6} ${r.price:>9.2f} {r.rsi:>6.1f}"
                )

        if not summary.top_buys and not summary.top_sells:
            lines.append(f"\n  No actionable signals found.")

        lines.append(f"\n{'=' * 72}\n")
        return "\n".join(lines)

    @staticmethod
    def to_csv(summary: ScanSummary, path: str) -> None:
        """Export scan results to CSV."""
        all_results = summary.top_buys + summary.top_sells

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "signal", "score", "price", "rsi", "atr"])
            for r in all_results:
                writer.writerow([r.symbol, r.signal, r.score, r.price, r.rsi, r.atr])

        logger.info("Exported %d results to %s", len(all_results), path)

    @staticmethod
    def list_watchlists() -> list[str]:
        """Return list of available built-in watchlists."""
        return list(BUILT_IN_WATCHLISTS.keys())
