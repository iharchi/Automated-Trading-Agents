"""Market Scanner — Daily Opportunity Finder

Scans a universe of stocks and identifies the best trading opportunities
based on technical signals, momentum, and regime alignment.

Usage:
    python scanner.py                          # Scan default universe
    python scanner.py --universe SP500         # Scan S&P 500
    python scanner.py --universe VOLATILE      # Scan volatile stocks
    python scanner.py --top 10                 # Show top 10 opportunities
    python scanner.py --min-score 0.15         # Filter by minimum score
    python scanner.py --regime TRENDING_UP     # Filter by market regime

Universes:
    - DEFAULT: Tech + volatile stocks (fast scan)
    - VOLATILE: High-beta, crypto-related stocks
    - SP500_TOP50: Top 50 S&P 500 by market cap
    - CUSTOM: Provide your own list via --symbols
"""

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.alpaca_client import AlpacaClient
from utils.market_regime import MarketRegimeDetector
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.sentiment_analysis_agent import SentimentAnalysisAgent

logger = logging.getLogger(__name__)


# ── Stock Universes ─────────────────────────────────────────────

UNIVERSES = {
    "DEFAULT": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "AMD", "COIN", "MARA", "SMCI", "PLTR", "RIVN", "LCID",
    ],
    "VOLATILE": [
        "NVDA", "AMD", "TSLA", "COIN", "MARA", "SMCI", "PLTR",
        "RIVN", "LCID", "RIOT", "MSTR", "AFRM", "UPST", "SOFI",
        "RBLX", "SNOW", "DKNG", "CRWD", "NET", "ROKU",
    ],
    "MOST_ACTIVE": [
        "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN",
        "META", "GOOGL", "NFLX", "COIN", "MARA", "RIOT", "SOFI",
        "PLTR", "INTC", "BAC", "F", "T", "PFE", "AAL", "NIO",
        "SNAP", "UBER", "HOOD", "RBLX", "DKNG", "LCID", "RIVN",
        "SMCI", "ARM", "MSTR", "CRWD", "NET", "ROKU", "SQ", "PYPL",
        "DIS", "WMT", "JPM", "V", "MA", "XOM", "CVX", "JNJ",
        "UNH", "HD", "PG", "KO",
    ],
    "SP500_TOP50": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "BRK.B", "UNH", "JNJ", "XOM", "JPM", "V", "PG", "MA",
        "HD", "CVX", "MRK", "ABBV", "LLY", "PEP", "KO", "COST",
        "AVGO", "MCD", "WMT", "CSCO", "TMO", "ACN", "ABT",
        "DHR", "CRM", "ADBE", "NKE", "NFLX", "INTC", "AMD",
        "TXN", "PM", "UPS", "RTX", "HON", "QCOM", "LOW",
        "NEE", "UNP", "ORCL", "IBM", "GS", "CAT",
    ],
    "TECH": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "AMD", "INTC", "CRM", "ADBE", "NFLX", "ORCL", "IBM",
        "CSCO", "QCOM", "TXN", "AVGO", "NOW", "SNOW",
    ],
    "CRYPTO_RELATED": [
        "COIN", "MARA", "RIOT", "MSTR", "HUT", "BITF", "CLSK",
        "SI", "SQ", "PYPL", "HOOD",
    ],
}


@dataclass
class ScanResult:
    """Result for one scanned symbol."""
    symbol: str
    signal: str = "HOLD"
    ta_score: int = 0
    sentiment_score: float = 0.0
    combined_score: float = 0.0
    regime: str = ""
    price: float = 0.0
    atr: float = 0.0
    rsi: float = 0.0
    volume_ratio: float = 0.0  # vs 20-day average
    error: str = ""


@dataclass
class ScanSummary:
    """Summary of a full market scan."""
    timestamp: str
    universe: str
    total_scanned: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0
    errors: int = 0
    top_buys: list[ScanResult] = field(default_factory=list)
    top_sells: list[ScanResult] = field(default_factory=list)
    market_regime: str = ""
    scan_duration_sec: float = 0.0


class MarketScanner:
    """Scans a universe of stocks for trading opportunities."""

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        max_workers: int = 5,
    ):
        self.client = client or AlpacaClient()
        self.max_workers = max_workers
        self.ta_agent = TechnicalAnalysisAgent(client=self.client)
        self.sentiment_agent = SentimentAnalysisAgent(client=self.client)
        self.regime_detector = MarketRegimeDetector(client=self.client)

    def _scan_symbol(self, symbol: str) -> ScanResult:
        """Scan a single symbol."""
        result = ScanResult(symbol=symbol)

        try:
            # Technical analysis
            ta = self.ta_agent.analyze(symbol)
            result.signal = ta.get("signal", "HOLD")
            result.ta_score = ta.get("composite_score", 0)
            result.price = ta.get("current_price", 0.0)
            result.atr = ta.get("atr", 0.0)

            # Extract RSI from indicators
            for ind in ta.get("indicators", []):
                if isinstance(ind, dict) and ind.get("name") == "RSI":
                    result.rsi = ind.get("value", 0.0)
                elif hasattr(ind, "name") and ind.name == "RSI":
                    result.rsi = ind.value

            # Sentiment (quick, limited news)
            try:
                sent = self.sentiment_agent.analyze(symbol)
                result.sentiment_score = sent.get("score", 0.0)
            except Exception:
                result.sentiment_score = 0.0

            # Combined score: TA normalized to [-1, 1] + sentiment
            ta_normalized = result.ta_score / 6.0  # Max TA score is 6
            result.combined_score = round(
                (ta_normalized * 0.7) + (result.sentiment_score * 0.3), 4
            )

            # Volume ratio (current vs average)
            try:
                bars = self.client.get_bars(symbol, limit=21)
                if len(bars) >= 21:
                    avg_vol = bars["volume"].iloc[:-1].mean()
                    curr_vol = bars["volume"].iloc[-1]
                    result.volume_ratio = round(curr_vol / avg_vol, 2) if avg_vol > 0 else 1.0
            except Exception:
                result.volume_ratio = 1.0

        except Exception as e:
            result.error = str(e)
            logger.warning("Scan failed for %s: %s", symbol, e)

        return result

    def scan(
        self,
        symbols: list[str],
        *,
        top_n: int = 10,
        min_score: float = 0.0,
        regime_filter: str | None = None,
    ) -> ScanSummary:
        """Scan multiple symbols and return summary with top opportunities."""
        import time
        start = time.time()

        summary = ScanSummary(
            timestamp=datetime.now().isoformat(),
            universe=f"{len(symbols)} symbols",
        )

        # Detect overall market regime (using SPY)
        try:
            regime_result = self.regime_detector.detect("SPY")
            # RegimeResult is a dataclass with .regime attribute (Regime enum)
            if hasattr(regime_result, "regime"):
                regime = regime_result.regime
                summary.market_regime = regime.value if hasattr(regime, "value") else str(regime)
            else:
                summary.market_regime = str(regime_result)
        except Exception as e:
            logger.warning("Could not detect market regime: %s", e)
            summary.market_regime = "UNKNOWN"

        # Scan all symbols in parallel
        results: list[ScanResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._scan_symbol, sym): sym
                for sym in symbols
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # Count signals
        for r in results:
            summary.total_scanned += 1
            if r.error:
                summary.errors += 1
            elif r.signal == "BUY":
                summary.buy_signals += 1
            elif r.signal == "SELL":
                summary.sell_signals += 1
            else:
                summary.hold_signals += 1

        # Filter by minimum score
        filtered = [r for r in results if not r.error and abs(r.combined_score) >= min_score]

        # Filter by regime if specified
        if regime_filter:
            filtered = [r for r in filtered if r.regime == regime_filter]

        # Sort by combined score
        buys = sorted(
            [r for r in filtered if r.signal == "BUY"],
            key=lambda x: x.combined_score,
            reverse=True,
        )[:top_n]

        sells = sorted(
            [r for r in filtered if r.signal == "SELL"],
            key=lambda x: x.combined_score,
        )[:top_n]

        summary.top_buys = buys
        summary.top_sells = sells
        summary.scan_duration_sec = round(time.time() - start, 2)

        return summary

    @staticmethod
    def format_summary(summary: ScanSummary) -> str:
        """Format scan summary for display."""
        lines = [
            "",
            "=" * 70,
            "  MARKET SCANNER — DAILY OPPORTUNITIES",
            "=" * 70,
            f"  Scanned    : {summary.total_scanned} symbols",
            f"  Market     : {summary.market_regime}",
            f"  Duration   : {summary.scan_duration_sec}s",
            f"  BUY signals: {summary.buy_signals}  |  "
            f"SELL signals: {summary.sell_signals}  |  "
            f"HOLD: {summary.hold_signals}",
            "-" * 70,
        ]

        if summary.top_buys:
            lines.append("")
            lines.append("  🟢 TOP BUY OPPORTUNITIES")
            lines.append("  " + "-" * 66)
            lines.append(
                f"  {'Symbol':<8} {'Score':>8} {'TA':>5} {'Sent':>6} "
                f"{'RSI':>6} {'Price':>10} {'Vol':>6}"
            )
            lines.append("  " + "-" * 66)
            for r in summary.top_buys:
                lines.append(
                    f"  {r.symbol:<8} {r.combined_score:>+8.4f} {r.ta_score:>+5d} "
                    f"{r.sentiment_score:>+6.2f} {r.rsi:>6.1f} "
                    f"${r.price:>9.2f} {r.volume_ratio:>5.1f}x"
                )

        if summary.top_sells:
            lines.append("")
            lines.append("  🔴 TOP SELL SIGNALS")
            lines.append("  " + "-" * 66)
            lines.append(
                f"  {'Symbol':<8} {'Score':>8} {'TA':>5} {'Sent':>6} "
                f"{'RSI':>6} {'Price':>10} {'Vol':>6}"
            )
            lines.append("  " + "-" * 66)
            for r in summary.top_sells:
                lines.append(
                    f"  {r.symbol:<8} {r.combined_score:>+8.4f} {r.ta_score:>+5d} "
                    f"{r.sentiment_score:>+6.2f} {r.rsi:>6.1f} "
                    f"${r.price:>9.2f} {r.volume_ratio:>5.1f}x"
                )

        if not summary.top_buys and not summary.top_sells:
            lines.append("")
            lines.append("  No strong signals found. Market may be neutral.")

        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  Scan completed at {summary.timestamp[:19]}")
        lines.append("=" * 70)
        lines.append("")

        return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Market Scanner — Find daily trading opportunities"
    )
    parser.add_argument(
        "--universe",
        choices=list(UNIVERSES.keys()) + ["CUSTOM", "ALL"],
        default="MOST_ACTIVE",
        help="Stock universe to scan (default: MOST_ACTIVE). Use ALL for all tradeable stocks.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=500,
        help="Max symbols to scan when using ALL universe (default: 500)",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Custom symbols to scan (use with --universe CUSTOM)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top opportunities to show (default: 10)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum combined score to include (default: 0.0)",
    )
    parser.add_argument(
        "--regime",
        help="Filter by market regime (e.g., TRENDING_UP)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Parallel workers for scanning (default: 5)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously every minute",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between scans when using --loop (default: 60)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine symbols to scan
    client = AlpacaClient()

    if args.universe == "CUSTOM":
        if not args.symbols:
            print("Error: --symbols required when using --universe CUSTOM")
            return
        symbols = args.symbols
    elif args.universe == "ALL":
        print(f"🔎 Fetching all tradeable assets (max {args.max_symbols})...")
        all_symbols = client.get_tradeable_assets()
        symbols = all_symbols[:args.max_symbols]
        print(f"   Found {len(all_symbols)} assets, scanning {len(symbols)}")
    else:
        symbols = UNIVERSES[args.universe]

    scanner = MarketScanner(client=client, max_workers=args.workers)

    def run_scan():
        print(f"\n🔍 Scanning {len(symbols)} symbols from {args.universe} universe...\n")
        summary = scanner.scan(
            symbols,
            top_n=args.top,
            min_score=args.min_score,
            regime_filter=args.regime,
        )
        print(MarketScanner.format_summary(summary))
        return summary

    if args.loop:
        import time
        print(f"📡 Continuous scanning mode — every {args.interval} seconds")
        print("Press Ctrl+C to stop\n")
        try:
            while True:
                run_scan()
                print(f"⏳ Next scan in {args.interval} seconds...\n")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\n🛑 Scanner stopped.")
    else:
        run_scan()


if __name__ == "__main__":
    main()
