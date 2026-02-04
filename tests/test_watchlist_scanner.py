"""Unit tests for the Watchlist Scanner.

Tests cover:
    - Watchlist loading from different sources
    - Scanning logic and result aggregation
    - Signal filtering and sorting
    - CSV export functionality
    - Built-in watchlist availability
"""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from utils.watchlist_scanner import (
    WatchlistScanner,
    ScanResult,
    ScanSummary,
    SP500_TOP50,
    NASDAQ100_TOP50,
    POPULAR_TECH,
    HIGH_DIVIDEND,
    ETFS,
    BUILT_IN_WATCHLISTS,
)


# ── Built-in watchlist tests ────────────────────────────────────────


class TestBuiltInWatchlists(unittest.TestCase):
    """Tests for built-in watchlist definitions."""

    def test_sp500_top50_has_50_symbols(self):
        self.assertEqual(len(SP500_TOP50), 50)

    def test_nasdaq100_top50_has_50_symbols(self):
        self.assertEqual(len(NASDAQ100_TOP50), 50)

    def test_popular_tech_has_symbols(self):
        self.assertGreater(len(POPULAR_TECH), 10)

    def test_high_dividend_has_symbols(self):
        self.assertGreater(len(HIGH_DIVIDEND), 10)

    def test_etfs_has_symbols(self):
        self.assertGreater(len(ETFS), 10)

    def test_built_in_watchlists_dict(self):
        self.assertIn("SP500_TOP50", BUILT_IN_WATCHLISTS)
        self.assertIn("NASDAQ100_TOP50", BUILT_IN_WATCHLISTS)
        self.assertIn("POPULAR_TECH", BUILT_IN_WATCHLISTS)
        self.assertIn("HIGH_DIVIDEND", BUILT_IN_WATCHLISTS)
        self.assertIn("ETFS", BUILT_IN_WATCHLISTS)

    def test_list_watchlists_returns_names(self):
        names = WatchlistScanner.list_watchlists()
        self.assertEqual(len(names), 5)
        self.assertIn("SP500_TOP50", names)


# ── Watchlist loading tests ─────────────────────────────────────────


class TestWatchlistLoading(unittest.TestCase):
    """Tests for loading watchlists from different sources."""

    def setUp(self):
        self.mock_client = MagicMock()
        with patch("utils.watchlist_scanner.TechnicalAnalysisAgent"):
            self.scanner = WatchlistScanner(client=self.mock_client)

    def test_load_builtin_sp500(self):
        symbols = self.scanner.load_watchlist("SP500_TOP50")
        self.assertEqual(len(symbols), 50)
        self.assertIn("AAPL", symbols)

    def test_load_builtin_case_insensitive(self):
        symbols_upper = self.scanner.load_watchlist("SP500_TOP50")
        symbols_lower = self.scanner.load_watchlist("sp500_top50")
        self.assertEqual(symbols_upper, symbols_lower)

    def test_load_comma_separated_symbols(self):
        symbols = self.scanner.load_watchlist("AAPL,MSFT,GOOGL")
        self.assertEqual(symbols, ["AAPL", "MSFT", "GOOGL"])

    def test_load_comma_separated_with_spaces(self):
        symbols = self.scanner.load_watchlist("AAPL, MSFT , GOOGL")
        self.assertEqual(symbols, ["AAPL", "MSFT", "GOOGL"])

    def test_load_single_symbol(self):
        symbols = self.scanner.load_watchlist("AAPL")
        self.assertEqual(symbols, ["AAPL"])

    def test_load_single_symbol_lowercase(self):
        symbols = self.scanner.load_watchlist("aapl")
        self.assertEqual(symbols, ["AAPL"])

    def test_load_csv_file(self):
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("symbol,name\n")
            f.write("AAPL,Apple Inc\n")
            f.write("MSFT,Microsoft Corp\n")
            f.write("GOOGL,Alphabet Inc\n")
            csv_path = f.name

        try:
            symbols = self.scanner.load_watchlist(csv_path)
            self.assertEqual(symbols, ["AAPL", "MSFT", "GOOGL"])
        finally:
            Path(csv_path).unlink()

    def test_load_csv_with_ticker_column(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ticker,company\n")
            f.write("AAPL,Apple\n")
            f.write("TSLA,Tesla\n")
            csv_path = f.name

        try:
            symbols = self.scanner.load_watchlist(csv_path)
            self.assertEqual(symbols, ["AAPL", "TSLA"])
        finally:
            Path(csv_path).unlink()

    def test_load_nonexistent_csv_returns_empty(self):
        symbols = self.scanner.load_watchlist("/nonexistent/file.csv")
        self.assertEqual(symbols, [])


# ── ScanResult dataclass tests ──────────────────────────────────────


class TestScanResult(unittest.TestCase):
    """Tests for ScanResult dataclass."""

    def test_default_values(self):
        result = ScanResult(symbol="AAPL")
        self.assertEqual(result.symbol, "AAPL")
        self.assertEqual(result.signal, "HOLD")
        self.assertEqual(result.score, 0)
        self.assertEqual(result.price, 0.0)
        self.assertEqual(result.atr, 0.0)
        self.assertEqual(result.rsi, 0.0)
        self.assertEqual(result.indicators, [])
        self.assertEqual(result.error, "")

    def test_full_values(self):
        result = ScanResult(
            symbol="AAPL",
            signal="BUY",
            score=3,
            price=150.0,
            atr=3.5,
            rsi=35.0,
            indicators=[{"name": "RSI", "value": 35.0}],
        )
        self.assertEqual(result.signal, "BUY")
        self.assertEqual(result.score, 3)
        self.assertEqual(result.price, 150.0)

    def test_error_result(self):
        result = ScanResult(symbol="BAD", error="API Error")
        self.assertEqual(result.error, "API Error")
        self.assertEqual(result.signal, "HOLD")


# ── ScanSummary dataclass tests ─────────────────────────────────────


class TestScanSummary(unittest.TestCase):
    """Tests for ScanSummary dataclass."""

    def test_default_values(self):
        summary = ScanSummary(
            source="TEST",
            total_scanned=10,
            buy_signals=3,
            sell_signals=2,
            hold_signals=5,
            errors=0,
        )
        self.assertEqual(summary.source, "TEST")
        self.assertEqual(summary.total_scanned, 10)
        self.assertEqual(summary.buy_signals, 3)
        self.assertEqual(summary.sell_signals, 2)
        self.assertEqual(summary.top_buys, [])
        self.assertEqual(summary.top_sells, [])

    def test_with_results(self):
        buy = ScanResult(symbol="AAPL", signal="BUY", score=3)
        sell = ScanResult(symbol="MSFT", signal="SELL", score=-3)

        summary = ScanSummary(
            source="TEST",
            total_scanned=2,
            buy_signals=1,
            sell_signals=1,
            hold_signals=0,
            errors=0,
            top_buys=[buy],
            top_sells=[sell],
        )
        self.assertEqual(len(summary.top_buys), 1)
        self.assertEqual(len(summary.top_sells), 1)
        self.assertEqual(summary.top_buys[0].symbol, "AAPL")


# ── Scanning logic tests ────────────────────────────────────────────


class TestScanning(unittest.TestCase):
    """Tests for the scanning logic."""

    def setUp(self):
        self.mock_client = MagicMock()

    def _make_scanner_with_mock_ta(self, results_map: dict):
        """Create a scanner with mocked TA agent responses."""
        with patch("utils.watchlist_scanner.TechnicalAnalysisAgent") as mock_ta_cls:
            mock_ta = MagicMock()

            def analyze_side_effect(symbol, **kwargs):
                return results_map.get(symbol, {
                    "signal": "HOLD",
                    "composite_score": 0,
                    "current_price": 100.0,
                    "atr": 2.0,
                    "indicators": [],
                })

            mock_ta.analyze.side_effect = analyze_side_effect
            mock_ta_cls.return_value = mock_ta

            scanner = WatchlistScanner(
                client=self.mock_client,
                rate_limit_delay=0,  # No delay for tests
            )
            return scanner

    def test_scan_empty_watchlist(self):
        scanner = self._make_scanner_with_mock_ta({})
        summary = scanner.scan(source="")
        self.assertEqual(summary.total_scanned, 0)

    def test_scan_counts_signals(self):
        results = {
            "AAPL": {"signal": "BUY", "composite_score": 3, "current_price": 150.0, "atr": 3.0, "indicators": []},
            "MSFT": {"signal": "BUY", "composite_score": 2, "current_price": 300.0, "atr": 4.0, "indicators": []},
            "GOOGL": {"signal": "SELL", "composite_score": -2, "current_price": 130.0, "atr": 3.5, "indicators": []},
            "TSLA": {"signal": "HOLD", "composite_score": 0, "current_price": 200.0, "atr": 8.0, "indicators": []},
        }
        scanner = self._make_scanner_with_mock_ta(results)
        summary = scanner.scan(source="AAPL,MSFT,GOOGL,TSLA")

        self.assertEqual(summary.total_scanned, 4)
        self.assertEqual(summary.buy_signals, 2)
        self.assertEqual(summary.sell_signals, 1)
        self.assertEqual(summary.hold_signals, 1)

    def test_scan_sorts_buys_by_score(self):
        results = {
            "AAPL": {"signal": "BUY", "composite_score": 2, "current_price": 150.0, "atr": 3.0, "indicators": []},
            "MSFT": {"signal": "BUY", "composite_score": 4, "current_price": 300.0, "atr": 4.0, "indicators": []},
            "GOOGL": {"signal": "BUY", "composite_score": 3, "current_price": 130.0, "atr": 3.5, "indicators": []},
        }
        scanner = self._make_scanner_with_mock_ta(results)
        summary = scanner.scan(source="AAPL,MSFT,GOOGL")

        self.assertEqual(len(summary.top_buys), 3)
        self.assertEqual(summary.top_buys[0].symbol, "MSFT")  # Highest score
        self.assertEqual(summary.top_buys[1].symbol, "GOOGL")
        self.assertEqual(summary.top_buys[2].symbol, "AAPL")  # Lowest score

    def test_scan_sorts_sells_by_score(self):
        results = {
            "AAPL": {"signal": "SELL", "composite_score": -2, "current_price": 150.0, "atr": 3.0, "indicators": []},
            "MSFT": {"signal": "SELL", "composite_score": -4, "current_price": 300.0, "atr": 4.0, "indicators": []},
            "GOOGL": {"signal": "SELL", "composite_score": -3, "current_price": 130.0, "atr": 3.5, "indicators": []},
        }
        scanner = self._make_scanner_with_mock_ta(results)
        summary = scanner.scan(source="AAPL,MSFT,GOOGL")

        self.assertEqual(len(summary.top_sells), 3)
        self.assertEqual(summary.top_sells[0].symbol, "MSFT")  # Most negative
        self.assertEqual(summary.top_sells[1].symbol, "GOOGL")
        self.assertEqual(summary.top_sells[2].symbol, "AAPL")

    def test_scan_respects_top_n(self):
        results = {
            "AAPL": {"signal": "BUY", "composite_score": 4, "current_price": 150.0, "atr": 3.0, "indicators": []},
            "MSFT": {"signal": "BUY", "composite_score": 3, "current_price": 300.0, "atr": 4.0, "indicators": []},
            "GOOGL": {"signal": "BUY", "composite_score": 2, "current_price": 130.0, "atr": 3.5, "indicators": []},
        }
        scanner = self._make_scanner_with_mock_ta(results)
        summary = scanner.scan(source="AAPL,MSFT,GOOGL", top_n=2)

        self.assertEqual(len(summary.top_buys), 2)
        self.assertEqual(summary.top_buys[0].symbol, "AAPL")
        self.assertEqual(summary.top_buys[1].symbol, "MSFT")

    def test_scan_filter_buy_only(self):
        results = {
            "AAPL": {"signal": "BUY", "composite_score": 3, "current_price": 150.0, "atr": 3.0, "indicators": []},
            "MSFT": {"signal": "SELL", "composite_score": -3, "current_price": 300.0, "atr": 4.0, "indicators": []},
        }
        scanner = self._make_scanner_with_mock_ta(results)
        summary = scanner.scan(source="AAPL,MSFT", signal_filter="buy")

        self.assertEqual(len(summary.top_buys), 1)
        self.assertEqual(len(summary.top_sells), 0)

    def test_scan_filter_sell_only(self):
        results = {
            "AAPL": {"signal": "BUY", "composite_score": 3, "current_price": 150.0, "atr": 3.0, "indicators": []},
            "MSFT": {"signal": "SELL", "composite_score": -3, "current_price": 300.0, "atr": 4.0, "indicators": []},
        }
        scanner = self._make_scanner_with_mock_ta(results)
        summary = scanner.scan(source="AAPL,MSFT", signal_filter="sell")

        self.assertEqual(len(summary.top_buys), 0)
        self.assertEqual(len(summary.top_sells), 1)

    def test_scan_min_score_filters(self):
        results = {
            "AAPL": {"signal": "BUY", "composite_score": 4, "current_price": 150.0, "atr": 3.0, "indicators": []},
            "MSFT": {"signal": "BUY", "composite_score": 2, "current_price": 300.0, "atr": 4.0, "indicators": []},
            "GOOGL": {"signal": "BUY", "composite_score": 1, "current_price": 130.0, "atr": 3.5, "indicators": []},
        }
        scanner = self._make_scanner_with_mock_ta(results)
        summary = scanner.scan(source="AAPL,MSFT,GOOGL", min_score=2)

        self.assertEqual(len(summary.top_buys), 2)  # Only AAPL and MSFT


# ── Formatting tests ────────────────────────────────────────────────


class TestFormatting(unittest.TestCase):
    """Tests for result formatting."""

    def test_format_results_has_header(self):
        summary = ScanSummary(
            source="TEST",
            total_scanned=10,
            buy_signals=3,
            sell_signals=2,
            hold_signals=5,
            errors=0,
            scan_time_seconds=1.5,
        )
        output = WatchlistScanner.format_results(summary)
        self.assertIn("WATCHLIST SCAN", output)
        self.assertIn("TEST", output)
        self.assertIn("10 symbols", output)

    def test_format_results_shows_buys(self):
        buy = ScanResult(symbol="AAPL", signal="BUY", score=3, price=150.0, rsi=35.0)
        summary = ScanSummary(
            source="TEST",
            total_scanned=1,
            buy_signals=1,
            sell_signals=0,
            hold_signals=0,
            errors=0,
            top_buys=[buy],
        )
        output = WatchlistScanner.format_results(summary)
        self.assertIn("TOP BUY CANDIDATES", output)
        self.assertIn("AAPL", output)

    def test_format_results_shows_sells(self):
        sell = ScanResult(symbol="MSFT", signal="SELL", score=-3, price=300.0, rsi=75.0)
        summary = ScanSummary(
            source="TEST",
            total_scanned=1,
            buy_signals=0,
            sell_signals=1,
            hold_signals=0,
            errors=0,
            top_sells=[sell],
        )
        output = WatchlistScanner.format_results(summary)
        self.assertIn("TOP SELL CANDIDATES", output)
        self.assertIn("MSFT", output)

    def test_format_results_no_signals(self):
        summary = ScanSummary(
            source="TEST",
            total_scanned=5,
            buy_signals=0,
            sell_signals=0,
            hold_signals=5,
            errors=0,
        )
        output = WatchlistScanner.format_results(summary)
        self.assertIn("No actionable signals", output)


# ── CSV export tests ────────────────────────────────────────────────


class TestCSVExport(unittest.TestCase):
    """Tests for CSV export functionality."""

    def test_export_csv_creates_file(self):
        buy = ScanResult(symbol="AAPL", signal="BUY", score=3, price=150.0, rsi=35.0, atr=3.5)
        sell = ScanResult(symbol="MSFT", signal="SELL", score=-3, price=300.0, rsi=75.0, atr=4.0)

        summary = ScanSummary(
            source="TEST",
            total_scanned=2,
            buy_signals=1,
            sell_signals=1,
            hold_signals=0,
            errors=0,
            top_buys=[buy],
            top_sells=[sell],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            WatchlistScanner.to_csv(summary, csv_path)
            self.assertTrue(Path(csv_path).exists())

            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["symbol"], "AAPL")
            self.assertEqual(rows[0]["signal"], "BUY")
            self.assertEqual(rows[1]["symbol"], "MSFT")
            self.assertEqual(rows[1]["signal"], "SELL")
        finally:
            Path(csv_path).unlink()

    def test_export_csv_has_headers(self):
        summary = ScanSummary(
            source="TEST",
            total_scanned=0,
            buy_signals=0,
            sell_signals=0,
            hold_signals=0,
            errors=0,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            WatchlistScanner.to_csv(summary, csv_path)

            with open(csv_path) as f:
                header = f.readline().strip()

            self.assertIn("symbol", header)
            self.assertIn("signal", header)
            self.assertIn("score", header)
            self.assertIn("price", header)
            self.assertIn("rsi", header)
            self.assertIn("atr", header)
        finally:
            Path(csv_path).unlink()


if __name__ == "__main__":
    unittest.main()
