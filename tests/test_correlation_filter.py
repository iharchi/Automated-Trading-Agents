"""Unit tests for the Correlation Filter.

Tests cover:
    - Sector group lookups
    - Correlation calculation
    - Correlation checking
    - Signal filtering
    - Portfolio analysis
    - Result formatting
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from utils.correlation_filter import (
    CorrelationFilter,
    CorrelationResult,
    CorrelationMatrix,
    SECTOR_GROUPS,
    SYMBOL_TO_GROUPS,
)


# ── Sector Group Tests ──────────────────────────────────────────────


class TestSectorGroups(unittest.TestCase):
    """Tests for pre-defined sector group functionality."""

    def test_sector_groups_exist(self):
        self.assertIn("BIG_TECH", SECTOR_GROUPS)
        self.assertIn("SEMICONDUCTORS", SECTOR_GROUPS)
        self.assertIn("BIG_BANKS", SECTOR_GROUPS)
        self.assertIn("ENERGY_MAJOR", SECTOR_GROUPS)

    def test_big_tech_contains_expected_symbols(self):
        big_tech = SECTOR_GROUPS["BIG_TECH"]
        self.assertIn("AAPL", big_tech)
        self.assertIn("MSFT", big_tech)
        self.assertIn("GOOGL", big_tech)
        self.assertIn("AMZN", big_tech)

    def test_symbol_to_groups_mapping(self):
        self.assertIn("AAPL", SYMBOL_TO_GROUPS)
        self.assertIn("BIG_TECH", SYMBOL_TO_GROUPS["AAPL"])

    def test_list_sector_groups(self):
        groups = CorrelationFilter.list_sector_groups()
        self.assertIsInstance(groups, list)
        self.assertIn("BIG_TECH", groups)
        self.assertGreater(len(groups), 10)

    def test_get_group_members(self):
        cf = CorrelationFilter(client=None)
        members = cf.get_group_members("BIG_TECH")
        self.assertIn("AAPL", members)
        self.assertIn("MSFT", members)

    def test_get_group_members_unknown(self):
        cf = CorrelationFilter(client=None)
        members = cf.get_group_members("UNKNOWN_GROUP")
        self.assertEqual(members, [])


# ── Shared Group Tests ──────────────────────────────────────────────


class TestSharedGroups(unittest.TestCase):
    """Tests for finding shared sector groups."""

    def setUp(self):
        self.cf = CorrelationFilter(client=None)

    def test_shared_groups_same_sector(self):
        groups = self.cf.get_shared_groups("AAPL", "MSFT")
        self.assertIn("BIG_TECH", groups)

    def test_shared_groups_different_sectors(self):
        groups = self.cf.get_shared_groups("AAPL", "JPM")
        self.assertEqual(groups, [])

    def test_are_same_sector_true(self):
        self.assertTrue(self.cf.are_same_sector("AAPL", "GOOGL"))
        self.assertTrue(self.cf.are_same_sector("JPM", "BAC"))

    def test_are_same_sector_false(self):
        self.assertFalse(self.cf.are_same_sector("AAPL", "JPM"))
        self.assertFalse(self.cf.are_same_sector("XOM", "MSFT"))

    def test_get_symbol_groups(self):
        groups = self.cf.get_symbol_groups("NVDA")
        self.assertIn("BIG_TECH", groups)
        self.assertIn("SEMICONDUCTORS", groups)


# ── CorrelationResult Tests ─────────────────────────────────────────


class TestCorrelationResult(unittest.TestCase):
    """Tests for CorrelationResult dataclass."""

    def test_not_correlated(self):
        result = CorrelationResult(
            symbol="TEST",
            is_correlated=False,
            reason="No correlation detected",
        )
        self.assertFalse(result.is_correlated)
        self.assertEqual(result.correlated_with, [])

    def test_correlated(self):
        result = CorrelationResult(
            symbol="AAPL",
            is_correlated=True,
            correlated_with=["MSFT", "GOOGL"],
            correlation_values={"MSFT": 0.85, "GOOGL": 0.78},
            shared_groups=["BIG_TECH"],
            max_correlation=0.85,
            reason="Same sector: BIG_TECH",
        )
        self.assertTrue(result.is_correlated)
        self.assertEqual(len(result.correlated_with), 2)
        self.assertEqual(result.max_correlation, 0.85)


# ── CorrelationMatrix Tests ─────────────────────────────────────────


class TestCorrelationMatrix(unittest.TestCase):
    """Tests for CorrelationMatrix dataclass."""

    def setUp(self):
        self.symbols = ["AAPL", "MSFT", "GOOGL"]
        data = {
            "AAPL": [1.0, 0.8, 0.6],
            "MSFT": [0.8, 1.0, 0.7],
            "GOOGL": [0.6, 0.7, 1.0],
        }
        self.matrix = CorrelationMatrix(
            symbols=self.symbols,
            matrix=pd.DataFrame(data, index=self.symbols),
            lookback_days=90,
        )

    def test_get_correlation(self):
        corr = self.matrix.get_correlation("AAPL", "MSFT")
        self.assertEqual(corr, 0.8)

    def test_get_correlation_self(self):
        corr = self.matrix.get_correlation("AAPL", "AAPL")
        self.assertEqual(corr, 1.0)

    def test_get_correlation_unknown_symbol(self):
        corr = self.matrix.get_correlation("AAPL", "UNKNOWN")
        self.assertEqual(corr, 0.0)

    def test_get_high_correlations(self):
        pairs = self.matrix.get_high_correlations(threshold=0.7)
        self.assertEqual(len(pairs), 2)  # AAPL-MSFT (0.8) and MSFT-GOOGL (0.7)

        # First should be highest
        self.assertEqual(pairs[0][0], "AAPL")
        self.assertEqual(pairs[0][1], "MSFT")
        self.assertEqual(pairs[0][2], 0.8)

    def test_get_high_correlations_higher_threshold(self):
        pairs = self.matrix.get_high_correlations(threshold=0.75)
        self.assertEqual(len(pairs), 1)  # Only AAPL-MSFT (0.8)


# ── Correlation Check Tests ─────────────────────────────────────────


class TestCorrelationCheck(unittest.TestCase):
    """Tests for checking correlation of a symbol against positions."""

    def setUp(self):
        self.cf = CorrelationFilter(
            client=None,
            threshold=0.7,
            use_sector_groups=True,
        )

    def test_check_correlation_no_positions(self):
        result = self.cf.check_correlation("AAPL", [])
        self.assertFalse(result.is_correlated)
        self.assertIn("No existing positions", result.reason)

    def test_check_correlation_same_sector(self):
        # AAPL and MSFT are both in BIG_TECH
        result = self.cf.check_correlation("AAPL", ["MSFT"])
        self.assertTrue(result.is_correlated)
        self.assertIn("MSFT", result.correlated_with)
        self.assertIn("BIG_TECH", result.shared_groups)

    def test_check_correlation_different_sector(self):
        # AAPL and JPM are in different sectors
        result = self.cf.check_correlation("AAPL", ["JPM"])
        # May or may not be correlated based on historical data
        # But they should have no shared groups
        self.assertEqual(result.shared_groups, [])

    def test_check_correlation_excludes_self(self):
        result = self.cf.check_correlation("AAPL", ["AAPL", "MSFT"])
        # Should not compare AAPL with itself
        if result.is_correlated:
            # AAPL should only be correlated with MSFT, not itself
            self.assertIn("MSFT", result.correlated_with)


# ── Signal Filtering Tests ──────────────────────────────────────────


class TestSignalFiltering(unittest.TestCase):
    """Tests for filtering correlated signals."""

    def setUp(self):
        self.cf = CorrelationFilter(
            client=None,
            threshold=0.7,
            use_sector_groups=True,
        )

    def test_filter_no_signals(self):
        passed, filtered = self.cf.filter_correlated_signals([], [])
        self.assertEqual(passed, [])
        self.assertEqual(filtered, [])

    def test_filter_no_existing_positions(self):
        signals = [
            {"symbol": "AAPL", "signal": "BUY"},
            {"symbol": "JPM", "signal": "BUY"},
        ]
        passed, filtered = self.cf.filter_correlated_signals(signals, [])
        self.assertEqual(len(passed), 2)
        self.assertEqual(len(filtered), 0)

    def test_filter_correlated_with_existing(self):
        signals = [
            {"symbol": "MSFT", "signal": "BUY"},  # Same sector as AAPL
            {"symbol": "JPM", "signal": "BUY"},   # Different sector
        ]
        passed, filtered = self.cf.filter_correlated_signals(signals, ["AAPL"])

        # JPM should pass, MSFT should be filtered (same sector as AAPL)
        passed_symbols = [s["symbol"] for s in passed]
        filtered_symbols = [s["symbol"] for s in filtered]

        self.assertIn("JPM", passed_symbols)
        self.assertIn("MSFT", filtered_symbols)

    def test_filter_prevents_accumulation(self):
        # Even if no existing positions, shouldn't accumulate correlated signals
        signals = [
            {"symbol": "AAPL", "signal": "BUY"},
            {"symbol": "MSFT", "signal": "BUY"},  # Same sector as AAPL
            {"symbol": "GOOGL", "signal": "BUY"},  # Same sector as AAPL
        ]
        passed, filtered = self.cf.filter_correlated_signals(signals, [])

        # AAPL should pass, then MSFT and GOOGL should be filtered
        passed_symbols = [s["symbol"] for s in passed]
        self.assertEqual(len(passed), 1)
        self.assertIn("AAPL", passed_symbols)
        self.assertEqual(len(filtered), 2)


# ── Portfolio Analysis Tests ────────────────────────────────────────


class TestPortfolioAnalysis(unittest.TestCase):
    """Tests for portfolio correlation analysis."""

    def setUp(self):
        self.cf = CorrelationFilter(client=None, threshold=0.7)

    def test_analyze_single_symbol(self):
        analysis = self.cf.analyze_portfolio_correlation(["AAPL"])
        self.assertEqual(analysis["symbols"], ["AAPL"])
        self.assertEqual(analysis["diversification_score"], 1.0)
        self.assertEqual(analysis["high_correlations"], [])

    def test_analyze_empty_portfolio(self):
        analysis = self.cf.analyze_portfolio_correlation([])
        self.assertEqual(analysis["diversification_score"], 1.0)

    def test_analyze_returns_expected_keys(self):
        analysis = self.cf.analyze_portfolio_correlation(["AAPL", "MSFT"])
        self.assertIn("symbols", analysis)
        self.assertIn("matrix", analysis)
        self.assertIn("high_correlations", analysis)
        self.assertIn("correlation_clusters", analysis)
        self.assertIn("diversification_score", analysis)
        self.assertIn("warnings", analysis)


# ── Formatting Tests ────────────────────────────────────────────────


class TestFormatting(unittest.TestCase):
    """Tests for result formatting."""

    def test_format_correlation_result_not_correlated(self):
        result = CorrelationResult(
            symbol="AAPL",
            is_correlated=False,
            reason="No correlation",
        )
        output = CorrelationFilter.format_correlation_result(result)
        self.assertIn("AAPL", output)
        self.assertIn("NOT CORRELATED", output)

    def test_format_correlation_result_correlated(self):
        result = CorrelationResult(
            symbol="AAPL",
            is_correlated=True,
            correlated_with=["MSFT"],
            correlation_values={"MSFT": 0.85},
            shared_groups=["BIG_TECH"],
            max_correlation=0.85,
            reason="Same sector: BIG_TECH",
        )
        output = CorrelationFilter.format_correlation_result(result)
        self.assertIn("AAPL", output)
        self.assertIn("CORRELATED", output)
        self.assertIn("MSFT", output)
        self.assertIn("0.85", output)

    def test_format_correlation_matrix(self):
        symbols = ["AAPL", "MSFT"]
        data = {
            "AAPL": [1.0, 0.8],
            "MSFT": [0.8, 1.0],
        }
        matrix = CorrelationMatrix(
            symbols=symbols,
            matrix=pd.DataFrame(data, index=symbols),
            lookback_days=90,
        )
        output = CorrelationFilter.format_correlation_matrix(matrix)
        self.assertIn("CORRELATION MATRIX", output)
        self.assertIn("AAPL", output)
        self.assertIn("MSFT", output)

    def test_format_portfolio_analysis(self):
        analysis = {
            "symbols": ["AAPL", "MSFT"],
            "matrix": None,
            "high_correlations": [("AAPL", "MSFT", 0.85)],
            "correlation_clusters": [["AAPL", "MSFT"]],
            "diversification_score": 0.15,
            "average_correlation": 0.85,
            "warnings": ["High average correlation"],
        }
        output = CorrelationFilter.format_portfolio_analysis(analysis)
        self.assertIn("PORTFOLIO CORRELATION ANALYSIS", output)
        self.assertIn("AAPL", output)
        self.assertIn("MSFT", output)
        self.assertIn("Diversification Score", output)


# ── Threshold Tests ─────────────────────────────────────────────────


class TestThreshold(unittest.TestCase):
    """Tests for correlation threshold handling."""

    def test_custom_threshold(self):
        cf = CorrelationFilter(client=None, threshold=0.9)
        self.assertEqual(cf.threshold, 0.9)

    def test_check_with_override_threshold(self):
        cf = CorrelationFilter(client=None, threshold=0.5, use_sector_groups=False)

        # Without sector groups, and no client, correlation can't be calculated
        result = cf.check_correlation("AAPL", ["MSFT"], threshold=0.9)
        # Should not be correlated since we can't calculate actual correlation
        # and sector groups are disabled
        self.assertFalse(result.is_correlated)


# ── Caching Tests ───────────────────────────────────────────────────


class TestCaching(unittest.TestCase):
    """Tests for correlation matrix caching."""

    def test_cache_key_generation(self):
        # Symbols should be sorted for consistent cache keys
        cf = CorrelationFilter(client=None)

        # Since no client, matrices will be identity matrices
        matrix1 = cf.get_correlation_matrix(["AAPL", "MSFT"])
        matrix2 = cf.get_correlation_matrix(["MSFT", "AAPL"])

        # Should produce same symbols list (sorted)
        self.assertEqual(matrix1.symbols, matrix2.symbols)


# ── Integration with Mocked Client ──────────────────────────────────


class TestWithMockedClient(unittest.TestCase):
    """Tests with mocked Alpaca client."""

    def _create_mock_bars(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Create mock price bars with realistic returns."""
        np.random.seed(hash(symbol) % 2**32)  # Deterministic based on symbol
        dates = pd.date_range(end="2024-01-01", periods=days, freq="D")
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, days))
        return pd.DataFrame(
            {"close": prices},
            index=dates,
        )

    def test_calculate_correlation_with_data(self):
        mock_client = MagicMock()

        def get_bars_side_effect(symbol, **kwargs):
            return self._create_mock_bars(symbol)

        mock_client.get_bars.side_effect = get_bars_side_effect

        cf = CorrelationFilter(client=mock_client, use_sector_groups=False)
        corr = cf.calculate_correlation("AAPL", "MSFT")

        # Should return a correlation value between -1 and 1
        self.assertGreaterEqual(corr, -1.0)
        self.assertLessEqual(corr, 1.0)

    def test_get_correlation_matrix_with_data(self):
        mock_client = MagicMock()

        def get_bars_side_effect(symbol, **kwargs):
            return self._create_mock_bars(symbol)

        mock_client.get_bars.side_effect = get_bars_side_effect

        cf = CorrelationFilter(client=mock_client)
        matrix = cf.get_correlation_matrix(["AAPL", "MSFT", "GOOGL"])

        self.assertEqual(len(matrix.symbols), 3)
        self.assertEqual(matrix.get_correlation("AAPL", "AAPL"), 1.0)


if __name__ == "__main__":
    unittest.main()
