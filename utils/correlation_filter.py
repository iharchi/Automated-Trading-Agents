"""Correlation Filter

Prevents portfolio overexposure by detecting and filtering highly correlated positions.
Uses historical price returns to calculate correlation coefficients between assets.

Features:
    - Dynamic correlation calculation from historical data
    - Pre-defined sector/industry groups for quick checks
    - Configurable correlation thresholds
    - Portfolio-wide correlation analysis
    - Caching to avoid redundant API calls

Usage:
    filter = CorrelationFilter(client)

    # Check if adding GOOGL would be correlated with existing portfolio
    is_correlated = filter.check_correlation("GOOGL", ["AAPL", "META", "MSFT"])

    # Get correlation matrix for a set of symbols
    matrix = filter.get_correlation_matrix(["AAPL", "MSFT", "GOOGL", "META"])

    # Filter out correlated signals from scan results
    filtered = filter.filter_correlated_signals(signals, existing_positions)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


# ── Pre-defined Sector Groups ───────────────────────────────────────
# Stocks within the same group are assumed to be correlated

SECTOR_GROUPS = {
    "BIG_TECH": ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA"],
    "SEMICONDUCTORS": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", "AMAT"],
    "CLOUD_SAAS": ["CRM", "NOW", "ADBE", "ORCL", "IBM", "SNOW", "PLTR", "NET"],
    "STREAMING": ["NFLX", "DIS", "WBD", "PARA", "CMCSA"],
    "SOCIAL_MEDIA": ["META", "SNAP", "PINS", "TWTR", "RDDT"],
    "E_COMMERCE": ["AMZN", "SHOP", "EBAY", "ETSY", "MELI", "SE", "JD", "BABA"],
    "FINTECH": ["SQ", "PYPL", "COIN", "SOFI", "AFRM", "HOOD"],
    "BIG_BANKS": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC"],
    "REGIONAL_BANKS": ["SCHW", "TFC", "FITB", "KEY", "CFG", "HBAN", "RF"],
    "INSURANCE": ["BRK.B", "AIG", "MET", "PRU", "AFL", "TRV", "ALL", "PGR"],
    "ENERGY_MAJOR": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX"],
    "RENEWABLE": ["NEE", "ENPH", "SEDG", "FSLR", "RUN", "PLUG", "BE"],
    "PHARMA": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "AMGN", "GILD"],
    "BIOTECH": ["MRNA", "BNTX", "REGN", "VRTX", "BIIB", "SGEN", "ILMN"],
    "HEALTHCARE": ["UNH", "CVS", "CI", "HUM", "ELV", "MCK", "CAH"],
    "RETAIL": ["WMT", "TGT", "COST", "HD", "LOW", "DG", "DLTR"],
    "CONSUMER": ["PG", "KO", "PEP", "PM", "MO", "CL", "KMB", "GIS"],
    "AUTO": ["TSLA", "F", "GM", "RIVN", "LCID", "NIO", "TM", "HMC"],
    "AIRLINES": ["DAL", "UAL", "AAL", "LUV", "JBLU", "ALK"],
    "DEFENSE": ["LMT", "RTX", "NOC", "BA", "GD", "HII", "LHX"],
    "REITS": ["AMT", "PLD", "CCI", "EQIX", "SPG", "O", "WELL", "AVB"],
    "CRYPTO_RELATED": ["COIN", "MARA", "RIOT", "MSTR", "SQ", "PYPL"],
}

# Inverse mapping: symbol -> list of groups it belongs to
SYMBOL_TO_GROUPS: dict[str, list[str]] = {}
for group_name, symbols in SECTOR_GROUPS.items():
    for symbol in symbols:
        SYMBOL_TO_GROUPS.setdefault(symbol, []).append(group_name)


@dataclass
class CorrelationResult:
    """Result of a correlation check."""

    symbol: str
    is_correlated: bool
    correlated_with: list[str] = field(default_factory=list)
    correlation_values: dict[str, float] = field(default_factory=dict)
    shared_groups: list[str] = field(default_factory=list)
    max_correlation: float = 0.0
    reason: str = ""


@dataclass
class CorrelationMatrix:
    """Correlation matrix for a set of symbols."""

    symbols: list[str]
    matrix: pd.DataFrame
    calculated_at: str = ""
    lookback_days: int = 0

    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        if symbol1 not in self.symbols or symbol2 not in self.symbols:
            return 0.0
        return float(self.matrix.loc[symbol1, symbol2])

    def get_high_correlations(self, threshold: float = 0.7) -> list[tuple[str, str, float]]:
        """Get all pairs with correlation above threshold."""
        pairs = []
        n = len(self.symbols)
        for i in range(n):
            for j in range(i + 1, n):
                corr = float(self.matrix.iloc[i, j])
                if abs(corr) >= threshold:
                    pairs.append((self.symbols[i], self.symbols[j], corr))
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)


class CorrelationFilter:
    """Filters out correlated positions to prevent portfolio overexposure."""

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        threshold: float = 0.7,
        lookback_days: int = 90,
        use_sector_groups: bool = True,
        cache_ttl_minutes: int = 60,
    ):
        """Initialize the correlation filter.

        Args:
            client: AlpacaClient for fetching price data (None = sector-only mode)
            threshold: Correlation coefficient threshold (0.0-1.0)
            lookback_days: Days of historical data for correlation calculation
            use_sector_groups: Use pre-defined sector groups for quick checks
            cache_ttl_minutes: How long to cache correlation calculations
        """
        self.client = client  # Can be None for sector-only mode
        self.threshold = threshold
        self.lookback_days = lookback_days
        self.use_sector_groups = use_sector_groups
        self.cache_ttl_minutes = cache_ttl_minutes

        # Cache for correlation matrices
        self._cache: dict[str, tuple[datetime, CorrelationMatrix]] = {}

    # ── Sector Group Checks ─────────────────────────────────────────

    def get_shared_groups(self, symbol1: str, symbol2: str) -> list[str]:
        """Get sector groups shared by two symbols."""
        groups1 = set(SYMBOL_TO_GROUPS.get(symbol1.upper(), []))
        groups2 = set(SYMBOL_TO_GROUPS.get(symbol2.upper(), []))
        return list(groups1 & groups2)

    def get_symbol_groups(self, symbol: str) -> list[str]:
        """Get all sector groups a symbol belongs to."""
        return SYMBOL_TO_GROUPS.get(symbol.upper(), [])

    def are_same_sector(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are in the same sector group."""
        return len(self.get_shared_groups(symbol1, symbol2)) > 0

    def get_group_members(self, group_name: str) -> list[str]:
        """Get all symbols in a sector group."""
        return SECTOR_GROUPS.get(group_name.upper(), [])

    @staticmethod
    def list_sector_groups() -> list[str]:
        """List all available sector groups."""
        return list(SECTOR_GROUPS.keys())

    # ── Correlation Calculation ─────────────────────────────────────

    def _get_returns(self, symbols: list[str], days: int | None = None) -> pd.DataFrame:
        """Fetch historical returns for symbols.

        Returns:
            DataFrame with daily returns, columns = symbols
        """
        days = days or self.lookback_days
        returns_data = {}

        for symbol in symbols:
            try:
                bars = self.client.get_bars(
                    symbol,
                    timeframe="1Day",
                    limit=days + 10,  # Extra buffer for missing days
                )
                if bars.empty or len(bars) < 20:
                    logger.warning("Insufficient data for %s", symbol)
                    continue

                # Calculate daily returns
                close = bars["close"]
                returns = close.pct_change().dropna()
                returns_data[symbol] = returns

            except Exception as e:
                logger.warning("Failed to get data for %s: %s", symbol, e)
                continue

        if not returns_data:
            return pd.DataFrame()

        # Align all returns to common dates
        df = pd.DataFrame(returns_data)
        df = df.dropna()

        return df

    def calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols.

        Returns:
            Pearson correlation coefficient (-1.0 to 1.0)
        """
        returns = self._get_returns([symbol1, symbol2])

        if returns.empty or len(returns.columns) < 2:
            # Fall back to sector check
            if self.use_sector_groups and self.are_same_sector(symbol1, symbol2):
                return 0.8  # Assumed high correlation for same sector
            return 0.0

        if len(returns) < 20:
            logger.warning("Insufficient overlapping data for correlation")
            return 0.0

        corr = returns[symbol1].corr(returns[symbol2])
        return float(corr) if not np.isnan(corr) else 0.0

    def get_correlation_matrix(
        self, symbols: list[str], *, use_cache: bool = True
    ) -> CorrelationMatrix:
        """Calculate correlation matrix for a set of symbols.

        Args:
            symbols: List of ticker symbols
            use_cache: Whether to use cached results

        Returns:
            CorrelationMatrix object
        """
        symbols = sorted(set(s.upper() for s in symbols))
        cache_key = ",".join(symbols)

        # Check cache
        if use_cache and cache_key in self._cache:
            cached_time, cached_matrix = self._cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=self.cache_ttl_minutes):
                return cached_matrix

        # Calculate returns
        returns = self._get_returns(symbols)

        if returns.empty:
            # Create empty matrix
            matrix = pd.DataFrame(
                np.eye(len(symbols)),
                index=symbols,
                columns=symbols,
            )
        else:
            # Calculate correlation matrix
            available = [s for s in symbols if s in returns.columns]
            matrix = returns[available].corr()

            # Add missing symbols with NaN
            for symbol in symbols:
                if symbol not in matrix.columns:
                    matrix[symbol] = np.nan
                    matrix.loc[symbol] = np.nan
                    matrix.loc[symbol, symbol] = 1.0

            # Reorder to match input
            matrix = matrix.reindex(index=symbols, columns=symbols)

        result = CorrelationMatrix(
            symbols=symbols,
            matrix=matrix,
            calculated_at=datetime.utcnow().isoformat(),
            lookback_days=self.lookback_days,
        )

        # Cache result
        self._cache[cache_key] = (datetime.now(), result)

        return result

    # ── Correlation Checks ──────────────────────────────────────────

    def check_correlation(
        self,
        symbol: str,
        existing_positions: list[str],
        *,
        threshold: float | None = None,
    ) -> CorrelationResult:
        """Check if a symbol is correlated with existing positions.

        Args:
            symbol: Symbol to check
            existing_positions: List of existing position symbols
            threshold: Override correlation threshold

        Returns:
            CorrelationResult with details
        """
        symbol = symbol.upper()
        existing = [s.upper() for s in existing_positions if s.upper() != symbol]
        threshold = threshold if threshold is not None else self.threshold

        if not existing:
            return CorrelationResult(
                symbol=symbol,
                is_correlated=False,
                reason="No existing positions to compare",
            )

        correlated_with = []
        correlation_values = {}
        shared_groups: list[str] = []
        max_corr = 0.0

        # First, quick sector group check
        if self.use_sector_groups:
            for pos in existing:
                groups = self.get_shared_groups(symbol, pos)
                if groups:
                    shared_groups.extend(groups)
                    if pos not in correlated_with:
                        correlated_with.append(pos)
                        correlation_values[pos] = 0.8  # Assumed

        # Then, calculate actual correlations
        all_symbols = [symbol] + existing
        try:
            matrix = self.get_correlation_matrix(all_symbols)

            for pos in existing:
                corr = matrix.get_correlation(symbol, pos)
                if not np.isnan(corr):
                    correlation_values[pos] = corr
                    if abs(corr) >= threshold:
                        if pos not in correlated_with:
                            correlated_with.append(pos)
                        max_corr = max(max_corr, abs(corr))

        except Exception as e:
            logger.warning("Failed to calculate correlations: %s", e)

        # Update max correlation
        if correlation_values:
            max_corr = max(abs(v) for v in correlation_values.values())

        is_correlated = len(correlated_with) > 0

        reason = ""
        if is_correlated:
            if shared_groups:
                reason = f"Same sector: {', '.join(set(shared_groups))}"
            else:
                reason = f"High correlation ({max_corr:.2f}) with {', '.join(correlated_with)}"

        return CorrelationResult(
            symbol=symbol,
            is_correlated=is_correlated,
            correlated_with=correlated_with,
            correlation_values=correlation_values,
            shared_groups=list(set(shared_groups)),
            max_correlation=max_corr,
            reason=reason,
        )

    def filter_correlated_signals(
        self,
        signals: list[dict],
        existing_positions: list[str],
        *,
        symbol_key: str = "symbol",
    ) -> tuple[list[dict], list[dict]]:
        """Filter out correlated signals from a list.

        Args:
            signals: List of signal dicts
            existing_positions: Existing portfolio symbols
            symbol_key: Key in signal dict for symbol

        Returns:
            Tuple of (passed_signals, filtered_signals)
        """
        passed = []
        filtered = []
        checked_symbols = set(s.upper() for s in existing_positions)

        for signal in signals:
            symbol = signal.get(symbol_key, "").upper()
            if not symbol:
                continue

            # Check against existing positions AND already-passed signals
            result = self.check_correlation(symbol, list(checked_symbols))

            if result.is_correlated:
                signal["_filtered_reason"] = result.reason
                signal["_correlated_with"] = result.correlated_with
                filtered.append(signal)
            else:
                passed.append(signal)
                checked_symbols.add(symbol)

        return passed, filtered

    # ── Portfolio Analysis ──────────────────────────────────────────

    def analyze_portfolio_correlation(
        self, symbols: list[str]
    ) -> dict:
        """Analyze correlation within a portfolio.

        Returns:
            Dict with correlation analysis results
        """
        symbols = [s.upper() for s in symbols]

        if len(symbols) < 2:
            return {
                "symbols": symbols,
                "matrix": None,
                "high_correlations": [],
                "correlation_clusters": [],
                "diversification_score": 1.0,
                "warnings": [],
            }

        matrix = self.get_correlation_matrix(symbols)
        high_corrs = matrix.get_high_correlations(self.threshold)

        # Find correlation clusters
        clusters = self._find_clusters(symbols, matrix)

        # Calculate diversification score (lower avg correlation = better)
        avg_corr = self._calculate_avg_correlation(matrix)
        div_score = max(0, 1 - avg_corr)

        # Generate warnings
        warnings = []
        if high_corrs:
            warnings.append(
                f"{len(high_corrs)} highly correlated pairs detected"
            )
        if avg_corr > 0.5:
            warnings.append(
                f"High average correlation ({avg_corr:.2f}) - portfolio may lack diversification"
            )

        return {
            "symbols": symbols,
            "matrix": matrix,
            "high_correlations": high_corrs,
            "correlation_clusters": clusters,
            "diversification_score": div_score,
            "average_correlation": avg_corr,
            "warnings": warnings,
        }

    def _find_clusters(
        self, symbols: list[str], matrix: CorrelationMatrix
    ) -> list[list[str]]:
        """Find clusters of correlated symbols."""
        clusters: list[set[str]] = []
        processed = set()

        for symbol in symbols:
            if symbol in processed:
                continue

            cluster = {symbol}
            for other in symbols:
                if other != symbol and other not in processed:
                    corr = matrix.get_correlation(symbol, other)
                    if abs(corr) >= self.threshold:
                        cluster.add(other)

            if len(cluster) > 1:
                clusters.append(cluster)
                processed.update(cluster)

        return [list(c) for c in clusters]

    def _calculate_avg_correlation(self, matrix: CorrelationMatrix) -> float:
        """Calculate average pairwise correlation."""
        n = len(matrix.symbols)
        if n < 2:
            return 0.0

        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                corr = float(matrix.matrix.iloc[i, j])
                if not np.isnan(corr):
                    total += abs(corr)
                    count += 1

        return total / count if count > 0 else 0.0

    # ── Formatting ──────────────────────────────────────────────────

    @staticmethod
    def format_correlation_matrix(matrix: CorrelationMatrix) -> str:
        """Format correlation matrix for display."""
        lines = [
            f"\n{'=' * 70}",
            f"  CORRELATION MATRIX ({matrix.lookback_days} days)",
            f"{'=' * 70}",
        ]

        # Header row
        header = "          " + "".join(f"{s:>8}" for s in matrix.symbols)
        lines.append(header)
        lines.append("  " + "-" * (8 + 8 * len(matrix.symbols)))

        # Data rows
        for symbol in matrix.symbols:
            row = f"  {symbol:<8}"
            for other in matrix.symbols:
                corr = matrix.get_correlation(symbol, other)
                if np.isnan(corr):
                    row += "     N/A"
                elif symbol == other:
                    row += "    1.00"
                else:
                    row += f"  {corr:>6.2f}"
            lines.append(row)

        lines.append(f"{'=' * 70}\n")
        return "\n".join(lines)

    @staticmethod
    def format_correlation_result(result: CorrelationResult) -> str:
        """Format correlation check result."""
        lines = [
            f"\n{'─' * 50}",
            f"  CORRELATION CHECK: {result.symbol}",
            f"{'─' * 50}",
        ]

        if result.is_correlated:
            lines.append(f"  Status: CORRELATED")
            lines.append(f"  Reason: {result.reason}")
            lines.append(f"  Correlated with: {', '.join(result.correlated_with)}")
            if result.shared_groups:
                lines.append(f"  Shared sectors: {', '.join(result.shared_groups)}")
            lines.append(f"  Max correlation: {result.max_correlation:.2f}")
        else:
            lines.append(f"  Status: NOT CORRELATED")
            lines.append(f"  {result.symbol} is sufficiently diversified")

        if result.correlation_values:
            lines.append(f"\n  Correlations:")
            for sym, corr in sorted(
                result.correlation_values.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            ):
                marker = " **" if abs(corr) >= 0.7 else ""
                lines.append(f"    {sym}: {corr:+.2f}{marker}")

        lines.append(f"{'─' * 50}\n")
        return "\n".join(lines)

    @staticmethod
    def format_portfolio_analysis(analysis: dict) -> str:
        """Format portfolio correlation analysis."""
        lines = [
            f"\n{'=' * 70}",
            f"  PORTFOLIO CORRELATION ANALYSIS",
            f"{'=' * 70}",
            f"  Symbols: {', '.join(analysis['symbols'])}",
            f"  Diversification Score: {analysis['diversification_score']:.2f} (1.0 = perfect)",
            f"  Average Correlation: {analysis.get('average_correlation', 0):.2f}",
        ]

        if analysis["warnings"]:
            lines.append(f"\n  Warnings:")
            for w in analysis["warnings"]:
                lines.append(f"    - {w}")

        if analysis["high_correlations"]:
            lines.append(f"\n  High Correlations (>= threshold):")
            for s1, s2, corr in analysis["high_correlations"][:10]:
                lines.append(f"    {s1} <-> {s2}: {corr:+.2f}")

        if analysis["correlation_clusters"]:
            lines.append(f"\n  Correlation Clusters:")
            for i, cluster in enumerate(analysis["correlation_clusters"], 1):
                lines.append(f"    {i}. {', '.join(cluster)}")

        lines.append(f"{'=' * 70}\n")
        return "\n".join(lines)
