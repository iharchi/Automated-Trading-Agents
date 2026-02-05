"""Market Regime Detector

Classifies the current market environment to enable strategy adaptation.

Regimes:
    - TRENDING_UP:   Strong uptrend (ADX > threshold, price above MAs, positive slope)
    - TRENDING_DOWN: Strong downtrend (ADX > threshold, price below MAs, negative slope)
    - RANGING:       Sideways market (low ADX, tight Bollinger Bands)
    - VOLATILE:      High volatility / choppy (wide ATR, wide Bollinger Bands)
    - BREAKOUT:      Transitioning out of range (BB squeeze releasing, ADX rising)

Strategy Adjustments:
    - TRENDING_UP:   Favour longs, wider stops, larger positions, trend-following
    - TRENDING_DOWN: Favour shorts/cash, wider stops, smaller positions
    - RANGING:       Mean-reversion, tighter stops, smaller positions
    - VOLATILE:      Reduce size, widen stops, fewer trades
    - BREAKOUT:      Momentum entries, tight trailing stops

Usage:
    detector = MarketRegimeDetector(client)
    result = detector.detect("SPY")
    print(MarketRegimeDetector.format_result(result))

    # Get strategy adjustments
    adjustments = result.adjustments
    print(f"Position size factor: {adjustments['position_size_factor']}")
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

Regime = Literal["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "BREAKOUT"]

# ── Strategy Adjustment Profiles ────────────────────────────────────

REGIME_ADJUSTMENTS = {
    "TRENDING_UP": {
        "position_size_factor": 1.2,    # Increase position size
        "stop_multiplier": 1.3,         # Wider stops for trend-riding
        "take_profit_multiplier": 1.5,  # Let winners run
        "signal_bias": "long",          # Prefer buy signals
        "min_score_adjustment": -1,     # Lower threshold (easier entry)
        "description": "Strong uptrend — favour longs, wider stops, let winners run",
    },
    "TRENDING_DOWN": {
        "position_size_factor": 0.7,    # Smaller positions
        "stop_multiplier": 1.3,         # Wider stops for volatility
        "take_profit_multiplier": 1.0,  # Normal targets
        "signal_bias": "short",         # Prefer sell signals
        "min_score_adjustment": 1,      # Higher threshold (harder entry)
        "description": "Strong downtrend — favour shorts/cash, reduce exposure",
    },
    "RANGING": {
        "position_size_factor": 0.8,    # Moderate positions
        "stop_multiplier": 0.8,         # Tighter stops
        "take_profit_multiplier": 0.8,  # Tighter targets (mean-reversion)
        "signal_bias": "neutral",
        "min_score_adjustment": 0,
        "description": "Sideways market — tighter stops, mean-reversion favoured",
    },
    "VOLATILE": {
        "position_size_factor": 0.5,    # Significantly smaller positions
        "stop_multiplier": 1.5,         # Much wider stops
        "take_profit_multiplier": 1.2,  # Slightly wider targets
        "signal_bias": "neutral",
        "min_score_adjustment": 2,      # Much higher threshold (fewer trades)
        "description": "High volatility — reduce position size, widen stops, fewer trades",
    },
    "BREAKOUT": {
        "position_size_factor": 1.0,    # Normal positions
        "stop_multiplier": 1.0,         # Normal stops
        "take_profit_multiplier": 1.3,  # Extended targets
        "signal_bias": "momentum",      # Follow breakout direction
        "min_score_adjustment": -1,     # Lower threshold for momentum
        "description": "Breakout detected — momentum entries, follow the move",
    },
}


@dataclass
class RegimeIndicators:
    """Raw indicator values used for regime classification."""

    adx: float = 0.0
    adx_slope: float = 0.0          # ADX change over last N bars
    atr: float = 0.0
    atr_pct: float = 0.0            # ATR as % of price
    historical_vol: float = 0.0     # Annualised historical volatility
    bb_width: float = 0.0           # Bollinger Band width (% of price)
    bb_squeeze: bool = False        # BB width at local minimum
    bb_expansion: bool = False      # BB width expanding
    ema_short: float = 0.0
    ema_long: float = 0.0
    ema_slope_short: float = 0.0    # Short EMA slope
    ema_slope_long: float = 0.0     # Long EMA slope
    price: float = 0.0
    price_vs_ema_short: float = 0.0  # % above/below short EMA
    price_vs_ema_long: float = 0.0   # % above/below long EMA
    volume_trend: float = 0.0       # Volume SMA ratio (current vs avg)


@dataclass
class RegimeResult:
    """Result of regime detection."""

    symbol: str
    regime: Regime
    confidence: float = 0.0         # 0.0 to 1.0
    indicators: RegimeIndicators = field(default_factory=RegimeIndicators)
    adjustments: dict = field(default_factory=dict)
    secondary_regime: str = ""      # Next most likely regime
    description: str = ""
    error: str = ""


class MarketRegimeDetector:
    """Detects market regime from price data and provides strategy adjustments."""

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        ema_short_period: int = 9,
        ema_long_period: int = 50,
        vol_lookback: int = 20,
        vol_high_threshold: float = 0.30,  # 30% annualised vol = "high"
        bb_squeeze_percentile: float = 20.0,  # BB width in bottom 20% = squeeze
        slope_lookback: int = 5,
    ):
        self.client = client
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.ema_short_period = ema_short_period
        self.ema_long_period = ema_long_period
        self.vol_lookback = vol_lookback
        self.vol_high_threshold = vol_high_threshold
        self.bb_squeeze_percentile = bb_squeeze_percentile
        self.slope_lookback = slope_lookback

    # ── Indicator Calculation ───────────────────────────────────────

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ADX from OHLC data."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.ewm(span=period, adjust=False).mean()

    def _calculate_bb_width(self, close: pd.Series, period: int, std: float) -> pd.Series:
        """Calculate Bollinger Band width as percentage of price."""
        sma = close.rolling(period).mean()
        rolling_std = close.rolling(period).std()
        upper = sma + std * rolling_std
        lower = sma - std * rolling_std
        width = (upper - lower) / sma * 100
        return width

    def _calculate_indicators(self, df: pd.DataFrame) -> RegimeIndicators:
        """Calculate all indicators from price data."""
        close = df["close"]
        n = len(df)

        if n < self.ema_long_period + 10:
            logger.warning("Insufficient data for regime detection (%d bars)", n)
            return RegimeIndicators(price=float(close.iloc[-1]) if n > 0 else 0)

        # ADX
        adx_series = self._calculate_adx(df, self.adx_period)
        adx = float(adx_series.iloc[-1])
        adx_slope = float(adx_series.iloc[-1] - adx_series.iloc[-self.slope_lookback]) if n > self.slope_lookback else 0

        # ATR
        atr_series = self._calculate_atr(df, self.adx_period)
        atr = float(atr_series.iloc[-1])
        price = float(close.iloc[-1])
        atr_pct = (atr / price * 100) if price > 0 else 0

        # Historical volatility (annualised)
        returns = close.pct_change().dropna()
        if len(returns) >= self.vol_lookback:
            recent_vol = float(returns.tail(self.vol_lookback).std() * np.sqrt(252))
        else:
            recent_vol = 0.0

        # Bollinger Band width
        bb_width_series = self._calculate_bb_width(close, self.bb_period, self.bb_std)
        bb_width = float(bb_width_series.iloc[-1])

        # BB squeeze/expansion detection
        if len(bb_width_series.dropna()) >= 50:
            recent_bb = bb_width_series.dropna().tail(50)
            bb_percentile = float((recent_bb < bb_width).sum() / len(recent_bb) * 100)
            bb_squeeze = bb_percentile <= self.bb_squeeze_percentile
            bb_expansion = bb_percentile >= 80
        else:
            bb_squeeze = False
            bb_expansion = False

        # EMAs
        ema_short = close.ewm(span=self.ema_short_period, adjust=False).mean()
        ema_long = close.ewm(span=self.ema_long_period, adjust=False).mean()

        ema_s = float(ema_short.iloc[-1])
        ema_l = float(ema_long.iloc[-1])

        # EMA slopes (normalised as %)
        ema_slope_short = float(
            (ema_short.iloc[-1] - ema_short.iloc[-self.slope_lookback])
            / ema_short.iloc[-self.slope_lookback] * 100
        ) if n > self.slope_lookback else 0

        ema_slope_long = float(
            (ema_long.iloc[-1] - ema_long.iloc[-self.slope_lookback])
            / ema_long.iloc[-self.slope_lookback] * 100
        ) if n > self.slope_lookback else 0

        # Price relative to EMAs
        price_vs_ema_short = (price - ema_s) / ema_s * 100 if ema_s > 0 else 0
        price_vs_ema_long = (price - ema_l) / ema_l * 100 if ema_l > 0 else 0

        # Volume trend (if available)
        volume_trend = 0.0
        if "volume" in df.columns:
            vol = df["volume"]
            vol_sma = vol.rolling(20).mean()
            if len(vol_sma.dropna()) > 0 and vol_sma.iloc[-1] > 0:
                volume_trend = float(vol.iloc[-1] / vol_sma.iloc[-1])

        return RegimeIndicators(
            adx=adx,
            adx_slope=adx_slope,
            atr=atr,
            atr_pct=atr_pct,
            historical_vol=recent_vol,
            bb_width=bb_width,
            bb_squeeze=bb_squeeze,
            bb_expansion=bb_expansion,
            ema_short=ema_s,
            ema_long=ema_l,
            ema_slope_short=ema_slope_short,
            ema_slope_long=ema_slope_long,
            price=price,
            price_vs_ema_short=price_vs_ema_short,
            price_vs_ema_long=price_vs_ema_long,
            volume_trend=volume_trend,
        )

    # ── Regime Classification ───────────────────────────────────────

    def _classify_regime(self, ind: RegimeIndicators) -> tuple[Regime, float, str]:
        """Classify regime based on indicators.

        Returns:
            Tuple of (regime, confidence, secondary_regime)
        """
        scores: dict[Regime, float] = {
            "TRENDING_UP": 0.0,
            "TRENDING_DOWN": 0.0,
            "RANGING": 0.0,
            "VOLATILE": 0.0,
            "BREAKOUT": 0.0,
        }

        # ── ADX analysis (trend strength) ─────────────
        if ind.adx >= self.adx_trend_threshold:
            # Strong trend
            if ind.price_vs_ema_long > 0 and ind.ema_slope_long > 0:
                scores["TRENDING_UP"] += 3.0
            elif ind.price_vs_ema_long < 0 and ind.ema_slope_long < 0:
                scores["TRENDING_DOWN"] += 3.0
        else:
            # Weak trend
            scores["RANGING"] += 2.0

        # ── EMA alignment ─────────────────────────────
        if ind.ema_short > ind.ema_long and ind.ema_slope_short > 0:
            scores["TRENDING_UP"] += 2.0
        elif ind.ema_short < ind.ema_long and ind.ema_slope_short < 0:
            scores["TRENDING_DOWN"] += 2.0
        else:
            scores["RANGING"] += 1.0

        # ── Price position ────────────────────────────
        if ind.price_vs_ema_short > 1.0 and ind.price_vs_ema_long > 2.0:
            scores["TRENDING_UP"] += 1.5
        elif ind.price_vs_ema_short < -1.0 and ind.price_vs_ema_long < -2.0:
            scores["TRENDING_DOWN"] += 1.5

        # ── Volatility analysis ───────────────────────
        if ind.historical_vol >= self.vol_high_threshold:
            scores["VOLATILE"] += 3.0
            # Reduce trending confidence in high vol
            scores["TRENDING_UP"] *= 0.7
            scores["TRENDING_DOWN"] *= 0.7
        elif ind.historical_vol < self.vol_high_threshold * 0.5:
            # Low volatility = more likely ranging
            scores["RANGING"] += 1.0

        # ── Bollinger Band analysis ───────────────────
        if ind.bb_squeeze:
            scores["BREAKOUT"] += 2.5
            scores["RANGING"] += 1.0
        elif ind.bb_expansion:
            if ind.adx_slope > 2:
                scores["BREAKOUT"] += 2.0
            scores["VOLATILE"] += 1.0

        # ── ADX slope (trend developing/weakening) ────
        if ind.adx_slope > 3:
            # ADX rising = trend strengthening or breakout
            scores["BREAKOUT"] += 1.5
            if ind.price_vs_ema_long > 0:
                scores["TRENDING_UP"] += 1.0
            else:
                scores["TRENDING_DOWN"] += 1.0
        elif ind.adx_slope < -3:
            # ADX falling = trend weakening
            scores["RANGING"] += 1.5

        # ── Volume confirmation ───────────────────────
        if ind.volume_trend > 1.5:
            # High volume = confirms breakout or trend
            scores["BREAKOUT"] += 1.0
            if ind.adx >= self.adx_trend_threshold:
                if ind.price_vs_ema_long > 0:
                    scores["TRENDING_UP"] += 0.5
                else:
                    scores["TRENDING_DOWN"] += 0.5

        # ── Determine winner ──────────────────────────
        sorted_regimes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        winner = sorted_regimes[0]
        runner_up = sorted_regimes[1]

        total_score = sum(scores.values())
        confidence = winner[1] / total_score if total_score > 0 else 0.0

        return winner[0], confidence, runner_up[0]

    # ── Public API ──────────────────────────────────────────────────

    def detect(
        self,
        symbol: str,
        *,
        timeframe: str = "1Day",
        bars: pd.DataFrame | None = None,
    ) -> RegimeResult:
        """Detect the market regime for a symbol.

        Args:
            symbol: Ticker symbol
            timeframe: Bar timeframe for analysis
            bars: Pre-fetched bars (optional, otherwise fetched via client)

        Returns:
            RegimeResult with regime classification and adjustments
        """
        if bars is None:
            if not self.client:
                return RegimeResult(
                    symbol=symbol,
                    regime="RANGING",
                    error="No client or data provided",
                )
            try:
                bars = self.client.get_bars(symbol, timeframe=timeframe, limit=200)
            except Exception as e:
                return RegimeResult(
                    symbol=symbol,
                    regime="RANGING",
                    error=f"Failed to fetch data: {e}",
                )

        if bars.empty or len(bars) < 30:
            return RegimeResult(
                symbol=symbol,
                regime="RANGING",
                error="Insufficient data",
            )

        # Calculate indicators
        indicators = self._calculate_indicators(bars)

        # Classify regime
        regime, confidence, secondary = self._classify_regime(indicators)

        # Get strategy adjustments
        adjustments = REGIME_ADJUSTMENTS.get(regime, REGIME_ADJUSTMENTS["RANGING"]).copy()

        return RegimeResult(
            symbol=symbol,
            regime=regime,
            confidence=confidence,
            indicators=indicators,
            adjustments=adjustments,
            secondary_regime=secondary,
            description=adjustments.get("description", ""),
        )

    def detect_multi(
        self,
        symbols: list[str],
        *,
        timeframe: str = "1Day",
    ) -> dict[str, RegimeResult]:
        """Detect regimes for multiple symbols.

        Returns:
            Dict mapping symbol to RegimeResult
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.detect(symbol, timeframe=timeframe)
        return results

    def detect_market_regime(self, *, timeframe: str = "1Day") -> RegimeResult:
        """Detect the broad market regime using SPY as proxy.

        Returns:
            RegimeResult for SPY representing overall market conditions
        """
        return self.detect("SPY", timeframe=timeframe)

    # ── Formatting ──────────────────────────────────────────────────

    @staticmethod
    def format_result(result: RegimeResult) -> str:
        """Format a single regime result."""
        ind = result.indicators

        regime_emoji_map = {
            "TRENDING_UP": "BULL",
            "TRENDING_DOWN": "BEAR",
            "RANGING": "FLAT",
            "VOLATILE": "CHOP",
            "BREAKOUT": "BRKT",
        }
        tag = regime_emoji_map.get(result.regime, "????")

        lines = [
            f"\n{'=' * 62}",
            f"  MARKET REGIME: {result.symbol}",
            f"{'=' * 62}",
            f"  Regime        : [{tag}] {result.regime}",
            f"  Confidence    : {result.confidence:.0%}",
            f"  Secondary     : {result.secondary_regime}",
            f"  Description   : {result.description}",
        ]

        if result.error:
            lines.append(f"  Error         : {result.error}")
        else:
            lines.append(f"")
            lines.append(f"  Indicators:")
            lines.append(f"    Price       : ${ind.price:,.2f}")
            lines.append(f"    ADX         : {ind.adx:.1f}  (slope: {ind.adx_slope:+.1f})")
            lines.append(f"    ATR         : ${ind.atr:.2f}  ({ind.atr_pct:.2f}%)")
            lines.append(f"    Hist Vol    : {ind.historical_vol:.1%}")
            lines.append(f"    BB Width    : {ind.bb_width:.2f}%  {'[SQUEEZE]' if ind.bb_squeeze else '[EXPANDING]' if ind.bb_expansion else ''}")
            lines.append(f"    EMA Short   : ${ind.ema_short:,.2f}  (slope: {ind.ema_slope_short:+.3f}%)")
            lines.append(f"    EMA Long    : ${ind.ema_long:,.2f}  (slope: {ind.ema_slope_long:+.3f}%)")
            lines.append(f"    Price vs EMA: short={ind.price_vs_ema_short:+.2f}%  long={ind.price_vs_ema_long:+.2f}%")
            if ind.volume_trend > 0:
                lines.append(f"    Volume Trend: {ind.volume_trend:.2f}x avg")

            adj = result.adjustments
            lines.append(f"")
            lines.append(f"  Strategy Adjustments:")
            lines.append(f"    Position Size: {adj.get('position_size_factor', 1.0):.1f}x")
            lines.append(f"    Stop Width   : {adj.get('stop_multiplier', 1.0):.1f}x")
            lines.append(f"    Target Width : {adj.get('take_profit_multiplier', 1.0):.1f}x")
            lines.append(f"    Signal Bias  : {adj.get('signal_bias', 'neutral')}")
            lines.append(f"    Score Adjust : {adj.get('min_score_adjustment', 0):+d}")

        lines.append(f"{'=' * 62}\n")
        return "\n".join(lines)

    @staticmethod
    def format_multi(results: dict[str, RegimeResult]) -> str:
        """Format regime results for multiple symbols as a summary table."""
        if not results:
            return "\n  No regime data.\n"

        lines = [
            f"\n{'=' * 78}",
            f"  MARKET REGIME SUMMARY",
            f"{'=' * 78}",
            f"  {'Symbol':<8} {'Regime':<16} {'Conf':>6} {'ADX':>6} {'Vol':>8} {'BB Width':>8} {'Bias':<10}",
            f"  {'-' * 8} {'-' * 16} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 10}",
        ]

        for symbol, result in results.items():
            ind = result.indicators
            bias = result.adjustments.get("signal_bias", "neutral")
            lines.append(
                f"  {symbol:<8} "
                f"{result.regime:<16} "
                f"{result.confidence:>5.0%} "
                f"{ind.adx:>6.1f} "
                f"{ind.historical_vol:>7.1%} "
                f"{ind.bb_width:>7.2f}% "
                f"{bias:<10}"
            )

        lines.append(f"{'=' * 78}\n")
        return "\n".join(lines)
