"""Technical Analysis Agent

Computes a suite of technical indicators on historical price data and
produces a composite BUY / SELL / HOLD signal.  When running in *live*
mode it can also submit paper-trade orders through the Alpaca client.

Indicators used:
    - RSI (14)
    - MACD (12, 26, 9)
    - Bollinger Bands (20, 2)
    - EMA crossover (short=9, long=21)
    - Stochastic Oscillator (14, 3)
    - ADX (14) — trend strength filter
    - ATR (14) — for position-sizing context, not signal generation
"""

import logging
from dataclasses import dataclass, field

import pandas as pd
import ta

from agents.base_agent import BaseAgent
from utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


# ── Signal thresholds ────────────────────────────────────────────

RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
STOCH_OVERSOLD = 20
STOCH_OVERBOUGHT = 80
ADX_TREND_THRESHOLD = 25

# Each indicator vote is +1 (bullish), -1 (bearish), or 0 (neutral).
# The composite score is the sum; thresholds for action:
BUY_THRESHOLD = 2
SELL_THRESHOLD = -2


@dataclass
class IndicatorResult:
    """Container for a single indicator's output."""

    name: str
    value: float
    signal: int  # +1 bullish, -1 bearish, 0 neutral
    detail: str = ""


@dataclass
class AnalysisResult:
    """Aggregated analysis for one symbol."""

    symbol: str
    indicators: list[IndicatorResult] = field(default_factory=list)
    composite_score: int = 0
    signal: str = "HOLD"  # BUY | SELL | HOLD
    current_price: float = 0.0
    atr: float = 0.0


class TechnicalAnalysisAgent(BaseAgent):
    """Agent that generates trading signals from technical indicators."""

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        rsi_period: int = 14,
        ema_short: int = 9,
        ema_long: int = 21,
        bb_period: int = 20,
        bb_std: int = 2,
        stoch_period: int = 14,
        stoch_smooth: int = 3,
        adx_period: int = 14,
        auto_trade: bool = False,
        default_qty: int = 1,
    ):
        super().__init__(name="TechnicalAnalysis", client=client)
        self.rsi_period = rsi_period
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.stoch_period = stoch_period
        self.stoch_smooth = stoch_smooth
        self.adx_period = adx_period
        self.auto_trade = auto_trade
        self.default_qty = default_qty

    # ── Indicator helpers ────────────────────────────────────

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> IndicatorResult:
        rsi = ta.momentum.RSIIndicator(close, window=period).rsi().iloc[-1]
        if rsi <= RSI_OVERSOLD:
            sig, detail = 1, f"RSI {rsi:.1f} — oversold"
        elif rsi >= RSI_OVERBOUGHT:
            sig, detail = -1, f"RSI {rsi:.1f} — overbought"
        else:
            sig, detail = 0, f"RSI {rsi:.1f} — neutral"
        return IndicatorResult(name="RSI", value=round(rsi, 2), signal=sig, detail=detail)

    @staticmethod
    def _compute_macd(close: pd.Series) -> IndicatorResult:
        macd_ind = ta.trend.MACD(close)
        macd_line = macd_ind.macd().iloc[-1]
        signal_line = macd_ind.macd_signal().iloc[-1]
        hist = macd_ind.macd_diff().iloc[-1]

        if macd_line > signal_line and hist > 0:
            sig, detail = 1, f"MACD bullish crossover (hist={hist:.4f})"
        elif macd_line < signal_line and hist < 0:
            sig, detail = -1, f"MACD bearish crossover (hist={hist:.4f})"
        else:
            sig, detail = 0, f"MACD neutral (hist={hist:.4f})"
        return IndicatorResult(name="MACD", value=round(hist, 4), signal=sig, detail=detail)

    @staticmethod
    def _compute_bollinger(close: pd.Series, period: int = 20, std: int = 2) -> IndicatorResult:
        bb = ta.volatility.BollingerBands(close, window=period, window_dev=std)
        price = close.iloc[-1]
        upper = bb.bollinger_hband().iloc[-1]
        lower = bb.bollinger_lband().iloc[-1]

        if price <= lower:
            sig, detail = 1, f"Price ${price:.2f} at/below lower band ${lower:.2f}"
        elif price >= upper:
            sig, detail = -1, f"Price ${price:.2f} at/above upper band ${upper:.2f}"
        else:
            sig, detail = 0, f"Price ${price:.2f} within bands (${lower:.2f}–${upper:.2f})"
        return IndicatorResult(name="Bollinger", value=round(price, 2), signal=sig, detail=detail)

    @staticmethod
    def _compute_ema_crossover(close: pd.Series, short: int = 9, long: int = 21) -> IndicatorResult:
        ema_s = ta.trend.EMAIndicator(close, window=short).ema_indicator().iloc[-1]
        ema_l = ta.trend.EMAIndicator(close, window=long).ema_indicator().iloc[-1]

        if ema_s > ema_l:
            sig, detail = 1, f"EMA{short} ({ema_s:.2f}) > EMA{long} ({ema_l:.2f}) — bullish"
        elif ema_s < ema_l:
            sig, detail = -1, f"EMA{short} ({ema_s:.2f}) < EMA{long} ({ema_l:.2f}) — bearish"
        else:
            sig, detail = 0, f"EMAs equal ({ema_s:.2f})"
        return IndicatorResult(name="EMA_Cross", value=round(ema_s - ema_l, 4), signal=sig, detail=detail)

    @staticmethod
    def _compute_stochastic(
        high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14, smooth: int = 3,
    ) -> IndicatorResult:
        stoch = ta.momentum.StochasticOscillator(
            high, low, close, window=period, smooth_window=smooth,
        )
        k = stoch.stoch().iloc[-1]
        d = stoch.stoch_signal().iloc[-1]

        if k <= STOCH_OVERSOLD and d <= STOCH_OVERSOLD:
            sig, detail = 1, f"Stoch %K={k:.1f} %D={d:.1f} — oversold"
        elif k >= STOCH_OVERBOUGHT and d >= STOCH_OVERBOUGHT:
            sig, detail = -1, f"Stoch %K={k:.1f} %D={d:.1f} — overbought"
        else:
            sig, detail = 0, f"Stoch %K={k:.1f} %D={d:.1f} — neutral"
        return IndicatorResult(name="Stochastic", value=round(k, 2), signal=sig, detail=detail)

    @staticmethod
    def _compute_adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14,
    ) -> IndicatorResult:
        adx_ind = ta.trend.ADXIndicator(high, low, close, window=period)
        adx = adx_ind.adx().iloc[-1]
        plus_di = adx_ind.adx_pos().iloc[-1]
        minus_di = adx_ind.adx_neg().iloc[-1]

        if adx >= ADX_TREND_THRESHOLD:
            # Strong trend — vote with the direction
            if plus_di > minus_di:
                sig, detail = 1, f"ADX {adx:.1f} trending UP (+DI={plus_di:.1f} > -DI={minus_di:.1f})"
            else:
                sig, detail = -1, f"ADX {adx:.1f} trending DOWN (-DI={minus_di:.1f} > +DI={plus_di:.1f})"
        else:
            sig, detail = 0, f"ADX {adx:.1f} — no strong trend"
        return IndicatorResult(name="ADX", value=round(adx, 2), signal=sig, detail=detail)

    @staticmethod
    def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        return round(
            ta.volatility.AverageTrueRange(high, low, close, window=period)
            .average_true_range()
            .iloc[-1],
            4,
        )

    # ── Core lifecycle ───────────────────────────────────────

    def analyze(self, symbol: str, **kwargs) -> dict:
        """Fetch bars, compute indicators, return AnalysisResult as dict."""
        timeframe = kwargs.get("timeframe", "1Day")
        limit = kwargs.get("limit", 100)

        bars = self.client.get_bars(symbol, timeframe=timeframe, limit=limit)
        if bars.empty or len(bars) < self.ema_long:
            self.logger.warning("Insufficient data for %s (%d bars)", symbol, len(bars))
            return AnalysisResult(symbol=symbol).__dict__

        close = bars["close"]
        high = bars["high"]
        low = bars["low"]

        indicators = [
            self._compute_rsi(close, self.rsi_period),
            self._compute_macd(close),
            self._compute_bollinger(close, self.bb_period, self.bb_std),
            self._compute_ema_crossover(close, self.ema_short, self.ema_long),
            self._compute_stochastic(high, low, close, self.stoch_period, self.stoch_smooth),
            self._compute_adx(high, low, close, self.adx_period),
        ]

        composite = sum(ind.signal for ind in indicators)
        if composite >= BUY_THRESHOLD:
            signal = "BUY"
        elif composite <= SELL_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"

        result = AnalysisResult(
            symbol=symbol,
            indicators=indicators,
            composite_score=composite,
            signal=signal,
            current_price=round(float(close.iloc[-1]), 2),
            atr=self._compute_atr(high, low, close),
        )
        return result.__dict__

    def execute(self, symbol: str, analysis: dict, **kwargs) -> dict:
        """Optionally place a paper trade based on the signal."""
        signal = analysis.get("signal", "HOLD")
        result = {
            "symbol": symbol,
            "signal": signal,
            "composite_score": analysis.get("composite_score", 0),
            "current_price": analysis.get("current_price"),
            "atr": analysis.get("atr"),
            "order": None,
        }

        if not self.auto_trade or signal == "HOLD":
            return result

        qty = kwargs.get("qty", self.default_qty)
        side = "buy" if signal == "BUY" else "sell"

        # Safety: don't sell if we have no position
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
        """Return a human-readable summary of an analysis dict."""
        lines = [
            f"\n{'=' * 55}",
            f"  Technical Analysis — {analysis['symbol']}",
            f"  Price: ${analysis['current_price']}  |  ATR(14): {analysis['atr']}",
            f"{'=' * 55}",
        ]
        for ind in analysis.get("indicators", []):
            if isinstance(ind, dict):
                arrow = {1: "▲", -1: "▼", 0: "—"}[ind["signal"]]
                lines.append(f"  {arrow} {ind['name']:12s}  {ind['detail']}")
            else:
                arrow = {1: "▲", -1: "▼", 0: "—"}[ind.signal]
                lines.append(f"  {arrow} {ind.name:12s}  {ind.detail}")
        lines.append(f"{'─' * 55}")
        lines.append(
            f"  Composite Score: {analysis['composite_score']:+d}  →  "
            f"Signal: {analysis['signal']}"
        )
        lines.append(f"{'=' * 55}\n")
        return "\n".join(lines)
