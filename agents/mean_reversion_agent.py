"""Mean Reversion Strategy Agent

Identifies overbought / oversold conditions where price has deviated
significantly from its mean, then trades the expected snap-back.

Best suited to **ranging / sideways** markets (detected by low ADX or a
"sideways" regime from the Market Regime Detector).

Indicators:
    - Bollinger Bands (20, 2) — deviation from mean
    - RSI (14) — momentum confirmation
    - Z-score of price vs. 20-day SMA — statistical distance
    - Stochastic Oscillator — additional oversold/overbought filter
    - ATR (14) — for stop-loss / take-profit sizing

Entry rules (long):
    1. Price closes below the lower Bollinger Band.
    2. RSI < oversold threshold.
    3. Z-score < -z_entry_threshold.
    4. Stochastic %K < stochastic oversold level.
    (short entries mirror these above the upper band)

Exit rules:
    - Price reverts to the SMA (mean) — take profit.
    - ATR-based stop-loss if the move continues against us.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from agents.base_agent import BaseAgent
from utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


@dataclass
class MeanReversionResult:
    """Analysis result for mean-reversion signal."""
    symbol: str = ""
    signal: str = "HOLD"          # BUY, SELL, HOLD
    score: float = 0.0            # -1.0 to +1.0 continuous score

    # Indicator values
    current_price: float = 0.0
    sma: float = 0.0
    upper_band: float = 0.0
    lower_band: float = 0.0
    bb_width: float = 0.0
    z_score: float = 0.0
    rsi: float = 0.0
    stochastic_k: float = 0.0
    atr: float = 0.0
    adx: float = 0.0

    # Levels
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # Context
    regime: str = ""
    detail: str = ""


class MeanReversionAgent(BaseAgent):
    """Agent that trades mean-reversion setups in ranging markets."""

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        # Bollinger
        bb_period: int = 20,
        bb_std: float = 2.0,
        # RSI
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        # Z-score
        z_entry_threshold: float = 1.5,
        z_exit_threshold: float = 0.0,
        # Stochastic
        stoch_period: int = 14,
        stoch_smooth: int = 3,
        stoch_oversold: int = 20,
        stoch_overbought: int = 80,
        # ADX filter
        adx_period: int = 14,
        adx_max_threshold: float = 25.0,
        # Risk
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,
        take_profit_ratio: float = 2.0,
        # Execution
        auto_trade: bool = False,
        default_qty: int = 1,
        # Signal
        min_confirmations: int = 2,
    ):
        super().__init__(name="MeanReversion", client=client)

        # Bollinger
        self.bb_period = bb_period
        self.bb_std = bb_std
        # RSI
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        # Z-score
        self.z_entry_threshold = z_entry_threshold
        self.z_exit_threshold = z_exit_threshold
        # Stochastic
        self.stoch_period = stoch_period
        self.stoch_smooth = stoch_smooth
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought
        # ADX
        self.adx_period = adx_period
        self.adx_max_threshold = adx_max_threshold
        # Risk
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.take_profit_ratio = take_profit_ratio
        # Execution
        self.auto_trade = auto_trade
        self.default_qty = default_qty
        # Signal
        self.min_confirmations = min_confirmations

    # ── Indicator computation ─────────────────────────────────

    @staticmethod
    def _z_score(close: pd.Series, period: int = 20) -> pd.Series:
        """Compute rolling z-score of price vs SMA."""
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        return (close - sma) / std.replace(0, float("nan"))

    # ── Core analysis ─────────────────────────────────────────

    def analyze(self, symbol: str, **kwargs) -> dict:
        """Compute mean-reversion indicators and generate signal."""
        import ta as ta_lib

        timeframe = kwargs.get("timeframe", "1Day")
        limit = kwargs.get("limit", 100)

        bars = self.client.get_bars(symbol, timeframe=timeframe, limit=limit)
        if bars.empty or len(bars) < max(self.bb_period, self.rsi_period) + 5:
            self.logger.warning(
                "Insufficient data for %s (%d bars)", symbol, len(bars)
            )
            return MeanReversionResult(symbol=symbol).__dict__

        close = bars["close"]
        high = bars["high"]
        low = bars["low"]

        # ── Indicators ──────────────────────────────────────
        # Bollinger Bands
        bb = ta_lib.volatility.BollingerBands(
            close, window=self.bb_period, window_dev=self.bb_std,
        )
        upper = bb.bollinger_hband().iloc[-1]
        lower = bb.bollinger_lband().iloc[-1]
        sma = bb.bollinger_mavg().iloc[-1]
        bb_width = (upper - lower) / sma if sma > 0 else 0.0

        # RSI
        rsi = ta_lib.momentum.RSIIndicator(
            close, window=self.rsi_period,
        ).rsi().iloc[-1]

        # Z-score
        z_series = self._z_score(close, self.bb_period)
        z = z_series.iloc[-1] if not pd.isna(z_series.iloc[-1]) else 0.0

        # Stochastic
        stoch = ta_lib.momentum.StochasticOscillator(
            high, low, close,
            window=self.stoch_period, smooth_window=self.stoch_smooth,
        )
        stoch_k = stoch.stoch().iloc[-1]

        # ATR
        atr_val = ta_lib.volatility.AverageTrueRange(
            high, low, close, window=self.atr_period,
        ).average_true_range().iloc[-1]

        # ADX — trend strength filter
        adx_ind = ta_lib.trend.ADXIndicator(
            high, low, close, window=self.adx_period,
        )
        adx_val = adx_ind.adx().iloc[-1]

        price = float(close.iloc[-1])

        # ── Signal logic ────────────────────────────────────
        bull_signals = 0
        bear_signals = 0
        details = []

        # 1. Bollinger Band touch / breach
        if price <= lower:
            bull_signals += 1
            details.append(f"Price ${price:.2f} <= lower BB ${lower:.2f}")
        elif price >= upper:
            bear_signals += 1
            details.append(f"Price ${price:.2f} >= upper BB ${upper:.2f}")

        # 2. RSI extreme
        if rsi <= self.rsi_oversold:
            bull_signals += 1
            details.append(f"RSI {rsi:.1f} <= {self.rsi_oversold} (oversold)")
        elif rsi >= self.rsi_overbought:
            bear_signals += 1
            details.append(f"RSI {rsi:.1f} >= {self.rsi_overbought} (overbought)")

        # 3. Z-score
        if z <= -self.z_entry_threshold:
            bull_signals += 1
            details.append(f"Z-score {z:.2f} <= -{self.z_entry_threshold}")
        elif z >= self.z_entry_threshold:
            bear_signals += 1
            details.append(f"Z-score {z:.2f} >= {self.z_entry_threshold}")

        # 4. Stochastic
        if stoch_k <= self.stoch_oversold:
            bull_signals += 1
            details.append(f"Stoch %K {stoch_k:.1f} <= {self.stoch_oversold}")
        elif stoch_k >= self.stoch_overbought:
            bear_signals += 1
            details.append(f"Stoch %K {stoch_k:.1f} >= {self.stoch_overbought}")

        # ADX filter: only trade mean reversion in low-trend environments
        trending = adx_val >= self.adx_max_threshold
        if trending:
            details.append(
                f"ADX {adx_val:.1f} >= {self.adx_max_threshold} — TRENDING, "
                "mean reversion suppressed"
            )

        # Determine signal
        signal = "HOLD"
        score = 0.0
        stop_loss = 0.0
        take_profit = 0.0

        if not trending:
            if bull_signals >= self.min_confirmations:
                signal = "BUY"
                score = min(bull_signals / 4.0, 1.0)
                stop_loss = price - atr_val * self.atr_stop_mult
                take_profit = sma  # revert to mean
                # If mean is too close, use ATR-based TP
                if take_profit <= price:
                    take_profit = price + atr_val * self.atr_stop_mult * self.take_profit_ratio

            elif bear_signals >= self.min_confirmations:
                signal = "SELL"
                score = -min(bear_signals / 4.0, 1.0)
                stop_loss = price + atr_val * self.atr_stop_mult
                take_profit = sma
                if take_profit >= price:
                    take_profit = price - atr_val * self.atr_stop_mult * self.take_profit_ratio

        result = MeanReversionResult(
            symbol=symbol,
            signal=signal,
            score=score,
            current_price=round(price, 2),
            sma=round(sma, 2),
            upper_band=round(upper, 2),
            lower_band=round(lower, 2),
            bb_width=round(bb_width, 4),
            z_score=round(z, 4),
            rsi=round(rsi, 2),
            stochastic_k=round(stoch_k, 2),
            atr=round(atr_val, 4),
            adx=round(adx_val, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            regime="trending" if trending else "ranging",
            detail="; ".join(details) if details else "No signals triggered",
        )
        return result.__dict__

    # ── Execution ─────────────────────────────────────────────

    def execute(self, symbol: str, analysis: dict, **kwargs) -> dict:
        """Optionally place a trade based on mean-reversion signal."""
        signal = analysis.get("signal", "HOLD")
        result = {
            "symbol": symbol,
            "signal": signal,
            "score": analysis.get("score", 0.0),
            "current_price": analysis.get("current_price"),
            "stop_loss": analysis.get("stop_loss"),
            "take_profit": analysis.get("take_profit"),
            "regime": analysis.get("regime"),
            "atr": analysis.get("atr"),
            "order": None,
        }

        if not self.auto_trade or signal == "HOLD":
            return result

        qty = kwargs.get("qty", self.default_qty)
        side = "buy" if signal == "BUY" else "sell"

        # Don't sell if we have no position
        if side == "sell":
            positions = {p["symbol"]: p for p in self.client.get_positions()}
            if symbol not in positions:
                self.logger.info("No position in %s to sell — skipping.", symbol)
                result["order"] = "skipped_no_position"
                return result

        order = self.client.submit_order(symbol=symbol, qty=qty, side=side)
        result["order"] = order
        return result

    # ── Formatting ────────────────────────────────────────────

    @staticmethod
    def format_analysis(analysis: dict) -> str:
        """Return a human-readable summary of a mean-reversion analysis."""
        lines = [
            f"\n{'=' * 60}",
            f"  Mean Reversion Analysis — {analysis.get('symbol', '?')}",
            f"  Price: ${analysis.get('current_price', 0):.2f}  |  "
            f"Regime: {analysis.get('regime', '?')}",
            f"{'=' * 60}",
            f"  SMA(20)       : ${analysis.get('sma', 0):.2f}",
            f"  Upper BB      : ${analysis.get('upper_band', 0):.2f}",
            f"  Lower BB      : ${analysis.get('lower_band', 0):.2f}",
            f"  BB Width      : {analysis.get('bb_width', 0):.4f}",
            f"  Z-Score       : {analysis.get('z_score', 0):+.4f}",
            f"  RSI           : {analysis.get('rsi', 0):.2f}",
            f"  Stoch %K      : {analysis.get('stochastic_k', 0):.2f}",
            f"  ADX           : {analysis.get('adx', 0):.2f}",
            f"  ATR(14)       : {analysis.get('atr', 0):.4f}",
            f"{'─' * 60}",
        ]

        signal = analysis.get("signal", "HOLD")
        score = analysis.get("score", 0)
        lines.append(f"  Signal: {signal}  (score: {score:+.2f})")

        if signal != "HOLD":
            lines.append(
                f"  Stop Loss     : ${analysis.get('stop_loss', 0):.2f}"
            )
            lines.append(
                f"  Take Profit   : ${analysis.get('take_profit', 0):.2f}"
            )

        detail = analysis.get("detail", "")
        if detail:
            lines.append(f"\n  Reasons: {detail}")

        lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)
