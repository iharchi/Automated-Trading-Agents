"""Scalping-Specific Technical Indicators

Fast indicators optimized for sub-minute trading:
- VWAP (Volume Weighted Average Price)
- Momentum oscillators
- Volume spike detection
- Price action patterns
- Microstructure signals
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScalpSignal:
    """Result of scalping analysis."""
    symbol: str
    signal: str  # "BUY", "SELL", "HOLD"
    strength: float  # -1.0 to 1.0
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    reasons: list[str]

    # Indicator values
    vwap: float = 0.0
    vwap_deviation: float = 0.0
    momentum: float = 0.0
    volume_ratio: float = 0.0
    spread_pct: float = 0.0
    rsi_fast: float = 50.0
    macd_histogram: float = 0.0


class ScalpingIndicators:
    """Fast technical indicators for scalping."""

    def __init__(
        self,
        *,
        vwap_std_entry: float = 1.5,      # Enter when price is X std from VWAP
        vwap_std_exit: float = 0.5,       # Exit when price returns to X std
        momentum_period: int = 5,          # Fast momentum lookback
        rsi_period: int = 7,              # Fast RSI
        rsi_oversold: int = 25,           # More extreme for scalping
        rsi_overbought: int = 75,
        volume_spike_mult: float = 2.0,   # Volume must be 2x average
        min_spread_pct: float = 0.05,     # Minimum spread to trade
        max_spread_pct: float = 0.50,     # Maximum spread (avoid illiquid)
        profit_target_pct: float = 0.75,  # 0.75% profit target
        stop_loss_pct: float = 0.50,      # 0.50% stop loss (1.5:1 R:R, wider for execution)
    ):
        self.vwap_std_entry = vwap_std_entry
        self.vwap_std_exit = vwap_std_exit
        self.momentum_period = momentum_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_spike_mult = volume_spike_mult
        self.min_spread_pct = min_spread_pct
        self.max_spread_pct = max_spread_pct
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct

    # ── VWAP Calculation ─────────────────────────────────────────

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        return vwap

    @staticmethod
    def calculate_vwap_bands(df: pd.DataFrame, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate VWAP with standard deviation bands."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

        # Rolling standard deviation of price from VWAP
        squared_diff = ((typical_price - vwap) ** 2 * df["volume"]).cumsum() / df["volume"].cumsum()
        std = np.sqrt(squared_diff)

        upper_band = vwap + (std * num_std)
        lower_band = vwap - (std * num_std)

        return vwap, upper_band, lower_band

    # ── Fast Momentum ────────────────────────────────────────────

    def calculate_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate fast momentum (rate of change)."""
        return df["close"].pct_change(self.momentum_period) * 100

    @staticmethod
    def calculate_roc(df: pd.DataFrame, period: int = 3) -> pd.Series:
        """Rate of Change - very fast momentum."""
        return ((df["close"] - df["close"].shift(period)) / df["close"].shift(period)) * 100

    # ── Fast RSI ─────────────────────────────────────────────────

    def calculate_fast_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate fast RSI for scalping."""
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # ── MACD for Scalping ────────────────────────────────────────

    @staticmethod
    def calculate_fast_macd(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Fast MACD (5, 13, 4) for scalping."""
        ema_fast = df["close"].ewm(span=5, adjust=False).mean()
        ema_slow = df["close"].ewm(span=13, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=4, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    # ── Volume Analysis ──────────────────────────────────────────

    def detect_volume_spike(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Detect volume spikes relative to recent average."""
        avg_volume = df["volume"].rolling(lookback).mean()
        volume_ratio = df["volume"] / avg_volume
        return volume_ratio

    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume for momentum confirmation."""
        obv = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
        return obv

    # ── Price Action ─────────────────────────────────────────────

    @staticmethod
    def detect_engulfing(df: pd.DataFrame) -> pd.Series:
        """Detect bullish/bearish engulfing patterns."""
        body_prev = df["close"].shift(1) - df["open"].shift(1)
        body_curr = df["close"] - df["open"]

        # Bullish engulfing: previous red, current green engulfs
        bullish = (body_prev < 0) & (body_curr > 0) & \
                  (df["open"] <= df["close"].shift(1)) & \
                  (df["close"] >= df["open"].shift(1))

        # Bearish engulfing: previous green, current red engulfs
        bearish = (body_prev > 0) & (body_curr < 0) & \
                  (df["open"] >= df["close"].shift(1)) & \
                  (df["close"] <= df["open"].shift(1))

        result = pd.Series(0, index=df.index)
        result[bullish] = 1
        result[bearish] = -1
        return result

    @staticmethod
    def detect_pin_bar(df: pd.DataFrame, wick_ratio: float = 2.5) -> pd.Series:
        """Detect pin bars (rejection candles)."""
        body = abs(df["close"] - df["open"])
        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

        # Bullish pin bar: long lower wick
        bullish = (lower_wick > body * wick_ratio) & (upper_wick < body)

        # Bearish pin bar: long upper wick
        bearish = (upper_wick > body * wick_ratio) & (lower_wick < body)

        result = pd.Series(0, index=df.index)
        result[bullish] = 1
        result[bearish] = -1
        return result

    # ── Main Analysis ────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> ScalpSignal:
        """Run full scalping analysis on price data.

        Args:
            df: DataFrame with OHLCV data (1-minute bars recommended)
            symbol: Stock symbol

        Returns:
            ScalpSignal with entry/exit recommendations
        """
        if len(df) < 30:
            return ScalpSignal(
                symbol=symbol,
                signal="HOLD",
                strength=0.0,
                entry_price=0.0,
                target_price=0.0,
                stop_loss=0.0,
                confidence=0.0,
                reasons=["Insufficient data for scalping analysis"],
            )

        current_price = float(df["close"].iloc[-1])
        reasons = []
        signals = []  # List of (signal, weight) tuples

        # ── VWAP Analysis ────────────────────────────────────────
        vwap, upper_band, lower_band = self.calculate_vwap_bands(df)
        current_vwap = float(vwap.iloc[-1])
        vwap_std = float((upper_band.iloc[-1] - current_vwap) / 2) if upper_band.iloc[-1] != current_vwap else current_price * 0.01
        vwap_deviation = (current_price - current_vwap) / vwap_std if vwap_std > 0 else 0

        if vwap_deviation < -self.vwap_std_entry:
            signals.append(("BUY", 0.3))
            reasons.append(f"Price {abs(vwap_deviation):.1f} std below VWAP (mean reversion)")
        elif vwap_deviation > self.vwap_std_entry:
            signals.append(("SELL", 0.3))
            reasons.append(f"Price {vwap_deviation:.1f} std above VWAP (mean reversion)")

        # ── Momentum ─────────────────────────────────────────────
        momentum = self.calculate_momentum(df)
        current_momentum = float(momentum.iloc[-1]) if not pd.isna(momentum.iloc[-1]) else 0

        if current_momentum > 0.3:  # Strong upward momentum
            signals.append(("BUY", 0.25))
            reasons.append(f"Strong momentum: +{current_momentum:.2f}%")
        elif current_momentum < -0.3:  # Strong downward momentum
            signals.append(("SELL", 0.25))
            reasons.append(f"Weak momentum: {current_momentum:.2f}%")

        # ── Fast RSI ─────────────────────────────────────────────
        rsi = self.calculate_fast_rsi(df)
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50

        if current_rsi < self.rsi_oversold:
            signals.append(("BUY", 0.2))
            reasons.append(f"RSI oversold: {current_rsi:.0f}")
        elif current_rsi > self.rsi_overbought:
            signals.append(("SELL", 0.2))
            reasons.append(f"RSI overbought: {current_rsi:.0f}")

        # ── MACD Histogram ───────────────────────────────────────
        _, _, histogram = self.calculate_fast_macd(df)
        current_hist = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0
        prev_hist = float(histogram.iloc[-2]) if len(histogram) > 1 and not pd.isna(histogram.iloc[-2]) else 0

        # MACD histogram turning positive
        if current_hist > 0 and prev_hist <= 0:
            signals.append(("BUY", 0.15))
            reasons.append("MACD histogram turned positive")
        elif current_hist < 0 and prev_hist >= 0:
            signals.append(("SELL", 0.15))
            reasons.append("MACD histogram turned negative")

        # ── Volume Spike ─────────────────────────────────────────
        volume_ratio = self.detect_volume_spike(df)
        current_vol_ratio = float(volume_ratio.iloc[-1]) if not pd.isna(volume_ratio.iloc[-1]) else 1

        if current_vol_ratio >= self.volume_spike_mult:
            # Volume spike - amplify signal
            reasons.append(f"Volume spike: {current_vol_ratio:.1f}x average")

        # ── Price Action Patterns ────────────────────────────────
        engulfing = self.detect_engulfing(df)
        pin_bar = self.detect_pin_bar(df)

        if int(engulfing.iloc[-1]) == 1:
            signals.append(("BUY", 0.1))
            reasons.append("Bullish engulfing pattern")
        elif int(engulfing.iloc[-1]) == -1:
            signals.append(("SELL", 0.1))
            reasons.append("Bearish engulfing pattern")

        if int(pin_bar.iloc[-1]) == 1:
            signals.append(("BUY", 0.1))
            reasons.append("Bullish pin bar rejection")
        elif int(pin_bar.iloc[-1]) == -1:
            signals.append(("SELL", 0.1))
            reasons.append("Bearish pin bar rejection")

        # ── Aggregate Signals ────────────────────────────────────
        buy_weight = sum(w for sig, w in signals if sig == "BUY")
        sell_weight = sum(w for sig, w in signals if sig == "SELL")

        # Apply volume multiplier
        if current_vol_ratio >= self.volume_spike_mult:
            buy_weight *= 1.2
            sell_weight *= 1.2

        # Determine final signal
        if buy_weight > sell_weight and buy_weight >= 0.3:
            final_signal = "BUY"
            strength = min(buy_weight, 1.0)
            target = current_price * (1 + self.profit_target_pct / 100)
            stop = current_price * (1 - self.stop_loss_pct / 100)
        elif sell_weight > buy_weight and sell_weight >= 0.3:
            final_signal = "SELL"
            strength = -min(sell_weight, 1.0)
            target = current_price * (1 - self.profit_target_pct / 100)
            stop = current_price * (1 + self.stop_loss_pct / 100)
        else:
            final_signal = "HOLD"
            strength = 0.0
            target = current_price
            stop = current_price

        confidence = abs(buy_weight - sell_weight) / max(buy_weight + sell_weight, 0.01)

        return ScalpSignal(
            symbol=symbol,
            signal=final_signal,
            strength=strength,
            entry_price=current_price,
            target_price=round(target, 2),
            stop_loss=round(stop, 2),
            confidence=round(confidence, 2),
            reasons=reasons if reasons else ["No strong signals"],
            vwap=round(current_vwap, 2),
            vwap_deviation=round(vwap_deviation, 2),
            momentum=round(current_momentum, 2),
            volume_ratio=round(current_vol_ratio, 2),
            rsi_fast=round(current_rsi, 1),
            macd_histogram=round(current_hist, 4),
        )

    # ── Formatting ───────────────────────────────────────────────

    @staticmethod
    def format_signal(sig: ScalpSignal) -> str:
        """Format scalp signal for display."""
        emoji = {"BUY": "+", "SELL": "-", "HOLD": "="}[sig.signal]

        lines = [
            f"\n{'~' * 60}",
            f"  SCALP SIGNAL: {sig.symbol}  [{emoji} {sig.signal}]",
            f"{'~' * 60}",
            f"  Entry: ${sig.entry_price:.2f}  |  Target: ${sig.target_price:.2f}  |  Stop: ${sig.stop_loss:.2f}",
            f"  Strength: {sig.strength:+.2f}  |  Confidence: {sig.confidence:.0%}",
            f"  ──────────────────────────────────────────────────────────",
            f"  VWAP: ${sig.vwap:.2f} (deviation: {sig.vwap_deviation:+.1f} std)",
            f"  Momentum: {sig.momentum:+.2f}%  |  RSI: {sig.rsi_fast:.0f}  |  Vol Ratio: {sig.volume_ratio:.1f}x",
            f"  ──────────────────────────────────────────────────────────",
            f"  Reasons:",
        ]
        for reason in sig.reasons:
            lines.append(f"    - {reason}")
        lines.append(f"{'~' * 60}\n")

        return "\n".join(lines)
