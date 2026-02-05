"""Walk-Forward Optimization

Optimises strategy parameters using rolling in-sample / out-of-sample
windows on historical bar data.  Prevents overfitting by requiring
each parameter set to prove itself on unseen data.

Process:
    1. Split historical bars into rolling windows:
       [  in-sample  ] [ out-of-sample ]
                    [  in-sample  ] [ out-of-sample ]
                                 [  in-sample  ] [ out-of-sample ]
    2. For each window, grid-search parameter combos on in-sample data.
    3. Pick the best combo and evaluate it on out-of-sample data.
    4. Aggregate out-of-sample performance to get a realistic estimate.

Usage:
    wfo = WalkForwardOptimizer()
    result = wfo.optimize(
        bars=df,
        param_grid={"buy_threshold": [1, 2, 3], "sell_threshold": [-1, -2, -3]},
    )
    print(WalkForwardOptimizer.format_result(result))
"""

import logging
import math
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Result from one in-sample / out-of-sample window."""
    window_id: int = 0
    is_start: int = 0
    is_end: int = 0
    oos_start: int = 0
    oos_end: int = 0
    best_params: dict = field(default_factory=dict)
    is_return_pct: float = 0.0      # in-sample return
    oos_return_pct: float = 0.0     # out-of-sample return
    oos_trades: int = 0
    oos_win_rate: float = 0.0
    oos_sharpe: float = 0.0


@dataclass
class WFOResult:
    """Aggregated walk-forward optimization result."""
    windows: list[WindowResult] = field(default_factory=list)
    total_oos_return_pct: float = 0.0
    avg_oos_return_pct: float = 0.0
    avg_oos_sharpe: float = 0.0
    avg_oos_win_rate: float = 0.0
    total_oos_trades: int = 0
    best_params_frequency: dict = field(default_factory=dict)
    most_robust_params: dict = field(default_factory=dict)
    param_grid: dict = field(default_factory=dict)
    num_windows: int = 0
    is_ratio: float = 0.0
    oos_ratio: float = 0.0


class WalkForwardOptimizer:
    """Walk-forward parameter optimization for trading strategies."""

    def __init__(
        self,
        *,
        is_ratio: float = 0.70,
        num_windows: int = 5,
        min_trades: int = 3,
        risk_per_trade: float = 0.02,
        atr_stop_mult: float = 1.5,
        take_profit_ratio: float = 2.0,
    ):
        """
        Args:
            is_ratio: Fraction of each window used for in-sample (0.7 = 70%).
            num_windows: Number of rolling windows.
            min_trades: Minimum trades in a window to consider results valid.
            risk_per_trade: Risk per trade for position sizing.
            atr_stop_mult: ATR stop-loss multiplier.
            take_profit_ratio: Take-profit to risk ratio.
        """
        self.is_ratio = is_ratio
        self.oos_ratio = 1.0 - is_ratio
        self.num_windows = num_windows
        self.min_trades = min_trades
        self.risk_per_trade = risk_per_trade
        self.atr_stop_mult = atr_stop_mult
        self.take_profit_ratio = take_profit_ratio

    # ── Signal generation ────────────────────────────────────────

    @staticmethod
    def compute_signals(
        bars: pd.DataFrame,
        buy_threshold: int = 2,
        sell_threshold: int = -2,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        ema_short: int = 9,
        ema_long: int = 21,
    ) -> pd.Series:
        """Compute TA composite score for each bar.

        Uses the same indicators as TechnicalAnalysisAgent but parametrised.
        Returns a Series of integer scores.
        """
        import ta as ta_lib

        df = bars.copy()
        close = df["close"]

        # RSI
        rsi = ta_lib.momentum.RSIIndicator(close, window=14).rsi()
        rsi_score = pd.Series(0, index=df.index)
        rsi_score[rsi < rsi_oversold] = 1
        rsi_score[rsi > rsi_overbought] = -1

        # EMA crossover
        ema_s = close.ewm(span=ema_short, adjust=False).mean()
        ema_l = close.ewm(span=ema_long, adjust=False).mean()
        ema_score = pd.Series(0, index=df.index)
        ema_score[ema_s > ema_l] = 1
        ema_score[ema_s < ema_l] = -1

        # MACD
        macd = ta_lib.trend.MACD(close)
        macd_diff = macd.macd_diff()
        macd_score = pd.Series(0, index=df.index)
        macd_score[macd_diff > 0] = 1
        macd_score[macd_diff < 0] = -1

        # Bollinger Bands
        bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_score = pd.Series(0, index=df.index)
        bb_score[close < bb.bollinger_lband()] = 1
        bb_score[close > bb.bollinger_hband()] = -1

        # Stochastic
        stoch = ta_lib.momentum.StochasticOscillator(
            df["high"], df["low"], close, window=14, smooth_window=3,
        )
        stoch_k = stoch.stoch()
        stoch_score = pd.Series(0, index=df.index)
        stoch_score[stoch_k < 20] = 1
        stoch_score[stoch_k > 80] = -1

        composite = rsi_score + ema_score + macd_score + bb_score + stoch_score
        return composite

    # ── Backtest one window ──────────────────────────────────────

    def backtest_window(
        self,
        bars: pd.DataFrame,
        buy_threshold: int = 2,
        sell_threshold: int = -2,
        **signal_params,
    ) -> dict:
        """Run a simple backtest on a bar DataFrame.

        Returns dict with: return_pct, trades, win_rate, sharpe.
        """
        if len(bars) < 30:
            return {"return_pct": 0, "trades": 0, "win_rate": 0, "sharpe": 0}

        scores = self.compute_signals(bars, buy_threshold=buy_threshold,
                                      sell_threshold=sell_threshold, **signal_params)

        # ATR for stops
        import ta as ta_lib
        atr = ta_lib.volatility.AverageTrueRange(
            bars["high"], bars["low"], bars["close"], window=14,
        ).average_true_range()

        capital = 100000.0
        equity = capital
        position = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        trades = []
        equity_curve = [equity]

        for i in range(26, len(bars)):
            price = bars["close"].iloc[i]
            cur_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0

            # Check exit conditions
            if position > 0:
                if price <= stop_loss or price >= take_profit:
                    pnl = (price - entry_price) * position
                    equity += pnl
                    trades.append(pnl)
                    position = 0

            score = scores.iloc[i]

            # Entry signals
            if position == 0 and cur_atr > 0:
                risk_amount = equity * self.risk_per_trade
                stop_dist = cur_atr * self.atr_stop_mult

                if score >= buy_threshold and stop_dist > 0:
                    shares = int(risk_amount / stop_dist)
                    if shares > 0 and shares * price < equity * 0.10:
                        position = shares
                        entry_price = price
                        stop_loss = price - stop_dist
                        take_profit = price + stop_dist * self.take_profit_ratio

                elif score <= sell_threshold and position > 0:
                    pnl = (price - entry_price) * position
                    equity += pnl
                    trades.append(pnl)
                    position = 0

            equity_curve.append(equity + (price - entry_price) * position if position > 0 else equity)

        # Close any remaining position
        if position > 0:
            price = bars["close"].iloc[-1]
            pnl = (price - entry_price) * position
            equity += pnl
            trades.append(pnl)

        ret_pct = (equity - capital) / capital * 100
        wins = sum(1 for t in trades if t > 0)
        win_rate = wins / len(trades) if trades else 0

        # Sharpe
        sharpe = 0.0
        if len(equity_curve) > 2:
            daily_ret = []
            for i in range(1, len(equity_curve)):
                if equity_curve[i - 1] > 0:
                    daily_ret.append(
                        (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                    )
            if daily_ret:
                mean_r = sum(daily_ret) / len(daily_ret)
                if len(daily_ret) > 1:
                    var = sum((r - mean_r) ** 2 for r in daily_ret) / (len(daily_ret) - 1)
                    std_r = math.sqrt(var)
                    if std_r > 0:
                        sharpe = mean_r / std_r * math.sqrt(252)

        return {
            "return_pct": ret_pct,
            "trades": len(trades),
            "win_rate": win_rate,
            "sharpe": sharpe,
        }

    # ── Main optimize ────────────────────────────────────────────

    def optimize(
        self,
        bars: pd.DataFrame,
        param_grid: dict[str, list] | None = None,
    ) -> WFOResult:
        """Run walk-forward optimization.

        Args:
            bars: Historical OHLCV DataFrame.
            param_grid: Dict of param_name → list of values to search.

        Returns:
            WFOResult with per-window and aggregate results.
        """
        if param_grid is None:
            param_grid = {
                "buy_threshold": [1, 2, 3],
                "sell_threshold": [-1, -2, -3],
            }

        total_bars = len(bars)
        if total_bars < 60:
            logger.warning("Not enough bars for WFO (%d < 60)", total_bars)
            return WFOResult(param_grid=param_grid)

        # Window step calculation
        window_size = total_bars // self.num_windows
        if window_size < 30:
            self.num_windows = max(2, total_bars // 30)
            window_size = total_bars // self.num_windows

        is_size = int(window_size * self.is_ratio)
        oos_size = window_size - is_size

        # Build parameter combos
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combos = [dict(zip(param_names, v)) for v in product(*param_values)]

        windows: list[WindowResult] = []
        param_counts: dict[str, int] = {}

        for w in range(self.num_windows):
            start = w * window_size
            is_end = start + is_size
            oos_end = min(start + window_size, total_bars)

            if is_end >= total_bars or oos_end <= is_end:
                break

            is_bars = bars.iloc[start:is_end]
            oos_bars = bars.iloc[is_end:oos_end]

            if len(is_bars) < 30 or len(oos_bars) < 10:
                continue

            # Grid search on in-sample
            best_combo = combos[0]
            best_metric = -float("inf")

            for combo in combos:
                result = self.backtest_window(is_bars, **combo)
                # Optimise for Sharpe (or return if no trades)
                metric = result["sharpe"] if result["trades"] >= self.min_trades else -999
                if metric > best_metric:
                    best_metric = metric
                    best_combo = combo

            # Evaluate best params on out-of-sample
            is_result = self.backtest_window(is_bars, **best_combo)
            oos_result = self.backtest_window(oos_bars, **best_combo)

            wr = WindowResult(
                window_id=w,
                is_start=start,
                is_end=is_end,
                oos_start=is_end,
                oos_end=oos_end,
                best_params=best_combo,
                is_return_pct=is_result["return_pct"],
                oos_return_pct=oos_result["return_pct"],
                oos_trades=oos_result["trades"],
                oos_win_rate=oos_result["win_rate"],
                oos_sharpe=oos_result["sharpe"],
            )
            windows.append(wr)

            # Track param frequency
            key = str(sorted(best_combo.items()))
            param_counts[key] = param_counts.get(key, 0) + 1

        # Aggregate
        result = WFOResult(
            windows=windows,
            param_grid=param_grid,
            num_windows=len(windows),
            is_ratio=self.is_ratio,
            oos_ratio=self.oos_ratio,
        )

        if windows:
            result.total_oos_return_pct = sum(w.oos_return_pct for w in windows)
            result.avg_oos_return_pct = result.total_oos_return_pct / len(windows)
            result.avg_oos_sharpe = sum(w.oos_sharpe for w in windows) / len(windows)
            oos_with_trades = [w for w in windows if w.oos_trades > 0]
            result.avg_oos_win_rate = (
                sum(w.oos_win_rate for w in oos_with_trades) / len(oos_with_trades)
                if oos_with_trades else 0
            )
            result.total_oos_trades = sum(w.oos_trades for w in windows)
            result.best_params_frequency = param_counts

            # Most robust = most frequently chosen
            if param_counts:
                best_key = max(param_counts, key=param_counts.get)
                result.most_robust_params = dict(eval(best_key))

        return result

    # ── Formatting ───────────────────────────────────────────────

    @staticmethod
    def format_result(result: WFOResult) -> str:
        lines = [
            f"\n{'=' * 70}",
            f"  WALK-FORWARD OPTIMIZATION",
            f"{'=' * 70}",
            f"  Windows        : {result.num_windows}",
            f"  IS / OOS Ratio : {result.is_ratio:.0%} / {result.oos_ratio:.0%}",
            f"  Param Grid     : {result.param_grid}",
        ]

        # Per-window
        if result.windows:
            lines.append(f"\n  {'Win':>4s} {'IS Ret':>8s} {'OOS Ret':>8s} "
                        f"{'Trades':>7s} {'Win%':>6s} {'Sharpe':>7s} {'Params'}")
            lines.append(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*7} {'-'*20}")

            for w in result.windows:
                params_str = ", ".join(f"{k}={v}" for k, v in w.best_params.items())
                lines.append(
                    f"  {w.window_id:>4d} {w.is_return_pct:>+7.2f}% "
                    f"{w.oos_return_pct:>+7.2f}% {w.oos_trades:>7d} "
                    f"{w.oos_win_rate:>5.1%} {w.oos_sharpe:>7.2f} {params_str}"
                )

        # Aggregate
        lines.append(f"\n  {'~' * 66}")
        lines.append(f"  Avg OOS Return  : {result.avg_oos_return_pct:+.2f}%")
        lines.append(f"  Avg OOS Sharpe  : {result.avg_oos_sharpe:.3f}")
        lines.append(f"  Avg OOS Win Rate: {result.avg_oos_win_rate:.1%}")
        lines.append(f"  Total OOS Trades: {result.total_oos_trades}")

        if result.most_robust_params:
            lines.append(f"\n  Most Robust Parameters:")
            for k, v in result.most_robust_params.items():
                lines.append(f"    {k}: {v}")

        lines.append(f"\n{'=' * 70}\n")
        return "\n".join(lines)
