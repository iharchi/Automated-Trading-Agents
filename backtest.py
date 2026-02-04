"""Backtesting Engine

Walk-forward simulation that replays historical bars through the
Technical Analysis indicator logic, simulates trades with ATR-based
position sizing and stop-loss / take-profit exits, and produces
performance metrics.

Usage:
    python backtest.py AAPL                     # 1 year, daily bars
    python backtest.py AAPL --days 730          # 2 years
    python backtest.py AAPL --initial-capital 50000

Metrics reported:
    - Total return (%)
    - Annualised return (%)
    - Max drawdown (%)
    - Sharpe ratio (annualised, assuming 252 trading days)
    - Win rate (%)
    - Profit factor (gross wins / gross losses)
    - Total trades, avg win, avg loss

Note: Sentiment is excluded because bar-by-bar historical news
      data is not available for replay.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ta as ta_lib

from utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

# ── TA parameters (mirrored from TechnicalAnalysisAgent) ─────────

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
EMA_SHORT = 9
EMA_LONG = 21
BB_PERIOD = 20
BB_STD = 2
BUY_THRESHOLD = 2
SELL_THRESHOLD = -2

# ── Risk parameters (mirrored from RiskManagementAgent) ──────────

RISK_PER_TRADE_PCT = 0.02
ATR_STOP_MULTIPLIER = 1.5
TAKE_PROFIT_RATIO = 2.0

# Minimum lookback needed before first signal
MIN_LOOKBACK = 26  # MACD slow (26) is the longest lookback


@dataclass
class Trade:
    """Record of one completed round-trip trade."""

    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    shares: int
    side: str  # "long"
    pnl: float
    pnl_pct: float
    exit_reason: str  # "signal", "stop_loss", "take_profit"


@dataclass
class BacktestResult:
    """Full backtest output."""

    symbol: str
    period: str = ""
    initial_capital: float = 100_000
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    annualised_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


class Backtester:
    """Walk-forward backtesting engine using TA signals."""

    def __init__(
        self,
        initial_capital: float = 100_000,
        risk_per_trade: float = RISK_PER_TRADE_PCT,
        atr_stop_mult: float = ATR_STOP_MULTIPLIER,
        tp_ratio: float = TAKE_PROFIT_RATIO,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.atr_stop_mult = atr_stop_mult
        self.tp_ratio = tp_ratio

    # ── Indicator computation (vectorised) ───────────────────

    @staticmethod
    def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to the bars DataFrame."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI
        df["rsi"] = ta_lib.momentum.RSIIndicator(close, window=RSI_PERIOD).rsi()

        # MACD
        macd = ta_lib.trend.MACD(close)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # Bollinger Bands
        bb = ta_lib.volatility.BollingerBands(close, window=BB_PERIOD, window_dev=BB_STD)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()

        # EMA crossover
        df["ema_short"] = ta_lib.trend.EMAIndicator(close, window=EMA_SHORT).ema_indicator()
        df["ema_long"] = ta_lib.trend.EMAIndicator(close, window=EMA_LONG).ema_indicator()

        # ATR
        df["atr"] = (
            ta_lib.volatility.AverageTrueRange(high, low, close, window=RSI_PERIOD)
            .average_true_range()
        )

        return df

    @staticmethod
    def _score_bar(row: pd.Series) -> int:
        """Return composite score for a single bar's indicators."""
        score = 0

        # RSI
        if row["rsi"] <= RSI_OVERSOLD:
            score += 1
        elif row["rsi"] >= RSI_OVERBOUGHT:
            score -= 1

        # MACD
        if row["macd"] > row["macd_signal"] and row["macd_hist"] > 0:
            score += 1
        elif row["macd"] < row["macd_signal"] and row["macd_hist"] < 0:
            score -= 1

        # Bollinger
        if row["close"] <= row["bb_lower"]:
            score += 1
        elif row["close"] >= row["bb_upper"]:
            score -= 1

        # EMA crossover
        if row["ema_short"] > row["ema_long"]:
            score += 1
        elif row["ema_short"] < row["ema_long"]:
            score -= 1

        return score

    # ── Position sizing ──────────────────────────────────────

    def _size_position(self, equity: float, price: float, atr: float) -> int:
        """ATR-based position sizing matching the Risk Management Agent."""
        if price <= 0 or atr <= 0:
            return 0
        risk_budget = equity * self.risk_per_trade
        stop_distance = atr * self.atr_stop_mult
        if stop_distance <= 0:
            return 0
        return max(int(risk_budget / stop_distance), 0)

    # ── Core simulation ──────────────────────────────────────

    def run(self, symbol: str, days: int = 365) -> BacktestResult:
        """Fetch historical data and run the walk-forward simulation."""
        client = AlpacaClient()

        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")

        bars = client.get_bars(symbol, timeframe="1Day", limit=days + 50, start=start, end=end)
        if bars.empty or len(bars) < MIN_LOOKBACK + 10:
            logger.error("Insufficient data for %s (%d bars)", symbol, len(bars))
            return BacktestResult(symbol=symbol, initial_capital=self.initial_capital)

        df = self._compute_indicators(bars.copy())
        df = df.dropna().reset_index()

        return self._simulate(symbol, df, days)

    def run_on_dataframe(self, symbol: str, df: pd.DataFrame) -> BacktestResult:
        """Run simulation on a pre-loaded DataFrame (for testing)."""
        df = self._compute_indicators(df.copy())
        df = df.dropna().reset_index()
        days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days if "timestamp" in df.columns else len(df)
        return self._simulate(symbol, df, days)

    def _simulate(self, symbol: str, df: pd.DataFrame, days: int) -> BacktestResult:
        """Walk forward through bars, trading on signals."""
        equity = self.initial_capital
        cash = self.initial_capital
        position_shares = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        entry_date = ""

        trades: list[Trade] = []
        equity_curve: list[float] = [equity]

        for i in range(len(df)):
            row = df.iloc[i]
            price = row["close"]
            atr = row["atr"]
            date_str = str(row.get("timestamp", i))

            # ── Check stop-loss / take-profit on open positions ──
            if position_shares > 0:
                # Use high/low to check intrabar triggers
                if row["low"] <= stop_loss:
                    # Stop-loss hit
                    exit_price = stop_loss
                    pnl = (exit_price - entry_price) * position_shares
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    cash += position_shares * exit_price
                    trades.append(Trade(
                        entry_date=entry_date, entry_price=entry_price,
                        exit_date=date_str, exit_price=exit_price,
                        shares=position_shares, side="long",
                        pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 2),
                        exit_reason="stop_loss",
                    ))
                    position_shares = 0
                elif row["high"] >= take_profit:
                    # Take-profit hit
                    exit_price = take_profit
                    pnl = (exit_price - entry_price) * position_shares
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    cash += position_shares * exit_price
                    trades.append(Trade(
                        entry_date=entry_date, entry_price=entry_price,
                        exit_date=date_str, exit_price=exit_price,
                        shares=position_shares, side="long",
                        pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 2),
                        exit_reason="take_profit",
                    ))
                    position_shares = 0

            # ── Generate signal ──────────────────────────────────
            score = self._score_bar(row)

            if score >= BUY_THRESHOLD and position_shares == 0:
                # BUY: enter long
                shares = self._size_position(cash, price, atr)
                cost = shares * price
                if shares > 0 and cost <= cash:
                    position_shares = shares
                    entry_price = price
                    entry_date = date_str
                    stop_distance = atr * self.atr_stop_mult
                    stop_loss = round(price - stop_distance, 2)
                    take_profit = round(price + stop_distance * self.tp_ratio, 2)
                    cash -= cost

            elif score <= SELL_THRESHOLD and position_shares > 0:
                # SELL signal: close position
                exit_price = price
                pnl = (exit_price - entry_price) * position_shares
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                cash += position_shares * exit_price
                trades.append(Trade(
                    entry_date=entry_date, entry_price=entry_price,
                    exit_date=date_str, exit_price=exit_price,
                    shares=position_shares, side="long",
                    pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 2),
                    exit_reason="signal",
                ))
                position_shares = 0

            # ── Track equity ─────────────────────────────────────
            mark_to_market = cash + (position_shares * price)
            equity_curve.append(round(mark_to_market, 2))

        # ── Close any open position at the last price ────────
        if position_shares > 0:
            final_price = df.iloc[-1]["close"]
            pnl = (final_price - entry_price) * position_shares
            pnl_pct = (final_price - entry_price) / entry_price * 100
            cash += position_shares * final_price
            trades.append(Trade(
                entry_date=entry_date, entry_price=entry_price,
                exit_date=str(df.iloc[-1].get("timestamp", len(df) - 1)),
                exit_price=final_price,
                shares=position_shares, side="long",
                pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 2),
                exit_reason="end_of_data",
            ))
            position_shares = 0

        final_equity = cash
        return self._build_result(symbol, days, final_equity, trades, equity_curve)

    # ── Metrics ──────────────────────────────────────────────

    def _build_result(
        self,
        symbol: str,
        days: int,
        final_equity: float,
        trades: list[Trade],
        equity_curve: list[float],
    ) -> BacktestResult:
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        years = max(days / 365.25, 0.01)
        ann_return = ((final_equity / self.initial_capital) ** (1 / years) - 1) * 100

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (daily returns, annualised)
        if len(equity_curve) > 2:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * math.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Win/loss stats
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0.0

        gross_wins = sum(t.pnl for t in wins)
        gross_losses = abs(sum(t.pnl for t in losses))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        avg_win = gross_wins / len(wins) if wins else 0.0
        avg_loss = gross_losses / len(losses) if losses else 0.0

        return BacktestResult(
            symbol=symbol,
            period=f"{days} days",
            initial_capital=self.initial_capital,
            final_equity=round(final_equity, 2),
            total_return_pct=round(total_return, 2),
            annualised_return_pct=round(ann_return, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 3),
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate_pct=round(win_rate, 1),
            profit_factor=round(profit_factor, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            trades=trades,
            equity_curve=equity_curve,
        )

    # ── Pretty print ─────────────────────────────────────────

    @staticmethod
    def format_result(r: BacktestResult) -> str:
        lines = [
            f"\n{'#' * 62}",
            f"  BACKTEST RESULTS — {r.symbol}  ({r.period})",
            f"{'#' * 62}",
            f"",
            f"  Capital        : ${r.initial_capital:>12,.2f}",
            f"  Final Equity   : ${r.final_equity:>12,.2f}",
            f"  Total Return   : {r.total_return_pct:>+11.2f}%",
            f"  Annual Return  : {r.annualised_return_pct:>+11.2f}%",
            f"  Max Drawdown   : {r.max_drawdown_pct:>11.2f}%",
            f"  Sharpe Ratio   : {r.sharpe_ratio:>11.3f}",
            f"",
            f"{'─' * 62}",
            f"  Trades         : {r.total_trades:>6d}",
            f"  Winners        : {r.winning_trades:>6d}   ({r.win_rate_pct:.1f}%)",
            f"  Losers         : {r.losing_trades:>6d}",
            f"  Avg Win        : ${r.avg_win:>10,.2f}",
            f"  Avg Loss       : ${r.avg_loss:>10,.2f}",
            f"  Profit Factor  : {r.profit_factor:>10.2f}",
            f"{'─' * 62}",
        ]

        if r.trades:
            lines.append(f"")
            lines.append(f"  {'Date':20s} {'Side':6s} {'Shares':>6s} {'Entry':>9s} {'Exit':>9s} {'P&L':>10s} {'Reason'}")
            lines.append(f"  {'─' * 70}")
            for t in r.trades:
                entry_short = t.entry_date[:10] if len(t.entry_date) > 10 else t.entry_date
                lines.append(
                    f"  {entry_short:20s} {t.side:6s} {t.shares:>6d} "
                    f"${t.entry_price:>8.2f} ${t.exit_price:>8.2f} "
                    f"${t.pnl:>+9.2f} {t.exit_reason}"
                )

        lines.append(f"\n{'#' * 62}\n")
        return "\n".join(lines)
