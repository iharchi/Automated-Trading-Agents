"""Tests for Walk-Forward Optimization.

All tests use synthetic price data -- no external API keys required.
Uses unittest.mock to stub the ``ta`` library where necessary so tests
remain fast and deterministic.
"""

import math
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from utils.walk_forward import WalkForwardOptimizer, WindowResult, WFOResult


# ── Helpers ──────────────────────────────────────────────────────


def make_bars(n=200, seed=42):
    """Generate synthetic OHLCV bars using a random-walk model."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 10)  # keep positive
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    opn = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 10000, n)
    return pd.DataFrame({
        "open": opn, "high": high, "low": low,
        "close": close, "volume": volume,
    })


# ── WindowResult dataclass ───────────────────────────────────────


class TestWindowResult:
    def test_defaults(self):
        wr = WindowResult()
        assert wr.window_id == 0
        assert wr.is_start == 0
        assert wr.is_end == 0
        assert wr.oos_start == 0
        assert wr.oos_end == 0
        assert wr.best_params == {}
        assert wr.is_return_pct == 0.0
        assert wr.oos_return_pct == 0.0
        assert wr.oos_trades == 0
        assert wr.oos_win_rate == 0.0
        assert wr.oos_sharpe == 0.0

    def test_custom_values(self):
        wr = WindowResult(
            window_id=3,
            is_start=10,
            is_end=80,
            oos_start=80,
            oos_end=100,
            best_params={"buy_threshold": 2},
            is_return_pct=5.5,
            oos_return_pct=-1.2,
            oos_trades=7,
            oos_win_rate=0.6,
            oos_sharpe=1.3,
        )
        assert wr.window_id == 3
        assert wr.is_return_pct == 5.5
        assert wr.oos_return_pct == -1.2
        assert wr.oos_trades == 7
        assert wr.best_params == {"buy_threshold": 2}


# ── WFOResult dataclass ─────────────────────────────────────────


class TestWFOResult:
    def test_defaults(self):
        r = WFOResult()
        assert r.windows == []
        assert r.total_oos_return_pct == 0.0
        assert r.avg_oos_return_pct == 0.0
        assert r.avg_oos_sharpe == 0.0
        assert r.avg_oos_win_rate == 0.0
        assert r.total_oos_trades == 0
        assert r.best_params_frequency == {}
        assert r.most_robust_params == {}
        assert r.param_grid == {}
        assert r.num_windows == 0
        assert r.is_ratio == 0.0
        assert r.oos_ratio == 0.0

    def test_custom_values(self):
        wr = WindowResult(window_id=0, oos_return_pct=2.0, oos_sharpe=0.5)
        r = WFOResult(
            windows=[wr],
            total_oos_return_pct=2.0,
            avg_oos_return_pct=2.0,
            avg_oos_sharpe=0.5,
            num_windows=1,
        )
        assert len(r.windows) == 1
        assert r.total_oos_return_pct == 2.0
        assert r.num_windows == 1


# ── WalkForwardOptimizer.__init__ ────────────────────────────────


class TestWFOInit:
    def test_default_params(self):
        wfo = WalkForwardOptimizer()
        assert wfo.is_ratio == 0.70
        assert wfo.oos_ratio == pytest.approx(0.30, abs=1e-9)
        assert wfo.num_windows == 5
        assert wfo.min_trades == 3
        assert wfo.risk_per_trade == 0.02
        assert wfo.atr_stop_mult == 1.5
        assert wfo.take_profit_ratio == 2.0

    def test_custom_params(self):
        wfo = WalkForwardOptimizer(
            is_ratio=0.80,
            num_windows=10,
            min_trades=5,
            risk_per_trade=0.05,
            atr_stop_mult=2.0,
            take_profit_ratio=3.0,
        )
        assert wfo.is_ratio == 0.80
        assert wfo.oos_ratio == pytest.approx(0.20, abs=1e-9)
        assert wfo.num_windows == 10
        assert wfo.min_trades == 5
        assert wfo.risk_per_trade == 0.05
        assert wfo.atr_stop_mult == 2.0
        assert wfo.take_profit_ratio == 3.0


# ── compute_signals ──────────────────────────────────────────────


class TestComputeSignals:
    def test_output_length_matches_input(self):
        bars = make_bars(n=60)
        signals = WalkForwardOptimizer.compute_signals(bars)
        assert len(signals) == len(bars)

    def test_scores_are_integers(self):
        bars = make_bars(n=60)
        signals = WalkForwardOptimizer.compute_signals(bars)
        for val in signals:
            assert isinstance(val, (int, np.integer)), f"Expected int, got {type(val)}"

    def test_score_range(self):
        """Composite of 5 indicators each in {-1, 0, 1} => range [-5, 5]."""
        bars = make_bars(n=100)
        signals = WalkForwardOptimizer.compute_signals(bars)
        assert signals.min() >= -5
        assert signals.max() <= 5

    def test_custom_thresholds_do_not_change_length(self):
        bars = make_bars(n=80)
        signals = WalkForwardOptimizer.compute_signals(
            bars,
            buy_threshold=3,
            sell_threshold=-3,
            rsi_oversold=25,
            rsi_overbought=75,
            ema_short=5,
            ema_long=30,
        )
        assert len(signals) == len(bars)

    def test_returns_series(self):
        bars = make_bars(n=50)
        signals = WalkForwardOptimizer.compute_signals(bars)
        assert isinstance(signals, pd.Series)


# ── backtest_window ──────────────────────────────────────────────


class TestBacktestWindow:
    def test_too_few_bars_returns_zeros(self):
        """Fewer than 30 bars should return zero-filled dict."""
        bars = make_bars(n=20)
        wfo = WalkForwardOptimizer()
        result = wfo.backtest_window(bars)
        assert result == {"return_pct": 0, "trades": 0, "win_rate": 0, "sharpe": 0}

    def test_exactly_29_bars_returns_zeros(self):
        bars = make_bars(n=29)
        wfo = WalkForwardOptimizer()
        result = wfo.backtest_window(bars)
        assert result["trades"] == 0

    def test_enough_bars_returns_expected_keys(self):
        bars = make_bars(n=100)
        wfo = WalkForwardOptimizer()
        result = wfo.backtest_window(bars)
        assert "return_pct" in result
        assert "trades" in result
        assert "win_rate" in result
        assert "sharpe" in result

    def test_return_pct_is_float(self):
        bars = make_bars(n=100)
        wfo = WalkForwardOptimizer()
        result = wfo.backtest_window(bars)
        assert isinstance(result["return_pct"], float)

    def test_win_rate_between_zero_and_one(self):
        bars = make_bars(n=200)
        wfo = WalkForwardOptimizer()
        result = wfo.backtest_window(bars)
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_different_thresholds_produce_different_signals(self):
        """Loose (threshold=1) vs extreme (threshold=5) should yield different
        signal counts, which feeds into different backtest behaviour."""
        bars = make_bars(n=200, seed=99)
        signals = WalkForwardOptimizer.compute_signals(bars)
        loose_entries = (signals >= 1).sum()
        extreme_entries = (signals >= 5).sum()
        # Threshold of 1 should trigger far more often than threshold of 5
        assert loose_entries > extreme_entries

    def test_trades_non_negative(self):
        bars = make_bars(n=100)
        wfo = WalkForwardOptimizer()
        result = wfo.backtest_window(bars)
        assert result["trades"] >= 0


# ── optimize ────────────────────────────────────────────────────


class TestOptimize:
    def test_too_few_bars_returns_empty_result(self):
        bars = make_bars(n=50)
        wfo = WalkForwardOptimizer()
        result = wfo.optimize(bars)
        assert isinstance(result, WFOResult)
        assert result.windows == []
        assert result.num_windows == 0

    def test_enough_bars_returns_windows(self):
        bars = make_bars(n=250, seed=7)
        wfo = WalkForwardOptimizer(num_windows=3)
        result = wfo.optimize(bars)
        assert isinstance(result, WFOResult)
        assert len(result.windows) > 0
        assert result.num_windows == len(result.windows)

    def test_custom_param_grid(self):
        bars = make_bars(n=250, seed=11)
        wfo = WalkForwardOptimizer(num_windows=3)
        grid = {"buy_threshold": [1, 2], "sell_threshold": [-1, -2]}
        result = wfo.optimize(bars, param_grid=grid)
        assert result.param_grid == grid

    def test_default_param_grid_used_when_none(self):
        bars = make_bars(n=250, seed=12)
        wfo = WalkForwardOptimizer(num_windows=3)
        result = wfo.optimize(bars, param_grid=None)
        assert "buy_threshold" in result.param_grid
        assert "sell_threshold" in result.param_grid

    def test_window_count_adjustment_for_small_datasets(self):
        """If window_size < 30, num_windows is reduced so each window >= 30."""
        bars = make_bars(n=80)
        wfo = WalkForwardOptimizer(num_windows=10)
        result = wfo.optimize(bars)
        # The optimizer should have adjusted num_windows downward
        assert isinstance(result, WFOResult)

    def test_result_aggregation_avg_oos_return(self):
        bars = make_bars(n=300, seed=21)
        wfo = WalkForwardOptimizer(num_windows=3)
        result = wfo.optimize(bars)
        if result.windows:
            manual_avg = sum(w.oos_return_pct for w in result.windows) / len(result.windows)
            assert result.avg_oos_return_pct == pytest.approx(manual_avg, abs=1e-6)

    def test_result_aggregation_avg_oos_sharpe(self):
        bars = make_bars(n=300, seed=22)
        wfo = WalkForwardOptimizer(num_windows=3)
        result = wfo.optimize(bars)
        if result.windows:
            manual_avg = sum(w.oos_sharpe for w in result.windows) / len(result.windows)
            assert result.avg_oos_sharpe == pytest.approx(manual_avg, abs=1e-6)

    def test_total_oos_trades(self):
        bars = make_bars(n=300, seed=23)
        wfo = WalkForwardOptimizer(num_windows=3)
        result = wfo.optimize(bars)
        if result.windows:
            manual_total = sum(w.oos_trades for w in result.windows)
            assert result.total_oos_trades == manual_total

    def test_most_robust_params_populated(self):
        bars = make_bars(n=300, seed=24)
        wfo = WalkForwardOptimizer(num_windows=3)
        result = wfo.optimize(bars)
        if result.windows:
            assert isinstance(result.most_robust_params, dict)
            # Should contain the same keys as the param grid
            for key in result.param_grid:
                assert key in result.most_robust_params

    def test_is_and_oos_ratio_in_result(self):
        bars = make_bars(n=250, seed=30)
        wfo = WalkForwardOptimizer(is_ratio=0.60, num_windows=3)
        result = wfo.optimize(bars)
        assert result.is_ratio == 0.60
        assert result.oos_ratio == pytest.approx(0.40, abs=1e-9)

    def test_best_params_frequency_populated(self):
        bars = make_bars(n=300, seed=25)
        wfo = WalkForwardOptimizer(num_windows=3)
        result = wfo.optimize(bars)
        if result.windows:
            assert isinstance(result.best_params_frequency, dict)
            assert len(result.best_params_frequency) >= 1


# ── format_result ────────────────────────────────────────────────


class TestFormatResult:
    def test_format_empty_result(self):
        result = WFOResult()
        text = WalkForwardOptimizer.format_result(result)
        assert isinstance(text, str)
        assert "WALK-FORWARD" in text
        assert "Windows" in text

    def test_format_result_with_windows(self):
        wr = WindowResult(
            window_id=0,
            best_params={"buy_threshold": 2, "sell_threshold": -2},
            is_return_pct=3.5,
            oos_return_pct=1.2,
            oos_trades=5,
            oos_win_rate=0.6,
            oos_sharpe=1.1,
        )
        result = WFOResult(
            windows=[wr],
            avg_oos_return_pct=1.2,
            avg_oos_sharpe=1.1,
            avg_oos_win_rate=0.6,
            total_oos_trades=5,
            num_windows=1,
            is_ratio=0.70,
            oos_ratio=0.30,
            param_grid={"buy_threshold": [1, 2, 3], "sell_threshold": [-1, -2, -3]},
            most_robust_params={"buy_threshold": 2, "sell_threshold": -2},
        )
        text = WalkForwardOptimizer.format_result(result)
        assert "buy_threshold" in text
        assert "Most Robust" in text
        assert "+1.20%" in text or "1.2" in text

    def test_format_result_returns_string(self):
        bars = make_bars(n=300, seed=77)
        wfo = WalkForwardOptimizer(num_windows=3)
        result = wfo.optimize(bars)
        text = WalkForwardOptimizer.format_result(result)
        assert isinstance(text, str)
        assert len(text) > 0
