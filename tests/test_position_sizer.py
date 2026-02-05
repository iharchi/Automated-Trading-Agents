"""Tests for utils/position_sizer.py"""

import pytest

from utils.position_sizer import (
    PositionSizer,
    SizingResult,
)


# ── Dataclass tests ───────────────────────────────────────────────

class TestSizingResult:
    def test_defaults(self):
        r = SizingResult(symbol="AAPL")
        assert r.signal == "HOLD"
        assert r.shares == 0
        assert r.error == ""

    def test_custom_values(self):
        r = SizingResult(symbol="AAPL", shares=100, price=150.0, equity=100_000)
        assert r.shares == 100
        assert r.price == 150.0


# ── Kelly Fraction tests ──────────────────────────────────────────

class TestKellyFraction:
    def test_positive_edge(self):
        # 55% win rate, 1.5 payoff → f* = (0.55 * 1.5 - 0.45) / 1.5 = 0.25
        f = PositionSizer.kelly_fraction(0.55, 1.5)
        assert f == pytest.approx(0.25, abs=0.001)

    def test_fifty_fifty_even_payoff(self):
        # 50% win, 1:1 payoff → f* = (0.5 * 1 - 0.5) / 1 = 0
        f = PositionSizer.kelly_fraction(0.50, 1.0)
        assert f == pytest.approx(0.0, abs=0.001)

    def test_negative_edge(self):
        # 40% win, 1:1 payoff → f* = (0.4 * 1 - 0.6) / 1 = -0.2
        f = PositionSizer.kelly_fraction(0.40, 1.0)
        assert f < 0

    def test_high_payoff_low_win_rate(self):
        # 30% win, 3:1 payoff → f* = (0.3 * 3 - 0.7) / 3 = 0.067
        f = PositionSizer.kelly_fraction(0.30, 3.0)
        assert f == pytest.approx(0.0667, abs=0.001)

    def test_zero_payoff(self):
        f = PositionSizer.kelly_fraction(0.55, 0.0)
        assert f == 0.0

    def test_negative_payoff(self):
        f = PositionSizer.kelly_fraction(0.55, -1.0)
        assert f == 0.0

    def test_win_rate_clamped(self):
        # Win rate > 1 should be clamped
        f = PositionSizer.kelly_fraction(1.5, 2.0)
        expected = PositionSizer.kelly_fraction(1.0, 2.0)
        assert f == expected

    def test_win_rate_zero(self):
        f = PositionSizer.kelly_fraction(0.0, 2.0)
        assert f < 0  # No wins → don't bet


# ── Volatility scaling tests ─────────────────────────────────────

class TestVolatilityScale:
    def test_high_vol_reduces_size(self):
        factor = PositionSizer.volatility_scale(0.30, vol_target=0.15)
        assert factor == pytest.approx(0.5)

    def test_low_vol_increases_size(self):
        factor = PositionSizer.volatility_scale(0.10, vol_target=0.15)
        assert factor == pytest.approx(1.5)

    def test_target_equals_actual(self):
        factor = PositionSizer.volatility_scale(0.15, vol_target=0.15)
        assert factor == pytest.approx(1.0)

    def test_zero_vol_returns_one(self):
        factor = PositionSizer.volatility_scale(0.0)
        assert factor == 1.0

    def test_capped_high(self):
        factor = PositionSizer.volatility_scale(0.01, vol_target=0.15)
        assert factor == 2.0  # Cap

    def test_capped_low(self):
        factor = PositionSizer.volatility_scale(1.0, vol_target=0.15)
        assert factor == 0.25  # Floor


# ── Core calculation tests ───────────────────────────────────────

class TestCalculation:
    def test_basic_buy(self):
        sizer = PositionSizer(kelly_factor=0.5)
        result = sizer.calculate(
            "AAPL",
            price=150.0,
            equity=100_000,
            signal="BUY",
            atr=2.5,
            win_rate=0.55,
            avg_win=3.0,
            avg_loss=2.0,
        )
        assert result.shares > 0
        assert result.signal == "BUY"
        assert result.stop_loss < result.price
        assert result.take_profit > result.price

    def test_basic_sell(self):
        sizer = PositionSizer(kelly_factor=0.5)
        result = sizer.calculate(
            "AAPL",
            price=150.0,
            equity=100_000,
            signal="SELL",
            atr=2.5,
            win_rate=0.55,
            avg_win=3.0,
            avg_loss=2.0,
        )
        assert result.shares > 0
        assert result.stop_loss > result.price
        assert result.take_profit < result.price

    def test_hold_signal_error(self):
        sizer = PositionSizer()
        result = sizer.calculate(
            "AAPL", price=150.0, equity=100_000, signal="HOLD",
        )
        assert result.shares == 0
        assert "Non-actionable" in result.error

    def test_zero_price_error(self):
        sizer = PositionSizer()
        result = sizer.calculate("AAPL", price=0, equity=100_000, signal="BUY")
        assert "Invalid price" in result.error

    def test_zero_equity_error(self):
        sizer = PositionSizer()
        result = sizer.calculate("AAPL", price=150.0, equity=0, signal="BUY")
        assert "Invalid price or equity" in result.error

    def test_negative_kelly_returns_zero(self):
        sizer = PositionSizer()
        result = sizer.calculate(
            "AAPL",
            price=150.0,
            equity=100_000,
            signal="BUY",
            atr=2.5,
            win_rate=0.30,
            avg_win=1.0,
            avg_loss=2.0,
        )
        assert result.shares == 0
        assert "Negative Kelly" in result.error

    def test_default_win_rate_used(self):
        sizer = PositionSizer(default_win_rate=0.60, default_payoff_ratio=2.0)
        result = sizer.calculate(
            "AAPL", price=150.0, equity=100_000, signal="BUY", atr=2.5,
        )
        assert result.win_rate == 0.60
        assert result.payoff_ratio == 2.0

    def test_fallback_risk_without_atr(self):
        sizer = PositionSizer()
        result = sizer.calculate(
            "AAPL",
            price=150.0,
            equity=100_000,
            signal="BUY",
            atr=0,
            win_rate=0.55,
            avg_win=3.0,
            avg_loss=2.0,
        )
        # Fallback: risk = 2% of price = $3.00
        assert result.risk_per_share == 3.0
        assert result.shares > 0


# ── Concentration limit tests ────────────────────────────────────

class TestConcentrationLimit:
    def test_concentration_caps_shares(self):
        sizer = PositionSizer(
            kelly_factor=1.0,  # Full Kelly for large position
            max_position_pct=0.05,
        )
        result = sizer.calculate(
            "AAPL",
            price=100.0,
            equity=100_000,
            signal="BUY",
            atr=2.0,
            win_rate=0.70,
            avg_win=5.0,
            avg_loss=2.0,
        )
        max_shares = int(100_000 * 0.05 / 100.0)
        assert result.shares <= max_shares
        assert result.concentration_limited is True

    def test_no_concentration_limit_if_small(self):
        sizer = PositionSizer(max_position_pct=0.50)
        result = sizer.calculate(
            "AAPL",
            price=150.0,
            equity=100_000,
            signal="BUY",
            atr=2.5,
            win_rate=0.55,
            avg_win=3.0,
            avg_loss=2.0,
        )
        assert result.concentration_limited is False


# ── Portfolio heat tests ─────────────────────────────────────────

class TestPortfolioHeat:
    def test_heat_limits_shares(self):
        sizer = PositionSizer(max_portfolio_heat=0.04, max_position_pct=0.50)
        result = sizer.calculate(
            "AAPL",
            price=150.0,
            equity=100_000,
            signal="BUY",
            atr=2.5,
            win_rate=0.55,
            avg_win=3.0,
            avg_loss=2.0,
            portfolio_heat=0.039,  # Only 0.1% remaining heat budget
        )
        # Max risk = 100_000 * 0.001 = $100
        # risk_per_share = 2.5 * 1.5 = $3.75
        # max_shares from heat = 100 / 3.75 = 26
        assert result.heat_limited is True
        assert result.shares <= 26

    def test_full_heat_returns_zero(self):
        sizer = PositionSizer(max_portfolio_heat=0.04)
        result = sizer.calculate(
            "AAPL",
            price=150.0,
            equity=100_000,
            signal="BUY",
            atr=2.5,
            win_rate=0.55,
            avg_win=3.0,
            avg_loss=2.0,
            portfolio_heat=0.06,  # Exceeds max
        )
        assert result.shares == 0
        assert result.heat_limited is True

    def test_zero_heat_no_limit(self):
        sizer = PositionSizer(max_portfolio_heat=0.06)
        result = sizer.calculate(
            "AAPL",
            price=150.0,
            equity=100_000,
            signal="BUY",
            atr=2.5,
            win_rate=0.55,
            avg_win=3.0,
            avg_loss=2.0,
            portfolio_heat=0.0,
        )
        assert result.portfolio_heat == 0.0


# ── Regime adjustments ───────────────────────────────────────────

class TestRegimeAdjustments:
    def test_trending_up_increases_size(self):
        sizer = PositionSizer()
        base = sizer.calculate(
            "AAPL", price=150.0, equity=100_000, signal="BUY",
            atr=2.5, win_rate=0.55, avg_win=3.0, avg_loss=2.0,
        )
        adjusted = sizer.calculate(
            "AAPL", price=150.0, equity=100_000, signal="BUY",
            atr=2.5, win_rate=0.55, avg_win=3.0, avg_loss=2.0,
            regime_adjustments={"position_size_factor": 1.2, "stop_multiplier": 1.0, "take_profit_multiplier": 1.0},
        )
        assert adjusted.regime_factor == 1.2
        assert adjusted.original_shares >= base.original_shares

    def test_volatile_reduces_size(self):
        sizer = PositionSizer()
        base = sizer.calculate(
            "AAPL", price=150.0, equity=100_000, signal="BUY",
            atr=2.5, win_rate=0.55, avg_win=3.0, avg_loss=2.0,
        )
        adjusted = sizer.calculate(
            "AAPL", price=150.0, equity=100_000, signal="BUY",
            atr=2.5, win_rate=0.55, avg_win=3.0, avg_loss=2.0,
            regime_adjustments={"position_size_factor": 0.5, "stop_multiplier": 1.5, "take_profit_multiplier": 1.0},
        )
        assert adjusted.regime_factor == 0.5
        assert adjusted.original_shares <= base.original_shares

    def test_regime_widens_stop(self):
        sizer = PositionSizer(atr_risk_mult=1.5)
        result = sizer.calculate(
            "AAPL", price=150.0, equity=100_000, signal="BUY",
            atr=2.5, win_rate=0.55, avg_win=3.0, avg_loss=2.0,
            regime_adjustments={"position_size_factor": 1.0, "stop_multiplier": 1.5, "take_profit_multiplier": 1.0},
        )
        # Normal stop: 150 - (2.5 * 1.5) = 146.25
        # With regime: 150 - (2.5 * 1.5 * 1.5) = 150 - 5.625 = 144.375
        expected_stop = round(150.0 - 2.5 * 1.5 * 1.5, 2)
        assert result.stop_loss == pytest.approx(expected_stop, abs=0.01)


# ── Volatility scaling integration ───────────────────────────────

class TestVolScalingIntegration:
    def test_high_vol_reduces_shares(self):
        sizer = PositionSizer()
        base = sizer.calculate(
            "AAPL", price=150.0, equity=100_000, signal="BUY",
            atr=2.5, win_rate=0.55, avg_win=3.0, avg_loss=2.0,
        )
        vol_result = sizer.calculate(
            "AAPL", price=150.0, equity=100_000, signal="BUY",
            atr=2.5, win_rate=0.55, avg_win=3.0, avg_loss=2.0,
            annual_vol=0.40,
        )
        assert vol_result.original_shares < base.original_shares
        assert vol_result.method == "kelly+vol"

    def test_low_vol_increases_shares(self):
        sizer = PositionSizer()
        base = sizer.calculate(
            "AAPL", price=150.0, equity=100_000, signal="BUY",
            atr=2.5, win_rate=0.55, avg_win=3.0, avg_loss=2.0,
        )
        vol_result = sizer.calculate(
            "AAPL", price=150.0, equity=100_000, signal="BUY",
            atr=2.5, win_rate=0.55, avg_win=3.0, avg_loss=2.0,
            annual_vol=0.08,
        )
        assert vol_result.original_shares >= base.original_shares


# ── Formatting tests ──────────────────────────────────────────────

class TestFormatting:
    def test_format_basic(self):
        sizer = PositionSizer()
        result = sizer.calculate(
            "AAPL", price=150.0, equity=100_000, signal="BUY",
            atr=2.5, win_rate=0.55, avg_win=3.0, avg_loss=2.0,
        )
        text = PositionSizer.format_result(result)
        assert "AAPL" in text
        assert "POSITION SIZER" in text
        assert "BUY" in text

    def test_format_error(self):
        result = SizingResult(symbol="AAPL", error="test error")
        text = PositionSizer.format_result(result)
        assert "test error" in text

    def test_format_with_limits(self):
        sizer = PositionSizer(kelly_factor=1.0, max_position_pct=0.01)
        result = sizer.calculate(
            "AAPL", price=150.0, equity=100_000, signal="BUY",
            atr=2.5, win_rate=0.70, avg_win=5.0, avg_loss=2.0,
        )
        text = PositionSizer.format_result(result)
        assert "concentration" in text

    def test_format_dict(self):
        sizer = PositionSizer()
        result = sizer.calculate(
            "AAPL", price=150.0, equity=100_000, signal="BUY",
            atr=2.5, win_rate=0.55, avg_win=3.0, avg_loss=2.0,
        )
        text = PositionSizer.format_result(result.__dict__)
        assert "AAPL" in text


# ── Edge cases ────────────────────────────────────────────────────

class TestEdgeCases:
    def test_very_high_win_rate(self):
        sizer = PositionSizer()
        result = sizer.calculate(
            "AAPL", price=150.0, equity=100_000, signal="BUY",
            atr=2.5, win_rate=0.95, avg_win=5.0, avg_loss=1.0,
        )
        assert result.shares > 0
        # Should be capped by concentration limit
        assert result.position_pct <= 0.10 + 0.001

    def test_penny_stock(self):
        sizer = PositionSizer()
        result = sizer.calculate(
            "PENNY", price=0.50, equity=100_000, signal="BUY",
            atr=0.05, win_rate=0.55, avg_win=0.10, avg_loss=0.05,
        )
        assert result.shares > 0
        assert result.stop_loss > 0

    def test_expensive_stock(self):
        sizer = PositionSizer()
        result = sizer.calculate(
            "BRK.A", price=500_000.0, equity=100_000, signal="BUY",
            atr=5000.0, win_rate=0.55, avg_win=10000.0, avg_loss=5000.0,
        )
        # Can't afford even 1 share at 10% concentration
        assert result.shares == 0
