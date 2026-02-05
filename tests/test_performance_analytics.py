"""Tests for utils/performance_analytics.py"""

import math

import pytest

from utils.performance_analytics import (
    PerformanceAnalytics,
    PerformanceReport,
    TradeRecord,
)


# ── Helpers ─────────────────────────────────────────────────────

def _make_trade(symbol="AAPL", side="buy", qty=10, price=100.0,
                pnl=0.0, pnl_pct=0.0, timestamp="", is_win=False):
    return TradeRecord(
        symbol=symbol, side=side, qty=qty, price=price,
        pnl=pnl, pnl_pct=pnl_pct, timestamp=timestamp, is_win=is_win,
    )


def _winning_trades(n, pnl=100.0, pnl_pct=1.0, symbol="AAPL"):
    return [_make_trade(pnl=pnl, pnl_pct=pnl_pct, symbol=symbol) for _ in range(n)]


def _losing_trades(n, pnl=-50.0, pnl_pct=-0.5, symbol="AAPL"):
    return [_make_trade(pnl=pnl, pnl_pct=pnl_pct, symbol=symbol) for _ in range(n)]


# ── TradeRecord dataclass ──────────────────────────────────────

class TestTradeRecord:
    def test_defaults(self):
        t = TradeRecord(symbol="AAPL", side="buy", qty=10, price=150.0)
        assert t.symbol == "AAPL"
        assert t.side == "buy"
        assert t.qty == 10
        assert t.price == 150.0
        assert t.pnl == 0.0
        assert t.pnl_pct == 0.0
        assert t.timestamp == ""
        assert t.is_win is False

    def test_custom_values(self):
        t = TradeRecord(
            symbol="MSFT", side="sell", qty=5, price=300.0,
            pnl=250.0, pnl_pct=2.5, timestamp="2025-01-01T10:00:00",
            is_win=True,
        )
        assert t.pnl == 250.0
        assert t.pnl_pct == 2.5
        assert t.timestamp == "2025-01-01T10:00:00"
        assert t.is_win is True


# ── PerformanceReport dataclass ────────────────────────────────

class TestPerformanceReport:
    def test_defaults(self):
        r = PerformanceReport()
        assert r.total_return_pct == 0.0
        assert r.total_trades == 0
        assert r.win_rate == 0.0
        assert r.per_symbol == {}

    def test_risk_reward_ratio_normal(self):
        r = PerformanceReport(avg_win=200.0, avg_loss=-100.0)
        assert r.risk_reward_ratio == pytest.approx(2.0)

    def test_risk_reward_ratio_zero_loss(self):
        r = PerformanceReport(avg_win=200.0, avg_loss=0.0)
        assert r.risk_reward_ratio == 0.0


# ── PerformanceAnalytics.__init__ ──────────────────────────────

class TestInit:
    def test_default_params(self):
        pa = PerformanceAnalytics()
        assert pa.risk_free_rate == 0.05
        assert pa.trading_days_year == 252

    def test_custom_params(self):
        pa = PerformanceAnalytics(risk_free_rate=0.03, trading_days_year=365)
        assert pa.risk_free_rate == 0.03
        assert pa.trading_days_year == 365


# ── analyze_trades ─────────────────────────────────────────────

class TestAnalyzeTrades:
    def test_empty_trades(self):
        pa = PerformanceAnalytics()
        report = pa.analyze_trades([])
        assert report.total_trades == 0
        assert report.win_rate == 0.0
        assert report.profit_factor == 0.0
        assert report.expectancy == 0.0

    def test_all_winning_trades(self):
        pa = PerformanceAnalytics()
        trades = _winning_trades(5, pnl=200.0, pnl_pct=2.0)
        report = pa.analyze_trades(trades)

        assert report.total_trades == 5
        assert report.winning_trades == 5
        assert report.losing_trades == 0
        assert report.win_rate == pytest.approx(1.0)
        assert report.profit_factor == float("inf")
        assert report.avg_win == pytest.approx(200.0)
        assert report.avg_loss == 0.0
        assert report.avg_win_pct == pytest.approx(2.0)
        assert report.avg_loss_pct == 0.0

    def test_all_losing_trades(self):
        pa = PerformanceAnalytics()
        trades = _losing_trades(4, pnl=-100.0, pnl_pct=-1.5)
        report = pa.analyze_trades(trades)

        assert report.total_trades == 4
        assert report.winning_trades == 0
        assert report.losing_trades == 4
        assert report.win_rate == pytest.approx(0.0)
        assert report.profit_factor == 0.0
        assert report.avg_win == 0.0
        assert report.avg_loss == pytest.approx(-100.0)
        assert report.avg_loss_pct == pytest.approx(-1.5)

    def test_mixed_trades_win_rate(self):
        pa = PerformanceAnalytics()
        trades = _winning_trades(3, pnl=100.0) + _losing_trades(2, pnl=-50.0)
        report = pa.analyze_trades(trades)

        assert report.total_trades == 5
        assert report.winning_trades == 3
        assert report.losing_trades == 2
        assert report.win_rate == pytest.approx(0.6)

    def test_mixed_trades_profit_factor(self):
        pa = PerformanceAnalytics()
        trades = _winning_trades(2, pnl=150.0) + _losing_trades(3, pnl=-50.0)
        report = pa.analyze_trades(trades)

        # gross_wins = 300, gross_losses = 150
        assert report.profit_factor == pytest.approx(2.0)

    def test_mixed_trades_expectancy(self):
        pa = PerformanceAnalytics()
        trades = _winning_trades(3, pnl=100.0) + _losing_trades(2, pnl=-60.0)
        report = pa.analyze_trades(trades)

        # win_rate=0.6, avg_win=100, avg_loss=-60
        # expectancy = 0.6*100 + 0.4*(-60) = 60 - 24 = 36
        assert report.expectancy == pytest.approx(36.0)

    def test_largest_win_and_loss(self):
        pa = PerformanceAnalytics()
        trades = [
            _make_trade(pnl=50.0),
            _make_trade(pnl=300.0),
            _make_trade(pnl=-20.0),
            _make_trade(pnl=-200.0),
            _make_trade(pnl=10.0),
        ]
        report = pa.analyze_trades(trades)

        assert report.largest_win == pytest.approx(300.0)
        assert report.largest_loss == pytest.approx(-200.0)

    def test_avg_win_avg_loss(self):
        pa = PerformanceAnalytics()
        trades = [
            _make_trade(pnl=100.0, pnl_pct=1.0),
            _make_trade(pnl=200.0, pnl_pct=2.0),
            _make_trade(pnl=-50.0, pnl_pct=-0.5),
            _make_trade(pnl=-150.0, pnl_pct=-1.5),
        ]
        report = pa.analyze_trades(trades)

        assert report.avg_win == pytest.approx(150.0)
        assert report.avg_loss == pytest.approx(-100.0)
        assert report.avg_win_pct == pytest.approx(1.5)
        assert report.avg_loss_pct == pytest.approx(-1.0)

    def test_zero_pnl_counted_as_loss(self):
        """A trade with pnl=0 is counted as a loss (pnl <= 0)."""
        pa = PerformanceAnalytics()
        trades = [_make_trade(pnl=0.0)]
        report = pa.analyze_trades(trades)

        assert report.winning_trades == 0
        assert report.losing_trades == 1


# ── analyze_equity_curve ───────────────────────────────────────

class TestAnalyzeEquityCurve:
    def test_empty_equity(self):
        pa = PerformanceAnalytics()
        report = pa.analyze_equity_curve([])
        assert report.final_equity == 100000.0  # default initial_capital
        assert report.trading_days == 0

    def test_single_value_equity(self):
        pa = PerformanceAnalytics()
        report = pa.analyze_equity_curve([105000.0])
        assert report.final_equity == 105000.0
        assert report.trading_days == 0

    def test_flat_equity(self):
        pa = PerformanceAnalytics()
        equity = [100000.0] * 10
        report = pa.analyze_equity_curve(equity, initial_capital=100000.0)

        assert report.total_return_pct == pytest.approx(0.0)
        assert report.sharpe_ratio == pytest.approx(0.0)
        assert report.max_drawdown_pct == pytest.approx(0.0)
        assert report.current_drawdown_pct == pytest.approx(0.0)

    def test_growing_equity_positive_return(self):
        pa = PerformanceAnalytics()
        # Linearly growing: 100k -> 110k over 10 days
        equity = [100000.0 + i * 1000.0 for i in range(11)]
        report = pa.analyze_equity_curve(equity, initial_capital=100000.0)

        assert report.total_return_pct == pytest.approx(10.0)
        assert report.final_equity == pytest.approx(110000.0)
        assert report.peak_equity == pytest.approx(110000.0)
        assert report.trading_days == 11

    def test_growing_equity_positive_sharpe(self):
        pa = PerformanceAnalytics()
        # Steadily growing equity
        equity = [100000.0 + i * 500.0 for i in range(100)]
        report = pa.analyze_equity_curve(equity, initial_capital=100000.0)

        assert report.sharpe_ratio > 0

    def test_declining_equity_negative_return(self):
        pa = PerformanceAnalytics()
        equity = [100000.0 - i * 1000.0 for i in range(11)]
        report = pa.analyze_equity_curve(equity, initial_capital=100000.0)

        assert report.total_return_pct == pytest.approx(-10.0)

    def test_drawdown_periods(self):
        pa = PerformanceAnalytics()
        # Peak at 120k, drops to 96k (20% dd), recovers partially to 108k
        equity = [100000.0, 110000.0, 120000.0, 108000.0, 96000.0, 108000.0]
        report = pa.analyze_equity_curve(equity, initial_capital=100000.0)

        # Max dd: (120000 - 96000) / 120000 = 20%
        assert report.max_drawdown_pct == pytest.approx(20.0)
        # Current dd: (120000 - 108000) / 120000 = 10%
        assert report.current_drawdown_pct == pytest.approx(10.0)

    def test_annualised_return_calculation(self):
        pa = PerformanceAnalytics(trading_days_year=252)
        # 252 trading days -> exactly 1 year, 10% total return
        equity = [100000.0] + [110000.0] * 251
        report = pa.analyze_equity_curve(equity, initial_capital=100000.0)

        # With 252 days and 10% return over 1 year, annualised ~ 10%
        assert report.annualised_return_pct == pytest.approx(10.0, abs=0.5)

    def test_annualised_return_partial_year(self):
        pa = PerformanceAnalytics(trading_days_year=252)
        # 126 days ~ half year, 10% total return
        equity = [100000.0] + [110000.0] * 125
        report = pa.analyze_equity_curve(equity, initial_capital=100000.0)

        # Annualised should be more than 10% (compounding over half year)
        assert report.annualised_return_pct > 10.0

    def test_total_loss_annualised_caps_at_minus_100(self):
        pa = PerformanceAnalytics()
        # Total loss: equity goes to zero
        equity = [100000.0, 50000.0, 0.0001]
        report = pa.analyze_equity_curve(equity, initial_capital=100000.0)

        # total_ret ~ -1, so annualised should be -100
        assert report.annualised_return_pct == pytest.approx(-100.0, abs=0.1)


# ── analyze_full ───────────────────────────────────────────────

class TestAnalyzeFull:
    def test_trades_only(self):
        pa = PerformanceAnalytics()
        trades = _winning_trades(3, pnl=100.0) + _losing_trades(2, pnl=-50.0)
        report = pa.analyze_full(trades)

        assert report.total_trades == 5
        assert report.winning_trades == 3
        # No equity curve provided, so equity metrics stay at defaults
        assert report.trading_days == 0

    def test_trades_and_equity_merged(self):
        pa = PerformanceAnalytics()
        trades = _winning_trades(3, pnl=100.0) + _losing_trades(2, pnl=-50.0)
        equity = [100000.0, 105000.0, 110000.0, 108000.0, 112000.0]
        report = pa.analyze_full(trades, equity_values=equity, initial_capital=100000.0)

        # Trade metrics
        assert report.total_trades == 5
        assert report.winning_trades == 3

        # Equity metrics merged in
        assert report.total_return_pct == pytest.approx(12.0)
        assert report.final_equity == pytest.approx(112000.0)
        assert report.peak_equity == pytest.approx(112000.0)
        assert report.trading_days == 5

    def test_equity_too_short_not_merged(self):
        pa = PerformanceAnalytics()
        trades = _winning_trades(2)
        report = pa.analyze_full(trades, equity_values=[100000.0])

        # Only 1 data point -> equity analysis skipped
        assert report.total_return_pct == 0.0
        assert report.trading_days == 0

    def test_equity_none(self):
        pa = PerformanceAnalytics()
        trades = _winning_trades(2)
        report = pa.analyze_full(trades, equity_values=None)

        assert report.total_trades == 2
        assert report.trading_days == 0


# ── _sharpe helper ─────────────────────────────────────────────

class TestSharpeHelper:
    def test_fewer_than_two_returns(self):
        pa = PerformanceAnalytics()
        assert pa._sharpe([]) == 0.0
        assert pa._sharpe([0.01]) == 0.0

    def test_zero_std_returns(self):
        pa = PerformanceAnalytics()
        # All returns identical -> std = 0 -> sharpe = 0
        assert pa._sharpe([0.01, 0.01, 0.01, 0.01]) == 0.0

    def test_positive_returns(self):
        pa = PerformanceAnalytics(risk_free_rate=0.0)
        # Varied positive returns should produce positive sharpe
        returns = [0.01, 0.02, 0.015, 0.005, 0.01]
        assert pa._sharpe(returns) > 0

    def test_negative_returns(self):
        pa = PerformanceAnalytics(risk_free_rate=0.0)
        returns = [-0.01, -0.02, -0.015, -0.005, -0.01]
        assert pa._sharpe(returns) < 0


# ── _sortino helper ────────────────────────────────────────────

class TestSortinoHelper:
    def test_fewer_than_two_returns(self):
        pa = PerformanceAnalytics()
        assert pa._sortino([]) == 0.0
        assert pa._sortino([0.01]) == 0.0

    def test_no_downside_returns_positive_mean(self):
        pa = PerformanceAnalytics(risk_free_rate=0.0)
        # All returns positive and above rf -> inf
        returns = [0.05, 0.06, 0.07, 0.08]
        assert pa._sortino(returns) == float("inf")

    def test_no_downside_returns_negative_mean(self):
        pa = PerformanceAnalytics(risk_free_rate=1.0)
        # Risk free rate so high that daily_rf dwarfs returns,
        # but none are below daily_rf? Actually with rf=1.0:
        # daily_rf = 1.0/252 ~ 0.00397
        # returns [0.004, 0.005] are above daily_rf so no downside
        returns = [0.004, 0.005, 0.006]
        result = pa._sortino(returns)
        assert result == float("inf")

    def test_with_downside_returns(self):
        pa = PerformanceAnalytics(risk_free_rate=0.0)
        returns = [0.01, -0.02, 0.015, -0.01, 0.005]
        result = pa._sortino(returns)
        # Should be finite and calculable
        assert math.isfinite(result)


# ── _drawdowns helper ──────────────────────────────────────────

class TestDrawdownsHelper:
    def test_empty_equity(self):
        max_dd, cur_dd = PerformanceAnalytics._drawdowns([])
        assert max_dd == 0.0
        assert cur_dd == 0.0

    def test_flat_equity(self):
        max_dd, cur_dd = PerformanceAnalytics._drawdowns([100.0, 100.0, 100.0])
        assert max_dd == pytest.approx(0.0)
        assert cur_dd == pytest.approx(0.0)

    def test_peaked_then_declined(self):
        # Peak 200, drops to 160 (20% dd), stays there
        equity = [100.0, 150.0, 200.0, 180.0, 160.0]
        max_dd, cur_dd = PerformanceAnalytics._drawdowns(equity)
        assert max_dd == pytest.approx(20.0)
        assert cur_dd == pytest.approx(20.0)

    def test_recovered_from_drawdown(self):
        # Peak 200, drops to 160, recovers to 210
        equity = [100.0, 200.0, 160.0, 210.0]
        max_dd, cur_dd = PerformanceAnalytics._drawdowns(equity)
        # Max dd: (200-160)/200 = 20%
        assert max_dd == pytest.approx(20.0)
        # New peak is 210, final is 210 -> current dd = 0
        assert cur_dd == pytest.approx(0.0)

    def test_monotonically_increasing(self):
        equity = [100.0, 110.0, 120.0, 130.0]
        max_dd, cur_dd = PerformanceAnalytics._drawdowns(equity)
        assert max_dd == pytest.approx(0.0)
        assert cur_dd == pytest.approx(0.0)


# ── _compute_streaks ───────────────────────────────────────────

class TestComputeStreaks:
    def test_consecutive_wins(self):
        pa = PerformanceAnalytics()
        trades = _winning_trades(5)
        report = PerformanceReport()
        pa._compute_streaks(trades, report)

        assert report.max_consecutive_wins == 5
        assert report.max_consecutive_losses == 0
        assert report.current_streak == 5
        assert report.current_streak_type == "win"

    def test_consecutive_losses(self):
        pa = PerformanceAnalytics()
        trades = _losing_trades(4)
        report = PerformanceReport()
        pa._compute_streaks(trades, report)

        assert report.max_consecutive_wins == 0
        assert report.max_consecutive_losses == 4
        assert report.current_streak == -4
        assert report.current_streak_type == "loss"

    def test_alternating_wins_losses(self):
        pa = PerformanceAnalytics()
        trades = [
            _make_trade(pnl=100.0),
            _make_trade(pnl=-50.0),
            _make_trade(pnl=100.0),
            _make_trade(pnl=-50.0),
        ]
        report = PerformanceReport()
        pa._compute_streaks(trades, report)

        assert report.max_consecutive_wins == 1
        assert report.max_consecutive_losses == 1

    def test_current_streak_type_after_mixed(self):
        pa = PerformanceAnalytics()
        # 3 wins then 2 losses
        trades = _winning_trades(3) + _losing_trades(2)
        report = PerformanceReport()
        pa._compute_streaks(trades, report)

        assert report.max_consecutive_wins == 3
        assert report.max_consecutive_losses == 2
        assert report.current_streak == -2
        assert report.current_streak_type == "loss"

    def test_empty_trades_streak(self):
        pa = PerformanceAnalytics()
        report = PerformanceReport()
        pa._compute_streaks([], report)

        assert report.max_consecutive_wins == 0
        assert report.max_consecutive_losses == 0
        assert report.current_streak == 0
        assert report.current_streak_type == ""

    def test_longest_streak_in_middle(self):
        pa = PerformanceAnalytics()
        # 1 loss, 4 wins, 1 loss -> max wins = 4
        trades = (
            _losing_trades(1) +
            _winning_trades(4) +
            _losing_trades(1)
        )
        report = PerformanceReport()
        pa._compute_streaks(trades, report)

        assert report.max_consecutive_wins == 4
        assert report.max_consecutive_losses == 1


# ── _compute_per_symbol ────────────────────────────────────────

class TestComputePerSymbol:
    def test_multiple_symbols(self):
        pa = PerformanceAnalytics()
        trades = [
            _make_trade(symbol="AAPL", pnl=100.0),
            _make_trade(symbol="AAPL", pnl=-50.0),
            _make_trade(symbol="MSFT", pnl=200.0),
            _make_trade(symbol="MSFT", pnl=150.0),
            _make_trade(symbol="GOOG", pnl=-100.0),
        ]
        report = PerformanceReport()
        pa._compute_per_symbol(trades, report)

        assert "AAPL" in report.per_symbol
        assert "MSFT" in report.per_symbol
        assert "GOOG" in report.per_symbol

        aapl = report.per_symbol["AAPL"]
        assert aapl["trades"] == 2
        assert aapl["wins"] == 1
        assert aapl["total_pnl"] == pytest.approx(50.0)
        assert aapl["win_rate"] == pytest.approx(0.5)

        msft = report.per_symbol["MSFT"]
        assert msft["trades"] == 2
        assert msft["wins"] == 2
        assert msft["total_pnl"] == pytest.approx(350.0)
        assert msft["profit_factor"] == float("inf")  # no losses

        goog = report.per_symbol["GOOG"]
        assert goog["trades"] == 1
        assert goog["wins"] == 0
        assert goog["total_pnl"] == pytest.approx(-100.0)

    def test_single_symbol(self):
        pa = PerformanceAnalytics()
        trades = _winning_trades(3, pnl=100.0, symbol="TSLA")
        report = PerformanceReport()
        pa._compute_per_symbol(trades, report)

        assert len(report.per_symbol) == 1
        assert "TSLA" in report.per_symbol
        tsla = report.per_symbol["TSLA"]
        assert tsla["trades"] == 3
        assert tsla["gross_wins"] == pytest.approx(300.0)
        assert tsla["gross_losses"] == pytest.approx(0.0)

    def test_per_symbol_profit_factor(self):
        pa = PerformanceAnalytics()
        trades = [
            _make_trade(symbol="SPY", pnl=300.0),
            _make_trade(symbol="SPY", pnl=-100.0),
        ]
        report = PerformanceReport()
        pa._compute_per_symbol(trades, report)

        spy = report.per_symbol["SPY"]
        assert spy["profit_factor"] == pytest.approx(3.0)


# ── trades_from_orders ─────────────────────────────────────────

class TestTradesFromOrders:
    def test_converts_filled_orders(self):
        orders = [
            {"symbol": "AAPL", "side": "buy", "qty": 10, "status": "filled",
             "timestamp": "2025-01-01"},
            {"symbol": "MSFT", "side": "sell", "qty": 5, "status": "dry_run",
             "timestamp": "2025-01-02"},
        ]
        trades = PerformanceAnalytics.trades_from_orders(orders)

        assert len(trades) == 2
        assert trades[0].symbol == "AAPL"
        assert trades[0].side == "buy"
        assert trades[0].qty == 10
        assert trades[0].timestamp == "2025-01-01"
        assert trades[1].symbol == "MSFT"
        assert trades[1].qty == 5

    def test_skips_non_filled_orders(self):
        orders = [
            {"symbol": "AAPL", "side": "buy", "qty": 10, "status": "pending"},
            {"symbol": "MSFT", "side": "sell", "qty": 5, "status": "cancelled"},
            {"symbol": "GOOG", "side": "buy", "qty": 3, "status": "rejected"},
        ]
        trades = PerformanceAnalytics.trades_from_orders(orders)
        assert len(trades) == 0

    def test_skips_zero_qty_orders(self):
        orders = [
            {"symbol": "AAPL", "side": "buy", "qty": 0, "status": "filled"},
        ]
        trades = PerformanceAnalytics.trades_from_orders(orders)
        assert len(trades) == 0

    def test_empty_orders(self):
        trades = PerformanceAnalytics.trades_from_orders([])
        assert trades == []

    def test_missing_fields_use_defaults(self):
        orders = [
            {"status": "filled", "qty": 1},
        ]
        trades = PerformanceAnalytics.trades_from_orders(orders)
        assert len(trades) == 1
        assert trades[0].symbol == ""
        assert trades[0].side == ""
        assert trades[0].price == 0.0
        assert trades[0].timestamp == ""


# ── format_report ──────────────────────────────────────────────

class TestFormatReport:
    def test_contains_header(self):
        report = PerformanceReport()
        text = PerformanceAnalytics.format_report(report)
        assert "PERFORMANCE ANALYTICS" in text

    def test_contains_return_section(self):
        report = PerformanceReport(total_return_pct=12.5, annualised_return_pct=25.0)
        text = PerformanceAnalytics.format_report(report)
        assert "Total Return" in text
        assert "+12.50%" in text
        assert "Annualised Return" in text
        assert "+25.00%" in text

    def test_contains_risk_adjusted_section(self):
        report = PerformanceReport(sharpe_ratio=1.5, sortino_ratio=2.1)
        text = PerformanceAnalytics.format_report(report)
        assert "Sharpe Ratio" in text
        assert "1.500" in text
        assert "Sortino Ratio" in text
        assert "2.100" in text

    def test_contains_trade_quality_section(self):
        report = PerformanceReport(
            total_trades=100, win_rate=0.65,
            profit_factor=2.0, expectancy=50.0,
        )
        text = PerformanceAnalytics.format_report(report)
        assert "Total Trades" in text
        assert "100" in text
        assert "Win Rate" in text
        assert "65.0%" in text
        assert "Profit Factor" in text
        assert "Expectancy" in text

    def test_streak_section_shown_when_nonzero(self):
        report = PerformanceReport(
            max_consecutive_wins=5,
            max_consecutive_losses=3,
            current_streak=2,
            current_streak_type="win",
        )
        text = PerformanceAnalytics.format_report(report)
        assert "Max Consec Wins" in text
        assert "5" in text
        assert "Current Streak" in text
        assert "2 win(s)" in text

    def test_streak_section_hidden_when_zero(self):
        report = PerformanceReport(current_streak=0, current_streak_type="")
        text = PerformanceAnalytics.format_report(report)
        assert "Current Streak" not in text

    def test_per_symbol_section_shown(self):
        report = PerformanceReport(
            per_symbol={
                "AAPL": {
                    "trades": 10, "wins": 6, "total_pnl": 500.0,
                    "win_rate": 0.6, "profit_factor": 2.5,
                    "gross_wins": 800.0, "gross_losses": 300.0,
                },
            }
        )
        text = PerformanceAnalytics.format_report(report)
        assert "Per-Symbol Breakdown" in text
        assert "AAPL" in text

    def test_per_symbol_section_hidden_when_empty(self):
        report = PerformanceReport(per_symbol={})
        text = PerformanceAnalytics.format_report(report)
        assert "Per-Symbol Breakdown" not in text

    def test_format_report_returns_string(self):
        report = PerformanceReport()
        result = PerformanceAnalytics.format_report(report)
        assert isinstance(result, str)


# ── _std helper ────────────────────────────────────────────────

class TestStdHelper:
    def test_fewer_than_two_values(self):
        assert PerformanceAnalytics._std([]) == 0.0
        assert PerformanceAnalytics._std([42.0]) == 0.0

    def test_identical_values(self):
        assert PerformanceAnalytics._std([5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_known_std(self):
        # Sample std (n-1) of [2, 4, 4, 4, 5, 5, 7, 9]:
        # mean=5, sum_sq_diff=32, variance=32/7~4.571, std~2.138
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = PerformanceAnalytics._std(values)
        assert result == pytest.approx(2.1381, abs=0.001)
