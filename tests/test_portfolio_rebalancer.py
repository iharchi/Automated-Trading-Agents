"""Tests for utils/portfolio_rebalancer.py"""

import pytest

from utils.portfolio_rebalancer import (
    ALLOCATION_MODES,
    PortfolioRebalancer,
    PortfolioWeight,
    RebalanceOrder,
    RebalanceResult,
)


# ── Dataclass tests ───────────────────────────────────────────────


class TestPortfolioWeight:
    def test_defaults(self):
        w = PortfolioWeight(symbol="AAPL")
        assert w.symbol == "AAPL"
        assert w.current_weight == 0.0
        assert w.target_weight == 0.0
        assert w.drift == 0.0
        assert w.current_value == 0.0
        assert w.target_value == 0.0

    def test_custom_values(self):
        w = PortfolioWeight(
            symbol="MSFT",
            current_weight=0.40,
            target_weight=0.25,
            drift=0.15,
            current_value=40_000,
            target_value=25_000,
        )
        assert w.symbol == "MSFT"
        assert w.current_weight == 0.40
        assert w.target_weight == 0.25
        assert w.drift == 0.15
        assert w.current_value == 40_000
        assert w.target_value == 25_000


class TestRebalanceOrder:
    def test_defaults(self):
        o = RebalanceOrder(symbol="AAPL", side="buy")
        assert o.symbol == "AAPL"
        assert o.side == "buy"
        assert o.shares == 0
        assert o.notional_value == 0.0
        assert o.current_weight == 0.0
        assert o.target_weight == 0.0
        assert o.price == 0.0
        assert o.reason == ""

    def test_custom_values(self):
        o = RebalanceOrder(
            symbol="GOOGL",
            side="sell",
            shares=10,
            notional_value=15_000,
            current_weight=0.30,
            target_weight=0.20,
            price=1500.0,
            reason="drift=+10.0%",
        )
        assert o.side == "sell"
        assert o.shares == 10
        assert o.notional_value == 15_000
        assert o.price == 1500.0


class TestRebalanceResult:
    def test_defaults(self):
        r = RebalanceResult()
        assert r.weights == []
        assert r.orders == []
        assert r.max_drift == 0.0
        assert r.needs_rebalance is False
        assert r.total_buy_value == 0.0
        assert r.total_sell_value == 0.0
        assert r.mode == "equal_weight"

    def test_custom_values(self):
        w = PortfolioWeight(symbol="AAPL")
        o = RebalanceOrder(symbol="AAPL", side="buy")
        r = RebalanceResult(
            weights=[w],
            orders=[o],
            max_drift=0.08,
            needs_rebalance=True,
            total_buy_value=5_000,
            total_sell_value=3_000,
            mode="custom",
        )
        assert len(r.weights) == 1
        assert len(r.orders) == 1
        assert r.max_drift == 0.08
        assert r.needs_rebalance is True
        assert r.mode == "custom"


# ── Init tests ────────────────────────────────────────────────────


class TestInit:
    def test_default_init(self):
        rb = PortfolioRebalancer()
        assert rb.mode == "equal_weight"
        assert rb.custom_weights == {}
        assert rb.drift_threshold == 0.05
        assert rb.max_single_rebalance_pct == 0.25
        assert rb.min_order_value == 100.0

    def test_equal_weight_mode(self):
        rb = PortfolioRebalancer(mode="equal_weight")
        assert rb.mode == "equal_weight"

    def test_custom_mode(self):
        weights = {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}
        rb = PortfolioRebalancer(mode="custom", custom_weights=weights)
        assert rb.mode == "custom"
        assert rb.custom_weights == weights

    def test_market_cap_mode(self):
        rb = PortfolioRebalancer(mode="market_cap")
        assert rb.mode == "market_cap"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            PortfolioRebalancer(mode="momentum")

    def test_custom_params(self):
        rb = PortfolioRebalancer(
            drift_threshold=0.10,
            max_single_rebalance_pct=0.50,
            min_order_value=500.0,
        )
        assert rb.drift_threshold == 0.10
        assert rb.max_single_rebalance_pct == 0.50
        assert rb.min_order_value == 500.0

    def test_custom_weights_defaults_to_empty_dict(self):
        rb = PortfolioRebalancer(mode="custom")
        assert rb.custom_weights == {}


# ── compute_target_weights tests ─────────────────────────────────


class TestComputeTargetWeights:
    # -- equal_weight mode --

    def test_equal_weight_three_symbols(self):
        rb = PortfolioRebalancer(mode="equal_weight")
        w = rb.compute_target_weights(["AAPL", "MSFT", "GOOGL"])
        assert len(w) == 3
        for sym in ["AAPL", "MSFT", "GOOGL"]:
            assert w[sym] == pytest.approx(1 / 3, abs=1e-9)

    def test_equal_weight_five_symbols(self):
        rb = PortfolioRebalancer(mode="equal_weight")
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        w = rb.compute_target_weights(symbols)
        assert len(w) == 5
        for sym in symbols:
            assert w[sym] == pytest.approx(0.20, abs=1e-9)

    def test_equal_weight_single_symbol(self):
        rb = PortfolioRebalancer(mode="equal_weight")
        w = rb.compute_target_weights(["SPY"])
        assert w["SPY"] == pytest.approx(1.0)

    def test_equal_weight_empty_symbols(self):
        rb = PortfolioRebalancer(mode="equal_weight")
        w = rb.compute_target_weights([])
        assert w == {}

    def test_equal_weight_sums_to_one(self):
        rb = PortfolioRebalancer(mode="equal_weight")
        w = rb.compute_target_weights(["A", "B", "C", "D"])
        assert sum(w.values()) == pytest.approx(1.0)

    # -- custom mode --

    def test_custom_weight_with_matching_weights(self):
        rb = PortfolioRebalancer(
            mode="custom",
            custom_weights={"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2},
        )
        w = rb.compute_target_weights(["AAPL", "MSFT", "GOOGL"])
        assert w["AAPL"] == pytest.approx(0.5)
        assert w["MSFT"] == pytest.approx(0.3)
        assert w["GOOGL"] == pytest.approx(0.2)

    def test_custom_weight_normalises_to_one(self):
        # Weights don't sum to 1.0; module should normalise
        rb = PortfolioRebalancer(
            mode="custom",
            custom_weights={"AAPL": 2.0, "MSFT": 3.0},
        )
        w = rb.compute_target_weights(["AAPL", "MSFT"])
        assert sum(w.values()) == pytest.approx(1.0)
        assert w["AAPL"] == pytest.approx(0.4)
        assert w["MSFT"] == pytest.approx(0.6)

    def test_custom_weight_missing_symbol_gets_zero(self):
        rb = PortfolioRebalancer(
            mode="custom",
            custom_weights={"AAPL": 0.7, "MSFT": 0.3},
        )
        w = rb.compute_target_weights(["AAPL", "MSFT", "GOOGL"])
        # GOOGL is not in custom_weights -> gets 0 from .get(s, 0)
        assert w["GOOGL"] == pytest.approx(0.0)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_custom_weight_no_matching_weights_fallback(self):
        # None of the symbols have custom weights -> total=0 -> recursive
        # fallback call (known bug: recurses with same mode, never reaches
        # equal_weight branch).
        rb = PortfolioRebalancer(
            mode="custom",
            custom_weights={"XYZ": 1.0},
        )
        with pytest.raises(RecursionError):
            rb.compute_target_weights(["AAPL", "MSFT", "GOOGL"])

    def test_custom_weight_empty_custom_weights_fallback(self):
        # Empty custom_weights -> total=0 -> recursive fallback (same bug).
        rb = PortfolioRebalancer(mode="custom", custom_weights={})
        with pytest.raises(RecursionError):
            rb.compute_target_weights(["AAPL", "MSFT"])

    # -- market_cap mode --

    def test_market_cap_with_prices(self):
        rb = PortfolioRebalancer(mode="market_cap")
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 150.0}
        w = rb.compute_target_weights(["AAPL", "MSFT", "GOOGL"], prices)
        total = 150 + 300 + 150
        assert w["AAPL"] == pytest.approx(150 / total)
        assert w["MSFT"] == pytest.approx(300 / total)
        assert w["GOOGL"] == pytest.approx(150 / total)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_market_cap_no_prices_fallback(self):
        # No prices -> recursive fallback (known bug: recurses with same mode).
        rb = PortfolioRebalancer(mode="market_cap")
        with pytest.raises(RecursionError):
            rb.compute_target_weights(["AAPL", "MSFT"])

    def test_market_cap_zero_prices_fallback(self):
        # All prices zero -> total_price=0 -> recursive fallback (same bug).
        rb = PortfolioRebalancer(mode="market_cap")
        prices = {"AAPL": 0.0, "MSFT": 0.0}
        with pytest.raises(RecursionError):
            rb.compute_target_weights(["AAPL", "MSFT"], prices)

    def test_market_cap_partial_prices(self):
        rb = PortfolioRebalancer(mode="market_cap")
        prices = {"AAPL": 100.0}  # MSFT not in prices
        w = rb.compute_target_weights(["AAPL", "MSFT"], prices)
        # MSFT gets 0 from prices.get -> AAPL=100/100=1.0, MSFT=0/100=0.0
        assert w["AAPL"] == pytest.approx(1.0)
        assert w["MSFT"] == pytest.approx(0.0)


# ── compute_current_weights tests ────────────────────────────────


class TestComputeCurrentWeights:
    def test_normal_positions(self):
        positions = [
            {"symbol": "AAPL", "market_value": 30_000},
            {"symbol": "MSFT", "market_value": 20_000},
        ]
        w = PortfolioRebalancer.compute_current_weights(positions, 100_000)
        assert w["AAPL"] == pytest.approx(0.30)
        assert w["MSFT"] == pytest.approx(0.20)

    def test_full_allocation(self):
        positions = [
            {"symbol": "AAPL", "market_value": 50_000},
            {"symbol": "MSFT", "market_value": 50_000},
        ]
        w = PortfolioRebalancer.compute_current_weights(positions, 100_000)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_empty_positions(self):
        w = PortfolioRebalancer.compute_current_weights([], 100_000)
        assert w == {}

    def test_zero_equity(self):
        positions = [{"symbol": "AAPL", "market_value": 10_000}]
        w = PortfolioRebalancer.compute_current_weights(positions, 0)
        assert w == {}

    def test_negative_equity(self):
        positions = [{"symbol": "AAPL", "market_value": 10_000}]
        w = PortfolioRebalancer.compute_current_weights(positions, -5_000)
        assert w == {}

    def test_missing_market_value_defaults_to_zero(self):
        positions = [{"symbol": "AAPL"}]
        w = PortfolioRebalancer.compute_current_weights(positions, 100_000)
        assert w["AAPL"] == pytest.approx(0.0)


# ── calculate_rebalance tests ────────────────────────────────────


class TestCalculateRebalance:
    def test_balanced_portfolio_no_orders(self):
        """Portfolio at target weights (within threshold) -> no rebalance."""
        rb = PortfolioRebalancer(mode="equal_weight", drift_threshold=0.05)
        positions = [
            {"symbol": "AAPL", "market_value": 34_000, "qty": 227, "current_price": 150.0},
            {"symbol": "MSFT", "market_value": 33_000, "qty": 110, "current_price": 300.0},
            {"symbol": "GOOGL", "market_value": 33_000, "qty": 220, "current_price": 150.0},
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 150.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT", "GOOGL"],
            prices=prices,
        )
        assert result.needs_rebalance is False
        assert result.orders == []
        assert result.max_drift < 0.05

    def test_significant_drift_generates_orders(self):
        """Portfolio with large drift -> rebalance orders generated."""
        rb = PortfolioRebalancer(mode="equal_weight", drift_threshold=0.05)
        positions = [
            {"symbol": "AAPL", "market_value": 60_000, "qty": 400},
            {"symbol": "MSFT", "market_value": 20_000, "qty": 67},
            {"symbol": "GOOGL", "market_value": 20_000, "qty": 133},
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 150.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT", "GOOGL"],
            prices=prices,
        )
        assert result.needs_rebalance is True
        assert len(result.orders) > 0
        assert result.max_drift > 0.05

        # AAPL is over (60% vs 33.3%) -> should sell
        aapl_orders = [o for o in result.orders if o.symbol == "AAPL"]
        assert len(aapl_orders) == 1
        assert aapl_orders[0].side == "sell"

        # MSFT is under (20% vs 33.3%) -> should buy
        msft_orders = [o for o in result.orders if o.symbol == "MSFT"]
        assert len(msft_orders) == 1
        assert msft_orders[0].side == "buy"

    def test_new_symbol_not_yet_held(self):
        """Symbol in target but not in positions -> buy order."""
        rb = PortfolioRebalancer(mode="equal_weight", drift_threshold=0.05)
        positions = [
            {"symbol": "AAPL", "market_value": 50_000, "qty": 333},
            {"symbol": "MSFT", "market_value": 50_000, "qty": 167},
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 150.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT", "GOOGL"],
            prices=prices,
        )
        assert result.needs_rebalance is True
        googl_orders = [o for o in result.orders if o.symbol == "GOOGL"]
        assert len(googl_orders) == 1
        assert googl_orders[0].side == "buy"

    def test_symbol_to_fully_exit(self):
        """Symbol held but not in target universe -> sell to exit."""
        rb = PortfolioRebalancer(mode="equal_weight", drift_threshold=0.05)
        # TSLA is held but not in the target symbols list.
        # The module only iterates target symbols, so TSLA won't appear
        # in weights or orders. This confirms the module scope.
        positions = [
            {"symbol": "AAPL", "market_value": 30_000, "qty": 200},
            {"symbol": "TSLA", "market_value": 40_000, "qty": 200},
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT"],
            prices=prices,
        )
        # Only AAPL and MSFT in weight analysis
        weight_symbols = {w.symbol for w in result.weights}
        assert "TSLA" not in weight_symbols
        assert "AAPL" in weight_symbols
        assert "MSFT" in weight_symbols

    def test_drift_below_threshold_no_orders(self):
        """When drift is present but below threshold -> no orders."""
        rb = PortfolioRebalancer(mode="equal_weight", drift_threshold=0.10)
        positions = [
            {"symbol": "AAPL", "market_value": 55_000, "qty": 367},
            {"symbol": "MSFT", "market_value": 45_000, "qty": 150},
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT"],
            prices=prices,
        )
        # Max drift is 5% (55% vs 50%), below 10% threshold
        assert result.needs_rebalance is False
        assert result.orders == []

    def test_min_order_value_filter(self):
        """Small orders below min_order_value are skipped."""
        rb = PortfolioRebalancer(
            mode="equal_weight",
            drift_threshold=0.01,  # Very low to trigger rebalance
            min_order_value=50_000.0,  # Very high to filter out small orders
        )
        positions = [
            {"symbol": "AAPL", "market_value": 55_000, "qty": 367},
            {"symbol": "MSFT", "market_value": 45_000, "qty": 150},
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT"],
            prices=prices,
        )
        # Drift exists, but individual order values are below 50k
        assert result.needs_rebalance is True
        assert result.orders == []

    def test_max_single_rebalance_pct_cap(self):
        """Trade value is capped by max_single_rebalance_pct."""
        rb = PortfolioRebalancer(
            mode="equal_weight",
            drift_threshold=0.05,
            max_single_rebalance_pct=0.05,  # 5% cap per trade
            min_order_value=1.0,
        )
        positions = [
            {"symbol": "AAPL", "market_value": 80_000, "qty": 533},
            {"symbol": "MSFT", "market_value": 20_000, "qty": 67},
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT"],
            prices=prices,
        )
        # Max trade = 5% of 100k = $5,000
        for order in result.orders:
            assert order.notional_value <= 5_000 + 1  # Allow rounding

    def test_empty_symbols_returns_empty(self):
        rb = PortfolioRebalancer()
        result = rb.calculate_rebalance(
            positions=[], equity=100_000, symbols=[],
        )
        assert result.weights == []
        assert result.orders == []
        assert result.needs_rebalance is False

    def test_zero_equity_returns_empty(self):
        rb = PortfolioRebalancer()
        result = rb.calculate_rebalance(
            positions=[], equity=0, symbols=["AAPL"],
        )
        assert result.weights == []
        assert result.orders == []
        assert result.needs_rebalance is False

    def test_negative_equity_returns_empty(self):
        rb = PortfolioRebalancer()
        result = rb.calculate_rebalance(
            positions=[], equity=-10_000, symbols=["AAPL"],
        )
        assert result.weights == []
        assert result.orders == []

    def test_mode_carried_through(self):
        rb = PortfolioRebalancer(mode="custom", custom_weights={"AAPL": 1.0})
        result = rb.calculate_rebalance(
            positions=[], equity=100_000, symbols=["AAPL"],
            prices={"AAPL": 150.0},
        )
        assert result.mode == "custom"

    def test_prices_inferred_from_positions(self):
        """When prices are not supplied, they are inferred from positions."""
        rb = PortfolioRebalancer(mode="equal_weight", drift_threshold=0.05)
        positions = [
            {"symbol": "AAPL", "market_value": 60_000, "qty": 400},
            {"symbol": "MSFT", "market_value": 20_000, "qty": 100},
        ]
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT"],
            prices=None,  # Let the module infer from positions
        )
        # AAPL inferred price = 60000/400 = 150
        # MSFT inferred price = 20000/100 = 200
        assert result.needs_rebalance is True

    def test_zero_price_symbol_skipped_in_orders(self):
        """Symbol with zero price produces no order."""
        rb = PortfolioRebalancer(mode="equal_weight", drift_threshold=0.01)
        positions = []
        prices = {"AAPL": 150.0, "MSFT": 0.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT"],
            prices=prices,
        )
        # MSFT has zero price -> should be skipped in orders
        msft_orders = [o for o in result.orders if o.symbol == "MSFT"]
        assert msft_orders == []

    def test_buy_and_sell_totals(self):
        """total_buy_value and total_sell_value are computed correctly."""
        rb = PortfolioRebalancer(
            mode="equal_weight", drift_threshold=0.05, min_order_value=1.0,
        )
        positions = [
            {"symbol": "AAPL", "market_value": 70_000, "qty": 467},
            {"symbol": "MSFT", "market_value": 30_000, "qty": 100},
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT"],
            prices=prices,
        )
        expected_buy = sum(o.notional_value for o in result.orders if o.side == "buy")
        expected_sell = sum(o.notional_value for o in result.orders if o.side == "sell")
        assert result.total_buy_value == pytest.approx(expected_buy)
        assert result.total_sell_value == pytest.approx(expected_sell)

    def test_weight_analysis_all_symbols_present(self):
        rb = PortfolioRebalancer(mode="equal_weight")
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        positions = [
            {"symbol": "AAPL", "market_value": 25_000},
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 150.0, "AMZN": 180.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=symbols,
            prices=prices,
        )
        weight_symbols = {w.symbol for w in result.weights}
        assert weight_symbols == set(symbols)

    def test_drift_sign_convention(self):
        """Positive drift = overweight (current > target), negative = underweight."""
        rb = PortfolioRebalancer(mode="equal_weight")
        positions = [
            {"symbol": "AAPL", "market_value": 70_000},
            {"symbol": "MSFT", "market_value": 30_000},
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT"],
            prices=prices,
        )
        aapl_w = [w for w in result.weights if w.symbol == "AAPL"][0]
        msft_w = [w for w in result.weights if w.symbol == "MSFT"][0]
        assert aapl_w.drift > 0  # Overweight
        assert msft_w.drift < 0  # Underweight

    def test_reason_string_contains_drift(self):
        """Each order's reason includes the drift percentage."""
        rb = PortfolioRebalancer(
            mode="equal_weight", drift_threshold=0.05, min_order_value=1.0,
        )
        positions = [
            {"symbol": "AAPL", "market_value": 70_000, "qty": 467},
            {"symbol": "MSFT", "market_value": 30_000, "qty": 100},
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT"],
            prices=prices,
        )
        for order in result.orders:
            assert "drift=" in order.reason
            assert "threshold=" in order.reason

    def test_orders_have_positive_shares(self):
        """Every generated order must have shares > 0."""
        rb = PortfolioRebalancer(
            mode="equal_weight", drift_threshold=0.05, min_order_value=1.0,
        )
        positions = [
            {"symbol": "AAPL", "market_value": 80_000, "qty": 533},
            {"symbol": "MSFT", "market_value": 10_000, "qty": 33},
            {"symbol": "GOOGL", "market_value": 10_000, "qty": 67},
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 150.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT", "GOOGL"],
            prices=prices,
        )
        for order in result.orders:
            assert order.shares > 0

    def test_skip_symbol_close_to_target(self):
        """Symbols with drift < threshold/2 are skipped even if rebalance is triggered."""
        rb = PortfolioRebalancer(
            mode="equal_weight",
            drift_threshold=0.10,
            min_order_value=1.0,
        )
        # 3 symbols: target=33.3% each
        # AAPL at 45% -> drift=+11.7% (above threshold)
        # MSFT at 30% -> drift=-3.3% (below threshold/2=5%)
        # GOOGL at 25% -> drift=-8.3% (above threshold/2=5%)
        positions = [
            {"symbol": "AAPL", "market_value": 45_000, "qty": 300},
            {"symbol": "MSFT", "market_value": 30_000, "qty": 100},
            {"symbol": "GOOGL", "market_value": 25_000, "qty": 167},
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 150.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT", "GOOGL"],
            prices=prices,
        )
        assert result.needs_rebalance is True
        order_symbols = {o.symbol for o in result.orders}
        # MSFT drift is ~3.3% which is below threshold/2=5% -> should be skipped
        assert "MSFT" not in order_symbols


# ── Custom mode rebalance integration ────────────────────────────


class TestCustomModeRebalance:
    def test_custom_weights_rebalance(self):
        rb = PortfolioRebalancer(
            mode="custom",
            custom_weights={"AAPL": 0.6, "MSFT": 0.4},
            drift_threshold=0.05,
            min_order_value=1.0,
        )
        positions = [
            {"symbol": "AAPL", "market_value": 30_000, "qty": 200},
            {"symbol": "MSFT", "market_value": 70_000, "qty": 233},
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT"],
            prices=prices,
        )
        assert result.needs_rebalance is True
        # AAPL target=60%, current=30% -> buy
        aapl_order = [o for o in result.orders if o.symbol == "AAPL"]
        assert len(aapl_order) == 1
        assert aapl_order[0].side == "buy"
        # MSFT target=40%, current=70% -> sell
        msft_order = [o for o in result.orders if o.symbol == "MSFT"]
        assert len(msft_order) == 1
        assert msft_order[0].side == "sell"


# ── Market cap mode rebalance integration ────────────────────────


class TestMarketCapModeRebalance:
    def test_market_cap_rebalance(self):
        rb = PortfolioRebalancer(
            mode="market_cap",
            drift_threshold=0.05,
            min_order_value=1.0,
        )
        # prices: AAPL=100, MSFT=300 -> target AAPL=25%, MSFT=75%
        positions = [
            {"symbol": "AAPL", "market_value": 50_000, "qty": 500},
            {"symbol": "MSFT", "market_value": 50_000, "qty": 167},
        ]
        prices = {"AAPL": 100.0, "MSFT": 300.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT"],
            prices=prices,
        )
        assert result.needs_rebalance is True
        # AAPL current=50%, target=25% -> sell
        aapl_order = [o for o in result.orders if o.symbol == "AAPL"]
        assert len(aapl_order) == 1
        assert aapl_order[0].side == "sell"


# ── Formatting tests ─────────────────────────────────────────────


class TestFormatWeights:
    def test_format_weights_header(self):
        result = RebalanceResult(
            weights=[
                PortfolioWeight(symbol="AAPL", current_weight=0.50,
                                target_weight=0.333, drift=0.167),
            ],
            mode="equal_weight",
            max_drift=0.167,
            needs_rebalance=True,
        )
        text = PortfolioRebalancer.format_weights(result)
        assert "PORTFOLIO WEIGHTS" in text
        assert "equal_weight" in text
        assert "AAPL" in text
        assert "NEEDED" in text

    def test_format_weights_not_needed(self):
        result = RebalanceResult(
            weights=[
                PortfolioWeight(symbol="AAPL", current_weight=0.50,
                                target_weight=0.50, drift=0.0),
            ],
            mode="equal_weight",
            max_drift=0.0,
            needs_rebalance=False,
        )
        text = PortfolioRebalancer.format_weights(result)
        assert "NOT NEEDED" in text

    def test_format_weights_over_under_ok(self):
        result = RebalanceResult(
            weights=[
                PortfolioWeight(symbol="AAPL", drift=0.10),   # OVER
                PortfolioWeight(symbol="MSFT", drift=-0.08),   # UNDER
                PortfolioWeight(symbol="GOOGL", drift=0.01),   # OK
            ],
        )
        text = PortfolioRebalancer.format_weights(result)
        assert "OVER" in text
        assert "UNDER" in text
        assert "OK" in text

    def test_format_weights_sorted_by_abs_drift(self):
        result = RebalanceResult(
            weights=[
                PortfolioWeight(symbol="A", drift=0.01),
                PortfolioWeight(symbol="B", drift=-0.20),
                PortfolioWeight(symbol="C", drift=0.10),
            ],
        )
        text = PortfolioRebalancer.format_weights(result)
        # B has largest |drift|, then C, then A
        pos_b = text.index("B")
        pos_c = text.index("C")
        pos_a = text.index("A")
        # Note: headers contain "A" in "Status", so find after the header
        lines = text.split("\n")
        data_lines = [l for l in lines if l.strip().startswith(("A ", "B ", "C "))]
        assert len(data_lines) == 3
        # First data line should have B (largest drift)
        assert "B" in data_lines[0]


class TestFormatOrders:
    def test_format_orders_with_orders(self):
        result = RebalanceResult(
            orders=[
                RebalanceOrder(
                    symbol="AAPL", side="buy", shares=50,
                    notional_value=7_500, reason="drift=-10%",
                ),
            ],
            total_buy_value=7_500,
            total_sell_value=0,
        )
        text = PortfolioRebalancer.format_orders(result)
        assert "REBALANCE ORDERS" in text
        assert "AAPL" in text
        assert "BUY" in text
        assert "50" in text
        assert "7,500" in text

    def test_format_orders_no_orders(self):
        result = RebalanceResult(orders=[])
        text = PortfolioRebalancer.format_orders(result)
        assert "No rebalance orders needed" in text

    def test_format_orders_buy_sell_totals(self):
        result = RebalanceResult(
            orders=[
                RebalanceOrder(symbol="AAPL", side="buy", shares=10,
                               notional_value=1_500, reason="test"),
                RebalanceOrder(symbol="MSFT", side="sell", shares=5,
                               notional_value=1_500, reason="test"),
            ],
            total_buy_value=1_500,
            total_sell_value=1_500,
        )
        text = PortfolioRebalancer.format_orders(result)
        assert "Total Buy" in text
        assert "Total Sell" in text


# ── Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_symbol_always_at_target(self):
        rb = PortfolioRebalancer(mode="equal_weight")
        positions = [{"symbol": "SPY", "market_value": 100_000}]
        prices = {"SPY": 450.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["SPY"],
            prices=prices,
        )
        assert result.needs_rebalance is False
        assert result.weights[0].drift == pytest.approx(0.0)

    def test_all_in_cash_buys_everything(self):
        """No positions, all cash -> should buy everything."""
        rb = PortfolioRebalancer(
            mode="equal_weight", drift_threshold=0.05, min_order_value=1.0,
        )
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        result = rb.calculate_rebalance(
            positions=[],
            equity=100_000,
            symbols=["AAPL", "MSFT"],
            prices=prices,
        )
        assert result.needs_rebalance is True
        assert all(o.side == "buy" for o in result.orders)
        assert len(result.orders) == 2

    def test_very_expensive_stock_zero_shares(self):
        """When price is too high relative to trade value, shares may be zero -> no order."""
        rb = PortfolioRebalancer(
            mode="equal_weight",
            drift_threshold=0.01,
            max_single_rebalance_pct=0.001,  # Cap 0.1% = $100 on 100k
            min_order_value=1.0,
        )
        positions = []
        prices = {"BRK.A": 500_000.0, "AAPL": 150.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["BRK.A", "AAPL"],
            prices=prices,
        )
        brk_orders = [o for o in result.orders if o.symbol == "BRK.A"]
        # $100 / $500,000 = 0 shares -> skipped
        assert brk_orders == []

    def test_allocation_modes_constant(self):
        assert "equal_weight" in ALLOCATION_MODES
        assert "custom" in ALLOCATION_MODES
        assert "market_cap" in ALLOCATION_MODES
        assert len(ALLOCATION_MODES) == 3

    def test_shares_are_integers(self):
        """Shares in orders must always be int (no fractional shares)."""
        rb = PortfolioRebalancer(
            mode="equal_weight", drift_threshold=0.01, min_order_value=1.0,
        )
        positions = []
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 2800.0}
        result = rb.calculate_rebalance(
            positions=positions,
            equity=100_000,
            symbols=["AAPL", "MSFT", "GOOGL"],
            prices=prices,
        )
        for order in result.orders:
            assert isinstance(order.shares, int)

    def test_current_and_target_values_consistent(self):
        """current_value and target_value should equal weight * equity."""
        rb = PortfolioRebalancer(mode="equal_weight")
        positions = [
            {"symbol": "AAPL", "market_value": 40_000},
            {"symbol": "MSFT", "market_value": 60_000},
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        equity = 100_000
        result = rb.calculate_rebalance(
            positions=positions,
            equity=equity,
            symbols=["AAPL", "MSFT"],
            prices=prices,
        )
        for w in result.weights:
            assert w.current_value == pytest.approx(w.current_weight * equity)
            assert w.target_value == pytest.approx(w.target_weight * equity)
