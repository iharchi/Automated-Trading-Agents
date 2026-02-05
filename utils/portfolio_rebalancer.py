"""Portfolio Rebalancer

Calculates portfolio drift from target allocations and generates
rebalance orders to bring the portfolio back in line.

Allocation modes:
    equal_weight — Each symbol gets 1/N of equity
    custom       — User-specified target weights
    market_cap   — Proxy weights based on relative price (approximation)

Usage:
    rebalancer = PortfolioRebalancer(mode="equal_weight")
    orders = rebalancer.calculate_rebalance(
        positions=positions,
        equity=100000,
        symbols=["AAPL", "MSFT", "GOOGL"],
    )
    for o in orders:
        print(PortfolioRebalancer.format_order(o))
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

ALLOCATION_MODES = ["equal_weight", "custom", "market_cap"]


@dataclass
class PortfolioWeight:
    """Current vs target weight for one symbol."""
    symbol: str
    current_weight: float = 0.0
    target_weight: float = 0.0
    drift: float = 0.0          # current - target
    current_value: float = 0.0
    target_value: float = 0.0


@dataclass
class RebalanceOrder:
    """A proposed rebalance trade."""
    symbol: str
    side: str               # "buy" or "sell"
    shares: int = 0
    notional_value: float = 0.0
    current_weight: float = 0.0
    target_weight: float = 0.0
    price: float = 0.0
    reason: str = ""


@dataclass
class RebalanceResult:
    """Full rebalance analysis."""
    weights: list[PortfolioWeight] = field(default_factory=list)
    orders: list[RebalanceOrder] = field(default_factory=list)
    max_drift: float = 0.0
    needs_rebalance: bool = False
    total_buy_value: float = 0.0
    total_sell_value: float = 0.0
    mode: str = "equal_weight"


class PortfolioRebalancer:
    """Portfolio rebalancer with configurable allocation targets."""

    def __init__(
        self,
        *,
        mode: str = "equal_weight",
        custom_weights: dict[str, float] | None = None,
        drift_threshold: float = 0.05,
        max_single_rebalance_pct: float = 0.25,
        min_order_value: float = 100.0,
    ):
        """
        Args:
            mode: Allocation mode (equal_weight, custom, market_cap).
            custom_weights: Dict of symbol → target weight (for custom mode).
            drift_threshold: Minimum drift to trigger rebalance (0.05 = 5%).
            max_single_rebalance_pct: Max % of equity to trade in one rebalance.
            min_order_value: Minimum notional value per order (skip tiny trades).
        """
        if mode not in ALLOCATION_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Use one of {ALLOCATION_MODES}")

        self.mode = mode
        self.custom_weights = custom_weights or {}
        self.drift_threshold = drift_threshold
        self.max_single_rebalance_pct = max_single_rebalance_pct
        self.min_order_value = min_order_value

    # ── Target weight calculation ────────────────────────────────

    def compute_target_weights(
        self,
        symbols: list[str],
        prices: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute target weight for each symbol.

        Returns:
            Dict of symbol → target weight (0.0 to 1.0, summing to 1.0).
        """
        if self.mode == "equal_weight":
            n = len(symbols)
            if n == 0:
                return {}
            w = 1.0 / n
            return {s: w for s in symbols}

        elif self.mode == "custom":
            # Normalise custom weights to sum to 1
            total = sum(self.custom_weights.get(s, 0) for s in symbols)
            if total <= 0:
                return self.compute_target_weights(symbols)  # fallback to equal
            return {s: self.custom_weights.get(s, 0) / total for s in symbols}

        elif self.mode == "market_cap":
            # Proxy: use relative price as a rough market-cap proxy
            if not prices:
                return self.compute_target_weights(symbols)  # fallback
            total_price = sum(prices.get(s, 0) for s in symbols)
            if total_price <= 0:
                return self.compute_target_weights(symbols)
            return {s: prices.get(s, 0) / total_price for s in symbols}

        return {}

    # ── Current weight calculation ───────────────────────────────

    @staticmethod
    def compute_current_weights(
        positions: list[dict],
        equity: float,
    ) -> dict[str, float]:
        """Compute current portfolio weights from positions.

        Args:
            positions: List of position dicts with 'symbol', 'market_value'.
            equity: Total portfolio equity.

        Returns:
            Dict of symbol → current weight.
        """
        if equity <= 0:
            return {}
        return {
            p["symbol"]: p.get("market_value", 0) / equity
            for p in positions
        }

    # ── Rebalance calculation ────────────────────────────────────

    def calculate_rebalance(
        self,
        *,
        positions: list[dict],
        equity: float,
        symbols: list[str],
        prices: dict[str, float] | None = None,
    ) -> RebalanceResult:
        """Calculate portfolio drift and generate rebalance orders.

        Args:
            positions: Current positions from broker.
            equity: Total portfolio equity.
            symbols: Target universe of symbols.
            prices: Current prices (required for share calculations).

        Returns:
            RebalanceResult with weights, orders, and analysis.
        """
        if not symbols or equity <= 0:
            return RebalanceResult(mode=self.mode)

        # Build price map from positions if not provided
        if prices is None:
            prices = {}
            for p in positions:
                mv = p.get("market_value", 0)
                qty = p.get("qty", 0)
                if qty > 0 and mv > 0:
                    prices[p["symbol"]] = mv / qty

        target_weights = self.compute_target_weights(symbols, prices)
        current_weights = self.compute_current_weights(positions, equity)

        # Build weight analysis
        weights: list[PortfolioWeight] = []
        for symbol in symbols:
            tw = target_weights.get(symbol, 0)
            cw = current_weights.get(symbol, 0)
            drift = cw - tw
            weights.append(PortfolioWeight(
                symbol=symbol,
                current_weight=cw,
                target_weight=tw,
                drift=drift,
                current_value=cw * equity,
                target_value=tw * equity,
            ))

        max_drift = max(abs(w.drift) for w in weights) if weights else 0.0
        needs_rebalance = max_drift >= self.drift_threshold

        # Generate orders
        orders: list[RebalanceOrder] = []
        if needs_rebalance:
            for w in weights:
                if abs(w.drift) < self.drift_threshold / 2:
                    continue  # Skip symbols close to target

                trade_value = abs(w.drift) * equity
                # Cap single trade
                max_trade = self.max_single_rebalance_pct * equity
                trade_value = min(trade_value, max_trade)

                if trade_value < self.min_order_value:
                    continue

                price = prices.get(w.symbol, 0)
                if price <= 0:
                    continue

                shares = int(trade_value / price)
                if shares <= 0:
                    continue

                side = "sell" if w.drift > 0 else "buy"
                reason = f"drift={w.drift:+.2%} (threshold={self.drift_threshold:.2%})"

                orders.append(RebalanceOrder(
                    symbol=w.symbol,
                    side=side,
                    shares=shares,
                    notional_value=shares * price,
                    current_weight=w.current_weight,
                    target_weight=w.target_weight,
                    price=price,
                    reason=reason,
                ))

        total_buy = sum(o.notional_value for o in orders if o.side == "buy")
        total_sell = sum(o.notional_value for o in orders if o.side == "sell")

        return RebalanceResult(
            weights=weights,
            orders=orders,
            max_drift=max_drift,
            needs_rebalance=needs_rebalance,
            total_buy_value=total_buy,
            total_sell_value=total_sell,
            mode=self.mode,
        )

    # ── Formatting ───────────────────────────────────────────────

    @staticmethod
    def format_weights(result: RebalanceResult) -> str:
        """Format portfolio weights table."""
        lines = [
            f"\n{'=' * 70}",
            f"  PORTFOLIO WEIGHTS ({result.mode})",
            f"{'=' * 70}",
            f"  {'Symbol':8s} {'Current':>10s} {'Target':>10s} {'Drift':>10s} {'Status':>10s}",
            f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}",
        ]

        for w in sorted(result.weights, key=lambda x: abs(x.drift), reverse=True):
            status = "OK"
            if abs(w.drift) >= 0.05:
                status = "OVER" if w.drift > 0 else "UNDER"
            lines.append(
                f"  {w.symbol:8s} {w.current_weight:>9.1%} {w.target_weight:>9.1%} "
                f"{w.drift:>+9.1%} {status:>10s}"
            )

        lines.append(f"{'~' * 70}")
        lines.append(f"  Max Drift: {result.max_drift:.1%}  |  "
                     f"Rebalance: {'NEEDED' if result.needs_rebalance else 'NOT NEEDED'}")
        lines.append(f"{'=' * 70}\n")
        return "\n".join(lines)

    @staticmethod
    def format_orders(result: RebalanceResult) -> str:
        """Format rebalance orders."""
        if not result.orders:
            return "\n  No rebalance orders needed.\n"

        lines = [
            f"\n{'=' * 70}",
            f"  REBALANCE ORDERS",
            f"{'=' * 70}",
            f"  {'Symbol':8s} {'Side':6s} {'Shares':>8s} {'Value':>12s} {'Reason'}",
            f"  {'-'*8} {'-'*6} {'-'*8} {'-'*12} {'-'*25}",
        ]

        for o in result.orders:
            lines.append(
                f"  {o.symbol:8s} {o.side.upper():6s} {o.shares:>8d} "
                f"${o.notional_value:>11,.2f} {o.reason}"
            )

        lines.append(f"{'~' * 70}")
        lines.append(
            f"  Total Buy: ${result.total_buy_value:,.2f}  |  "
            f"Total Sell: ${result.total_sell_value:,.2f}"
        )
        lines.append(f"{'=' * 70}\n")
        return "\n".join(lines)
