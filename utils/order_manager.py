"""Order Management System (OMS)

Tracks the full lifecycle of orders: creation, submission, partial fills,
completion, and cancellation.  Reconciles local state with broker state
and maintains a persistent order history.

Usage:
    oms = OrderManager(client)
    oid = oms.create_order("AAPL", "buy", 50, price=150.0)
    oms.submit(oid)
    oms.check_fills()
    print(OrderManager.format_blotter(oms.get_blotter()))
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    """Order lifecycle states."""
    CREATED = "created"
    SUBMITTED = "submitted"
    PARTIAL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class Order:
    """Internal order representation."""
    order_id: str
    symbol: str
    side: str                   # "buy" or "sell"
    qty: int
    order_type: str = "market"  # market, limit, stop, stop_limit
    time_in_force: str = "day"  # day, gtc, ioc, fok
    limit_price: float | None = None
    stop_price: float | None = None

    # State
    status: str = OrderStatus.CREATED
    filled_qty: int = 0
    filled_avg_price: float = 0.0
    broker_order_id: str = ""

    # Timestamps
    created_at: float = 0.0
    submitted_at: float = 0.0
    filled_at: float = 0.0
    cancelled_at: float = 0.0

    # Metadata
    reason: str = ""            # Why this order was created
    source: str = ""            # Which component created it
    error: str = ""

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    @property
    def is_terminal(self) -> bool:
        """Order is in a final state."""
        return self.status in (
            OrderStatus.FILLED, OrderStatus.CANCELLED,
            OrderStatus.REJECTED, OrderStatus.FAILED,
            OrderStatus.EXPIRED,
        )

    @property
    def remaining_qty(self) -> int:
        return self.qty - self.filled_qty

    @property
    def fill_pct(self) -> float:
        return self.filled_qty / self.qty if self.qty > 0 else 0.0


class OrderManager:
    """Manages order lifecycle and reconciliation with the broker."""

    def __init__(self, client=None):
        """
        Args:
            client: AlpacaClient instance (None for offline/testing).
        """
        self.client = client
        self._orders: dict[str, Order] = {}
        self._next_id = 1

    # ── Order creation ───────────────────────────────────────────

    def create_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        *,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        stop_price: float | None = None,
        reason: str = "",
        source: str = "",
    ) -> str:
        """Create a new order (not yet submitted).

        Returns:
            Internal order ID string.
        """
        oid = f"oms-{self._next_id:06d}"
        self._next_id += 1

        order = Order(
            order_id=oid,
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            reason=reason,
            source=source,
        )
        self._orders[oid] = order
        logger.info("Order created: %s %s %d %s (%s)", oid, side, qty, symbol, order_type)
        return oid

    # ── Submission ───────────────────────────────────────────────

    def submit(self, order_id: str) -> bool:
        """Submit an order to the broker.

        Returns:
            True if submitted successfully.
        """
        order = self._orders.get(order_id)
        if not order:
            logger.error("Order %s not found", order_id)
            return False

        if order.status != OrderStatus.CREATED:
            logger.warning("Cannot submit order %s in status %s", order_id, order.status)
            return False

        if self.client is None:
            # Dry-run: simulate immediate fill
            order.status = OrderStatus.FILLED
            order.filled_qty = order.qty
            order.filled_avg_price = order.limit_price or 0.0
            order.submitted_at = time.time()
            order.filled_at = time.time()
            order.broker_order_id = f"dry-{order_id}"
            logger.info("Order %s dry-run filled", order_id)
            return True

        try:
            result = self.client.submit_order(
                symbol=order.symbol,
                qty=order.qty,
                side=order.side,
                order_type=order.order_type,
                time_in_force=order.time_in_force,
            )
            order.broker_order_id = result.get("id", "")
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = time.time()

            broker_status = result.get("status", "")
            if broker_status == "filled":
                order.status = OrderStatus.FILLED
                order.filled_qty = order.qty
                order.filled_at = time.time()
            elif broker_status == "partially_filled":
                order.status = OrderStatus.PARTIAL

            logger.info("Order %s submitted → broker_id=%s", order_id, order.broker_order_id)
            return True

        except Exception as e:
            order.status = OrderStatus.FAILED
            order.error = str(e)
            logger.error("Order %s submission failed: %s", order_id, e)
            return False

    # ── Cancel ───────────────────────────────────────────────────

    def cancel(self, order_id: str) -> bool:
        """Cancel an open order.

        Returns:
            True if cancellation was successful.
        """
        order = self._orders.get(order_id)
        if not order:
            return False

        if order.is_terminal:
            logger.warning("Cannot cancel terminal order %s", order_id)
            return False

        if self.client and order.broker_order_id:
            try:
                self.client.api.cancel_order(order.broker_order_id)
            except Exception as e:
                logger.warning("Broker cancel failed for %s: %s", order_id, e)

        order.status = OrderStatus.CANCELLED
        order.cancelled_at = time.time()
        logger.info("Order %s cancelled", order_id)
        return True

    # ── Fill checks ──────────────────────────────────────────────

    def check_fills(self) -> list[str]:
        """Check broker for fill updates on open orders.

        Returns:
            List of order IDs that were updated.
        """
        updated = []

        for oid, order in self._orders.items():
            if order.is_terminal or not order.broker_order_id:
                continue
            if self.client is None:
                continue

            try:
                broker_order = self.client.api.get_order(order.broker_order_id)
                new_status = broker_order.status

                if new_status == "filled":
                    order.status = OrderStatus.FILLED
                    order.filled_qty = int(broker_order.filled_qty or order.qty)
                    order.filled_avg_price = float(broker_order.filled_avg_price or 0)
                    order.filled_at = time.time()
                    updated.append(oid)
                elif new_status == "partially_filled":
                    order.status = OrderStatus.PARTIAL
                    order.filled_qty = int(broker_order.filled_qty or 0)
                    order.filled_avg_price = float(broker_order.filled_avg_price or 0)
                    updated.append(oid)
                elif new_status in ("cancelled", "canceled"):
                    order.status = OrderStatus.CANCELLED
                    order.cancelled_at = time.time()
                    updated.append(oid)
                elif new_status == "expired":
                    order.status = OrderStatus.EXPIRED
                    updated.append(oid)
                elif new_status == "rejected":
                    order.status = OrderStatus.REJECTED
                    order.error = "Rejected by broker"
                    updated.append(oid)

            except Exception as e:
                logger.warning("Fill check failed for %s: %s", oid, e)

        return updated

    # ── Reconciliation ───────────────────────────────────────────

    def reconcile(self) -> dict:
        """Reconcile local orders with broker state.

        Returns:
            Dict with reconciliation stats.
        """
        if self.client is None:
            return {"reconciled": 0, "mismatches": 0}

        open_orders = [o for o in self._orders.values() if not o.is_terminal]
        reconciled = 0
        mismatches = 0

        for order in open_orders:
            if not order.broker_order_id:
                continue
            try:
                broker_order = self.client.api.get_order(order.broker_order_id)
                broker_status = broker_order.status

                if broker_status == "filled" and order.status != OrderStatus.FILLED:
                    order.status = OrderStatus.FILLED
                    order.filled_qty = int(broker_order.filled_qty or order.qty)
                    order.filled_avg_price = float(broker_order.filled_avg_price or 0)
                    order.filled_at = time.time()
                    mismatches += 1

                reconciled += 1
            except Exception as e:
                logger.warning("Reconcile failed for %s: %s", order.order_id, e)

        return {"reconciled": reconciled, "mismatches": mismatches}

    # ── Querying ─────────────────────────────────────────────────

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def get_open_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if not o.is_terminal]

    def get_filled_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.status == OrderStatus.FILLED]

    def get_orders_by_symbol(self, symbol: str) -> list[Order]:
        return [o for o in self._orders.values() if o.symbol == symbol]

    def get_blotter(self, limit: int = 50) -> list[Order]:
        """Return recent orders sorted by creation time (newest first)."""
        return sorted(
            self._orders.values(),
            key=lambda o: o.created_at,
            reverse=True,
        )[:limit]

    @property
    def total_orders(self) -> int:
        return len(self._orders)

    @property
    def open_count(self) -> int:
        return len(self.get_open_orders())

    @property
    def filled_count(self) -> int:
        return len(self.get_filled_orders())

    # ── Formatting ───────────────────────────────────────────────

    @staticmethod
    def format_order(order: Order) -> str:
        import time as _time
        ts = _time.strftime("%H:%M:%S", _time.localtime(order.created_at))
        fill = f" filled={order.filled_qty}/{order.qty}" if order.filled_qty > 0 else ""
        price = f" @${order.filled_avg_price:.2f}" if order.filled_avg_price > 0 else ""
        return (
            f"  [{order.status.upper():10s}] {ts} {order.order_id} "
            f"{order.side.upper()} {order.qty} {order.symbol} "
            f"({order.order_type}){fill}{price}"
        )

    @staticmethod
    def format_blotter(orders: list[Order]) -> str:
        if not orders:
            return "\n  No orders.\n"
        lines = [
            f"\n{'=' * 70}",
            f"  ORDER BLOTTER ({len(orders)} orders)",
            f"{'=' * 70}",
        ]
        for o in orders:
            lines.append(OrderManager.format_order(o))
        lines.append(f"{'=' * 70}\n")
        return "\n".join(lines)
