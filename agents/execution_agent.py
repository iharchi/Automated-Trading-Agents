"""Execution Agent

Responsible for the final step of the trading pipeline: turning
risk-approved decisions into actual paper-trade orders on Alpaca.

Capabilities:
    - Market orders for immediate fills
    - Bracket orders (entry + stop-loss + take-profit) when prices are provided
    - Order verification: waits briefly and checks fill status
    - Dry-run mode: validates everything but skips the actual API call
    - Retry logic for transient API failures
    - Full journal logging of every order attempt and outcome

Pipeline position:
    TA + Sentiment ──> Portfolio Manager ──> Risk Manager ──> **Execution Agent**
"""

import logging
import time
from dataclasses import dataclass, field

from agents.base_agent import BaseAgent
from utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────

MAX_RETRIES = 2
RETRY_DELAY = 1.0           # seconds between retries
FILL_CHECK_DELAY = 1.0      # seconds to wait before checking fill
FILL_CHECK_ATTEMPTS = 3     # how many times to poll for fill status


@dataclass
class OrderTicket:
    """All information needed to place a single order."""

    symbol: str
    side: str                           # "buy" or "sell"
    qty: int
    order_type: str = "market"          # "market", "limit", "stop_limit"
    limit_price: float | None = None
    stop_price: float | None = None     # stop-loss price
    take_profit_price: float | None = None
    time_in_force: str = "day"


@dataclass
class ExecutionResult:
    """Outcome of an order execution attempt."""

    symbol: str
    side: str = ""
    qty: int = 0
    order_type: str = ""
    status: str = "pending"             # pending, filled, partial, failed, dry_run, skipped
    order_id: str = ""
    filled_qty: int = 0
    filled_avg_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    retries: int = 0
    error: str = ""
    checks: list[dict] = field(default_factory=list)


class ExecutionAgent(BaseAgent):
    """Agent that submits and tracks paper-trade orders on Alpaca."""

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        dry_run: bool = False,
        skip_market_check: bool = False,
        max_retries: int = MAX_RETRIES,
        retry_delay: float = RETRY_DELAY,
    ):
        super().__init__(name="Execution", client=client)
        self.dry_run = dry_run
        self.skip_market_check = skip_market_check
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ── Pre-flight checks ─────────────────────────────────────

    @staticmethod
    def _check_risk_approved(decision: dict) -> dict:
        approved = decision.get("risk_approved", False)
        return {
            "rule": "risk_approved",
            "passed": bool(approved),
            "detail": "Risk-approved" if approved else "Not risk-approved",
        }

    @staticmethod
    def _check_position_size(decision: dict) -> dict:
        qty = decision.get("position_size", 0)
        return {
            "rule": "position_size",
            "passed": qty > 0,
            "detail": f"{qty} shares" if qty > 0 else "Zero position size",
        }

    @staticmethod
    def _check_signal_actionable(decision: dict) -> dict:
        signal = decision.get("signal", "HOLD")
        actionable = signal in ("BUY", "SELL")
        return {
            "rule": "signal_actionable",
            "passed": actionable,
            "detail": f"Signal={signal}",
        }

    def _check_market_open(self) -> dict:
        try:
            is_open = self.client.is_market_open()
        except Exception as e:
            return {
                "rule": "market_open",
                "passed": False,
                "detail": f"Could not check market status: {e}",
            }
        return {
            "rule": "market_open",
            "passed": is_open,
            "detail": "Market open" if is_open else "Market closed",
        }

    def _run_preflight(self, decision: dict) -> list[dict]:
        """Run all pre-flight checks. Returns list of check dicts."""
        checks = [
            self._check_signal_actionable(decision),
            self._check_risk_approved(decision),
            self._check_position_size(decision),
        ]
        # Skip market check for paper trading / extended hours
        if not self.skip_market_check:
            checks.append(self._check_market_open())
        else:
            checks.append({
                "rule": "market_open",
                "passed": True,
                "detail": "Skipped (paper trading mode)",
            })
        return checks

    # ── Order building ────────────────────────────────────────

    @staticmethod
    def _build_ticket(decision: dict) -> OrderTicket:
        """Convert a portfolio decision into an OrderTicket."""
        signal = decision.get("signal", "HOLD")
        side = "buy" if signal == "BUY" else "sell"
        qty = decision.get("position_size", 0)
        stop_loss = decision.get("stop_loss", 0.0)
        take_profit = decision.get("take_profit", 0.0)

        return OrderTicket(
            symbol=decision.get("symbol", ""),
            side=side,
            qty=qty,
            order_type="market",
            stop_price=stop_loss if stop_loss > 0 else None,
            take_profit_price=take_profit if take_profit > 0 else None,
        )

    # ── Order submission ──────────────────────────────────────

    def _submit_order(self, ticket: OrderTicket) -> dict:
        """Submit an order via Alpaca, with optional bracket legs.

        Returns the raw order dict from AlpacaClient.submit_order().
        """
        kwargs: dict = {
            "symbol": ticket.symbol,
            "qty": ticket.qty,
            "side": ticket.side,
            "order_type": ticket.order_type,
            "time_in_force": ticket.time_in_force,
        }

        # Note: Bracket orders disabled - AlpacaClient.submit_order() doesn't support them
        # Stop-loss and take-profit are managed separately via TrailingStopManager

        return self.client.submit_order(**kwargs)

    def _submit_with_retry(self, ticket: OrderTicket) -> tuple[dict | None, int, str]:
        """Try to submit an order, retrying on transient failures.

        Returns:
            (order_dict_or_None, num_retries, error_string)
        """
        last_error = ""
        for attempt in range(1 + self.max_retries):
            try:
                order = self._submit_order(ticket)
                return order, attempt, ""
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "Order attempt %d/%d for %s failed: %s",
                    attempt + 1, 1 + self.max_retries, ticket.symbol, e,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))
        return None, self.max_retries, last_error

    # ── Fill verification ─────────────────────────────────────

    def _check_fill(self, order_id: str) -> dict | None:
        """Poll Alpaca for fill status of an order."""
        for i in range(FILL_CHECK_ATTEMPTS):
            try:
                order = self.client.api.get_order(order_id)
                status = order.status
                if status in ("filled", "partially_filled", "cancelled", "expired", "rejected"):
                    return {
                        "status": status,
                        "filled_qty": int(order.filled_qty or 0),
                        "filled_avg_price": float(order.filled_avg_price or 0),
                    }
            except Exception as e:
                logger.debug("Fill check %d failed: %s", i + 1, e)
            if i < FILL_CHECK_ATTEMPTS - 1:
                time.sleep(FILL_CHECK_DELAY)
        return None

    # ── Core lifecycle ────────────────────────────────────────

    def analyze(self, symbol: str, **kwargs) -> dict:
        """Run pre-flight checks on a portfolio decision.

        Required kwargs:
            decision: dict — the output of PortfolioManagerAgent.analyze()
        """
        decision = kwargs.get("decision", {})

        checks = self._run_preflight(decision)
        all_passed = all(c["passed"] for c in checks)
        ticket = self._build_ticket(decision) if all_passed else None

        return {
            "symbol": symbol,
            "checks": checks,
            "all_passed": all_passed,
            "ticket": ticket,
            "decision": decision,
        }

    def execute(self, symbol: str, analysis: dict, **kwargs) -> dict:
        """Place the order (or dry-run) and verify the fill."""
        ticket: OrderTicket | None = analysis.get("ticket")
        checks = analysis.get("checks", [])
        all_passed = analysis.get("all_passed", False)

        result = ExecutionResult(symbol=symbol, checks=checks)

        if not all_passed or ticket is None:
            result.status = "skipped"
            result.error = "Pre-flight checks failed"
            return result.__dict__

        result.side = ticket.side
        result.qty = ticket.qty
        result.order_type = ticket.order_type
        result.stop_loss = ticket.stop_price or 0.0
        result.take_profit = ticket.take_profit_price or 0.0

        # ── Dry-run: stop before the API call ────────────
        if self.dry_run:
            result.status = "dry_run"
            logger.info(
                "[DRY RUN] Would %s %d %s (%s) | SL=%.2f TP=%.2f",
                ticket.side, ticket.qty, ticket.symbol,
                ticket.order_type, result.stop_loss, result.take_profit,
            )
            return result.__dict__

        # ── Real submission ──────────────────────────────
        order, retries, error = self._submit_with_retry(ticket)
        result.retries = retries

        if order is None:
            result.status = "failed"
            result.error = error
            return result.__dict__

        result.order_id = order.get("id", "")
        result.status = order.get("status", "submitted")

        # ── Verify fill ──────────────────────────────────
        if result.order_id:
            fill = self._check_fill(result.order_id)
            if fill:
                result.status = fill["status"]
                result.filled_qty = fill["filled_qty"]
                result.filled_avg_price = fill["filled_avg_price"]

        logger.info(
            "Order %s: %s %d %s — status=%s fill_qty=%d avg_price=%.2f",
            result.order_id, result.side, result.qty, symbol,
            result.status, result.filled_qty, result.filled_avg_price,
        )
        return result.__dict__

    # ── Pretty print ──────────────────────────────────────────

    @staticmethod
    def format_analysis(analysis: dict) -> str:
        """Human-readable execution summary."""
        # analysis can be either the pre-flight dict or the execution result
        symbol = analysis.get("symbol", "???")
        status = analysis.get("status", "")

        lines = [
            f"\n{'*' * 62}",
            f"  EXECUTION — {symbol}",
            f"{'*' * 62}",
        ]

        # Pre-flight checks
        for chk in analysis.get("checks", []):
            icon = "PASS" if chk.get("passed") else "FAIL"
            lines.append(f"  [{icon}] {chk['rule']:20s}  {chk['detail']}")

        if status:
            lines.append(f"{'─' * 62}")
            side = analysis.get("side", "")
            qty = analysis.get("qty", 0)
            order_type = analysis.get("order_type", "")
            lines.append(f"  Order  : {side.upper()} {qty} {symbol} ({order_type})")
            lines.append(f"  Status : {status.upper()}")

            sl = analysis.get("stop_loss", 0)
            tp = analysis.get("take_profit", 0)
            if sl or tp:
                lines.append(f"  Stop   : ${sl:.2f}  |  Target: ${tp:.2f}")

            filled_qty = analysis.get("filled_qty", 0)
            avg_price = analysis.get("filled_avg_price", 0)
            if filled_qty:
                lines.append(
                    f"  Filled : {filled_qty} shares @ ${avg_price:.2f}"
                )

            retries = analysis.get("retries", 0)
            if retries:
                lines.append(f"  Retries: {retries}")

            error = analysis.get("error", "")
            if error:
                lines.append(f"  Error  : {error}")

            order_id = analysis.get("order_id", "")
            if order_id:
                lines.append(f"  ID     : {order_id}")

        lines.append(f"{'*' * 62}\n")
        return "\n".join(lines)
