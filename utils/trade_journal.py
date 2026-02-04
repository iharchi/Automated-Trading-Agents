"""Trade Journal

Append-only CSV logger that records every signal, portfolio decision,
and executed order.  Each run appends rows — the file is never
overwritten, providing a full audit trail.

Files written (under --journal-dir, default ./journal/):
    signals.csv     — per-agent signal output (TA, Sentiment)
    decisions.csv   — portfolio manager combined decisions
    orders.csv      — executed or skipped orders
"""

import csv
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_JOURNAL_DIR = os.path.join(os.path.dirname(__file__), "journal")

# ── CSV schemas ──────────────────────────────────────────────────

SIGNAL_FIELDS = [
    "timestamp", "symbol", "agent", "signal", "score",
    "price", "detail",
]

DECISION_FIELDS = [
    "timestamp", "symbol", "signal", "combined_score", "confidence",
    "ta_score", "ta_weight", "sentiment_score", "sentiment_weight",
    "risk_approved", "position_size", "stop_loss", "take_profit",
]

ORDER_FIELDS = [
    "timestamp", "symbol", "side", "qty", "order_type",
    "status", "order_id", "reason",
]


class TradeJournal:
    """Append-only CSV journal for trading activity."""

    def __init__(self, journal_dir: str = DEFAULT_JOURNAL_DIR):
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)

        self._signals_path = self.journal_dir / "signals.csv"
        self._decisions_path = self.journal_dir / "decisions.csv"
        self._orders_path = self.journal_dir / "orders.csv"

        # Write headers if files don't exist yet
        self._ensure_headers(self._signals_path, SIGNAL_FIELDS)
        self._ensure_headers(self._decisions_path, DECISION_FIELDS)
        self._ensure_headers(self._orders_path, ORDER_FIELDS)

        logger.info("Trade journal initialised at %s", self.journal_dir)

    @staticmethod
    def _ensure_headers(path: Path, fields: list[str]) -> None:
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ── Logging methods ──────────────────────────────────────

    def log_signal(
        self,
        symbol: str,
        agent: str,
        signal: str,
        score: float,
        price: float = 0.0,
        detail: str = "",
    ) -> None:
        """Record an individual agent's signal."""
        row = {
            "timestamp": self._now(),
            "symbol": symbol,
            "agent": agent,
            "signal": signal,
            "score": round(score, 4),
            "price": round(price, 2),
            "detail": detail,
        }
        self._append(self._signals_path, SIGNAL_FIELDS, row)

    def log_decision(self, decision: dict) -> None:
        """Record a portfolio manager decision dict."""
        agent_signals = decision.get("agent_signals", [])
        ta_score = 0.0
        ta_weight = 0.0
        sent_score = 0.0
        sent_weight = 0.0

        for sig in agent_signals:
            s = sig if isinstance(sig, dict) else sig.__dict__
            if s.get("agent") == "TechnicalAnalysis":
                ta_score = s.get("normalised_score", 0)
                ta_weight = s.get("weight", 0)
            elif s.get("agent") == "SentimentAnalysis":
                sent_score = s.get("normalised_score", 0)
                sent_weight = s.get("weight", 0)

        row = {
            "timestamp": self._now(),
            "symbol": decision.get("symbol", ""),
            "signal": decision.get("signal", "HOLD"),
            "combined_score": decision.get("combined_score", 0),
            "confidence": decision.get("confidence", 0),
            "ta_score": round(ta_score, 4),
            "ta_weight": round(ta_weight, 2),
            "sentiment_score": round(sent_score, 4),
            "sentiment_weight": round(sent_weight, 2),
            "risk_approved": decision.get("risk_approved"),
            "position_size": decision.get("position_size", 0),
            "stop_loss": decision.get("stop_loss", 0),
            "take_profit": decision.get("take_profit", 0),
        }
        self._append(self._decisions_path, DECISION_FIELDS, row)

    def log_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_result: dict | str | None,
        reason: str = "",
    ) -> None:
        """Record an order execution (or skip)."""
        if isinstance(order_result, dict):
            order_id = order_result.get("id", "")
            status = order_result.get("status", "")
            order_type = order_result.get("type", "market")
        else:
            order_id = ""
            status = str(order_result) if order_result else "skipped"
            order_type = ""

        row = {
            "timestamp": self._now(),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_type": order_type,
            "status": status,
            "order_id": order_id,
            "reason": reason,
        }
        self._append(self._orders_path, ORDER_FIELDS, row)

    @staticmethod
    def _append(path: Path, fields: list[str], row: dict) -> None:
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

    # ── Reading ──────────────────────────────────────────────

    def read_signals(self, tail: int = 20) -> list[dict]:
        """Return the last N signal rows."""
        return self._read_tail(self._signals_path, tail)

    def read_decisions(self, tail: int = 20) -> list[dict]:
        """Return the last N decision rows."""
        return self._read_tail(self._decisions_path, tail)

    def read_orders(self, tail: int = 20) -> list[dict]:
        """Return the last N order rows."""
        return self._read_tail(self._orders_path, tail)

    @staticmethod
    def _read_tail(path: Path, n: int) -> list[dict]:
        if not path.exists():
            return []
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        return rows[-n:]
