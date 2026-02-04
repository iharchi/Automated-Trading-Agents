"""Performance Dashboard

Reads the trade journal CSV files and prints aggregate performance
statistics.  No API keys required — works entirely from local data.

Usage:
    python dashboard.py                    # full summary
    python dashboard.py --symbol AAPL      # filter to one symbol
    python dashboard.py --last 50          # only last 50 decisions
    python main.py --dashboard             # via main entry point

Sections:
    1. Overview          — total decisions, signal breakdown, date range
    2. Signal Accuracy   — per-agent hit rates (BUY/SELL vs HOLD)
    3. Per-Symbol Stats  — breakdown by ticker
    4. Order Summary     — executed orders, fill rate
    5. Score Distribution — combined score histogram (text-based)
    6. Recent Activity   — last N decisions
"""

import argparse
import csv
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_JOURNAL_DIR = os.path.join(
    os.path.dirname(__file__), "utils", "journal"
)


def _read_csv(path: Path) -> list[dict]:
    """Read an entire CSV into a list of dicts."""
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ── Dashboard sections ───────────────────────────────────────────


def section_overview(decisions: list[dict]) -> str:
    """High-level summary: count, date range, signal breakdown."""
    if not decisions:
        return "  No decisions recorded yet.\n"

    lines = []
    total = len(decisions)
    signals = Counter(d["signal"] for d in decisions)
    buy_count = signals.get("BUY", 0)
    sell_count = signals.get("SELL", 0)
    hold_count = signals.get("HOLD", 0)

    dates = [d["timestamp"][:10] for d in decisions if d.get("timestamp")]
    first = min(dates) if dates else "?"
    last = max(dates) if dates else "?"

    symbols = set(d["symbol"] for d in decisions)

    lines.append(f"  Total Decisions  : {total}")
    lines.append(f"  Date Range       : {first}  to  {last}")
    lines.append(f"  Symbols Tracked  : {', '.join(sorted(symbols))}")
    lines.append(f"  Signal Breakdown : BUY={buy_count}  SELL={sell_count}  HOLD={hold_count}")
    actionable = buy_count + sell_count
    action_pct = actionable / total * 100 if total else 0
    lines.append(f"  Actionable Rate  : {action_pct:.1f}% ({actionable}/{total})")

    return "\n".join(lines) + "\n"


def section_signal_accuracy(signals: list[dict]) -> str:
    """Per-agent signal distribution."""
    if not signals:
        return "  No signals recorded yet.\n"

    by_agent: dict[str, Counter] = defaultdict(Counter)
    for s in signals:
        by_agent[s["agent"]][s["signal"]] += 1

    lines = []
    lines.append(f"  {'Agent':22s} {'BUY':>6s} {'SELL':>6s} {'HOLD':>6s} {'Total':>6s} {'Action%':>8s}")
    lines.append(f"  {'─' * 58}")

    for agent in sorted(by_agent):
        c = by_agent[agent]
        total = sum(c.values())
        buy = c.get("BUY", 0)
        sell = c.get("SELL", 0)
        hold = c.get("HOLD", 0)
        action_pct = (buy + sell) / total * 100 if total else 0
        lines.append(
            f"  {agent:22s} {buy:>6d} {sell:>6d} {hold:>6d} {total:>6d} {action_pct:>7.1f}%"
        )

    return "\n".join(lines) + "\n"


def section_per_symbol(decisions: list[dict]) -> str:
    """Breakdown by symbol."""
    if not decisions:
        return "  No decisions recorded yet.\n"

    by_sym: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "buy": 0, "sell": 0, "hold": 0,
        "scores": [], "approved": 0, "rejected": 0,
    })

    for d in decisions:
        sym = d["symbol"]
        s = by_sym[sym]
        s["count"] += 1
        sig = d["signal"]
        if sig == "BUY":
            s["buy"] += 1
        elif sig == "SELL":
            s["sell"] += 1
        else:
            s["hold"] += 1

        s["scores"].append(_safe_float(d.get("combined_score", "0")))

        risk = d.get("risk_approved", "")
        if risk == "True":
            s["approved"] += 1
        elif risk == "False":
            s["rejected"] += 1

    lines = []
    lines.append(
        f"  {'Symbol':8s} {'Decisions':>9s} {'BUY':>5s} {'SELL':>5s} "
        f"{'HOLD':>5s} {'Avg Score':>10s} {'Approved':>9s} {'Rejected':>9s}"
    )
    lines.append(f"  {'─' * 70}")

    for sym in sorted(by_sym):
        s = by_sym[sym]
        avg = sum(s["scores"]) / len(s["scores"]) if s["scores"] else 0
        lines.append(
            f"  {sym:8s} {s['count']:>9d} {s['buy']:>5d} {s['sell']:>5d} "
            f"{s['hold']:>5d} {avg:>+10.4f} {s['approved']:>9d} {s['rejected']:>9d}"
        )

    return "\n".join(lines) + "\n"


def section_orders(orders: list[dict]) -> str:
    """Order execution summary."""
    if not orders:
        return "  No orders recorded yet.\n"

    total = len(orders)
    by_status = Counter(o.get("status", "unknown") for o in orders)
    by_side = Counter(o.get("side", "unknown") for o in orders)

    lines = []
    lines.append(f"  Total Orders     : {total}")
    lines.append(f"  By Side          : " + ", ".join(
        f"{k}={v}" for k, v in sorted(by_side.items())
    ))
    lines.append(f"  By Status        : " + ", ".join(
        f"{k}={v}" for k, v in sorted(by_status.items())
    ))

    total_qty = sum(int(o.get("qty", 0)) for o in orders)
    lines.append(f"  Total Shares     : {total_qty}")

    return "\n".join(lines) + "\n"


def section_score_distribution(decisions: list[dict]) -> str:
    """Text-based histogram of combined scores."""
    if not decisions:
        return "  No data.\n"

    scores = [_safe_float(d.get("combined_score", "0")) for d in decisions]
    if not scores:
        return "  No scores.\n"

    # Bucket into 10 bins from -1 to +1
    bins = [0] * 10
    for s in scores:
        idx = int((s + 1) / 2 * 9.999)
        idx = max(0, min(9, idx))
        bins[idx] += 1

    max_count = max(bins) if bins else 1
    bar_width = 30

    lines = []
    labels = [f"{-1 + i * 0.2:+.1f}" for i in range(10)]
    for i, (label, count) in enumerate(zip(labels, bins)):
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        bar = "█" * bar_len
        lines.append(f"  {label}  {bar}  {count}")

    avg = sum(scores) / len(scores)
    lines.append(f"")
    lines.append(f"  Mean: {avg:+.4f}  |  Min: {min(scores):+.4f}  |  Max: {max(scores):+.4f}")

    return "\n".join(lines) + "\n"


def section_recent(decisions: list[dict], n: int = 10) -> str:
    """Last N decisions in table format."""
    recent = decisions[-n:]
    if not recent:
        return "  No recent decisions.\n"

    lines = []
    lines.append(
        f"  {'Timestamp':20s} {'Symbol':8s} {'Signal':6s} "
        f"{'Score':>8s} {'Conf':>6s} {'Risk':>8s} {'Size':>6s}"
    )
    lines.append(f"  {'─' * 66}")

    for d in recent:
        ts = d.get("timestamp", "")[:19]
        risk = d.get("risk_approved", "")
        if risk == "True":
            risk_str = "OK"
        elif risk == "False":
            risk_str = "REJECT"
        else:
            risk_str = "—"

        lines.append(
            f"  {ts:20s} {d['symbol']:8s} {d['signal']:6s} "
            f"{_safe_float(d.get('combined_score', '0')):>+8.4f} "
            f"{_safe_float(d.get('confidence', '0')):>5.0%} "
            f"{risk_str:>8s} "
            f"{d.get('position_size', '0'):>6s}"
        )

    return "\n".join(lines) + "\n"


# ── Main dashboard ───────────────────────────────────────────────


def render_dashboard(
    journal_dir: str = DEFAULT_JOURNAL_DIR,
    symbol: str | None = None,
    last: int = 10,
) -> str:
    """Build the full dashboard string."""
    jdir = Path(journal_dir)

    signals = _read_csv(jdir / "signals.csv")
    decisions = _read_csv(jdir / "decisions.csv")
    orders = _read_csv(jdir / "orders.csv")

    # Filter by symbol if requested
    if symbol:
        signals = [s for s in signals if s.get("symbol") == symbol]
        decisions = [d for d in decisions if d.get("symbol") == symbol]
        orders = [o for o in orders if o.get("symbol") == symbol]

    filter_label = f" ({symbol})" if symbol else ""

    sections = [
        f"\n{'#' * 62}",
        f"  PERFORMANCE DASHBOARD{filter_label}",
        f"{'#' * 62}",
        "",
        f"{'─' * 62}",
        f"  1. OVERVIEW",
        f"{'─' * 62}",
        section_overview(decisions),
        f"{'─' * 62}",
        f"  2. AGENT SIGNALS",
        f"{'─' * 62}",
        section_signal_accuracy(signals),
        f"{'─' * 62}",
        f"  3. PER-SYMBOL STATS",
        f"{'─' * 62}",
        section_per_symbol(decisions),
        f"{'─' * 62}",
        f"  4. ORDERS",
        f"{'─' * 62}",
        section_orders(orders),
        f"{'─' * 62}",
        f"  5. SCORE DISTRIBUTION",
        f"{'─' * 62}",
        section_score_distribution(decisions),
        f"{'─' * 62}",
        f"  6. RECENT ACTIVITY (last {last})",
        f"{'─' * 62}",
        section_recent(decisions, n=last),
        f"{'#' * 62}\n",
    ]

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description="Performance Dashboard")
    parser.add_argument(
        "--journal-dir",
        default=DEFAULT_JOURNAL_DIR,
        help="Path to journal directory",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Filter to a specific symbol",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=10,
        help="Number of recent decisions to show (default: 10)",
    )
    args = parser.parse_args()

    print(render_dashboard(
        journal_dir=args.journal_dir,
        symbol=args.symbol,
        last=args.last,
    ))


if __name__ == "__main__":
    main()
