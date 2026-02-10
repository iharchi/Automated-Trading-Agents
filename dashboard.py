"""Performance Dashboard

Reads the trade journal CSV files and prints aggregate performance
statistics. Also connects to Alpaca API to show real-time P&L.

Usage:
    python dashboard.py                    # full summary
    python dashboard.py --symbol AAPL      # filter to one symbol
    python dashboard.py --last 50          # only last 50 decisions
    python dashboard.py --live             # show live account P&L
    python main.py --dashboard             # via main entry point

Sections:
    1. Overview          — total decisions, signal breakdown, date range
    2. P&L Summary       — daily, cumulative, and per-position P&L
    3. Signal Accuracy   — per-agent hit rates (BUY/SELL vs HOLD)
    4. Per-Symbol Stats  — breakdown by ticker
    5. Order Summary     — executed orders, fill rate
    6. Score Distribution — combined score histogram (text-based)
    7. Recent Activity   — last N decisions
"""

import argparse
import csv
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
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


# ── P&L from Alpaca ──────────────────────────────────────────────


def get_pnl_data() -> dict:
    """Fetch P&L data from Alpaca API."""
    try:
        from utils.alpaca_client import AlpacaClient
        client = AlpacaClient()

        # Get account info
        account = client.get_account()
        equity = account.get("equity", 0)
        last_equity = account.get("last_equity", equity)
        cash = account.get("cash", 0)
        buying_power = account.get("buying_power", 0)

        # Calculate daily P&L
        daily_pnl = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity else 0

        # Get positions for unrealized P&L
        positions = client.get_positions()
        unrealized_pnl = sum(float(p.get("unrealized_pl", 0)) for p in positions)
        unrealized_pnl_pct = sum(float(p.get("unrealized_plpc", 0)) * 100 for p in positions) / len(positions) if positions else 0

        # Get portfolio history for cumulative P&L
        try:
            history = client.api.get_portfolio_history(
                period="1M",
                timeframe="1D"
            )

            if history and hasattr(history, 'profit_loss'):
                cumulative_pnl = sum(history.profit_loss) if history.profit_loss else 0
                # Get starting equity from a month ago
                if history.equity and len(history.equity) > 0:
                    start_equity = history.equity[0]
                    cumulative_pnl_pct = (cumulative_pnl / start_equity * 100) if start_equity else 0
                else:
                    cumulative_pnl_pct = 0

                # Daily P&L history
                daily_pnls = list(zip(history.timestamp, history.profit_loss)) if history.profit_loss else []
            else:
                cumulative_pnl = 0
                cumulative_pnl_pct = 0
                daily_pnls = []
        except Exception as e:
            logger.warning("Could not fetch portfolio history: %s", e)
            cumulative_pnl = 0
            cumulative_pnl_pct = 0
            daily_pnls = []

        return {
            "equity": equity,
            "last_equity": last_equity,
            "cash": cash,
            "buying_power": buying_power,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "cumulative_pnl": cumulative_pnl,
            "cumulative_pnl_pct": cumulative_pnl_pct,
            "positions": positions,
            "daily_pnls": daily_pnls[-10:],  # Last 10 days
            "connected": True,
        }
    except Exception as e:
        logger.warning("Could not connect to Alpaca: %s", e)
        return {"connected": False, "error": str(e)}


def section_pnl(pnl_data: dict) -> str:
    """P&L summary section."""
    if not pnl_data.get("connected"):
        return f"  Could not connect to Alpaca: {pnl_data.get('error', 'Unknown error')}\n"

    lines = []

    # Account overview
    lines.append(f"  {'─' * 40}")
    lines.append(f"  ACCOUNT")
    lines.append(f"  {'─' * 40}")
    lines.append(f"  Equity         : ${pnl_data['equity']:,.2f}")
    lines.append(f"  Cash           : ${pnl_data['cash']:,.2f}")
    lines.append(f"  Buying Power   : ${pnl_data['buying_power']:,.2f}")
    lines.append("")

    # P&L summary
    lines.append(f"  {'─' * 40}")
    lines.append(f"  PROFIT & LOSS")
    lines.append(f"  {'─' * 40}")

    daily = pnl_data['daily_pnl']
    daily_pct = pnl_data['daily_pnl_pct']
    daily_color = "+" if daily >= 0 else ""
    lines.append(f"  Today's P&L    : {daily_color}${daily:,.2f} ({daily_color}{daily_pct:.2f}%)")

    unrealized = pnl_data['unrealized_pnl']
    unrealized_color = "+" if unrealized >= 0 else ""
    lines.append(f"  Unrealized P&L : {unrealized_color}${unrealized:,.2f}")

    cumulative = pnl_data['cumulative_pnl']
    cumulative_pct = pnl_data['cumulative_pnl_pct']
    cumulative_color = "+" if cumulative >= 0 else ""
    lines.append(f"  Cumulative (1M): {cumulative_color}${cumulative:,.2f} ({cumulative_color}{cumulative_pct:.2f}%)")
    lines.append("")

    # Position P&L
    positions = pnl_data.get('positions', [])
    if positions:
        lines.append(f"  {'─' * 40}")
        lines.append(f"  POSITIONS ({len(positions)})")
        lines.append(f"  {'─' * 40}")
        lines.append(f"  {'Symbol':8s} {'Qty':>7s} {'Entry':>10s} {'Current':>10s} {'P&L':>12s} {'%':>8s}")
        lines.append(f"  {'-' * 57}")

        for p in positions:
            symbol = p.get('symbol', '?')
            qty = int(float(p.get('qty', 0)))
            avg_entry = float(p.get('avg_entry_price', 0))
            current = float(p.get('current_price', 0))
            pnl = float(p.get('unrealized_pl', 0))
            pnl_pct = float(p.get('unrealized_plpc', 0)) * 100
            pnl_sign = "+" if pnl >= 0 else ""

            lines.append(
                f"  {symbol:8s} {qty:>7d} ${avg_entry:>9.2f} ${current:>9.2f} "
                f"{pnl_sign}${pnl:>10.2f} {pnl_sign}{pnl_pct:>6.2f}%"
            )

        total_pnl = sum(float(p.get('unrealized_pl', 0)) for p in positions)
        total_sign = "+" if total_pnl >= 0 else ""
        lines.append(f"  {'-' * 57}")
        lines.append(f"  {'TOTAL':8s} {'':<7s} {'':<10s} {'':<10s} {total_sign}${total_pnl:>10.2f}")
        lines.append("")

    # Daily P&L history
    daily_pnls = pnl_data.get('daily_pnls', [])
    if daily_pnls:
        lines.append(f"  {'─' * 40}")
        lines.append(f"  DAILY P&L HISTORY (Last {len(daily_pnls)} days)")
        lines.append(f"  {'─' * 40}")

        running_total = 0
        for ts, pnl in daily_pnls:
            date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            running_total += pnl
            pnl_sign = "+" if pnl >= 0 else ""
            cum_sign = "+" if running_total >= 0 else ""

            # Simple bar chart
            bar_len = min(abs(int(pnl / 10)), 20)
            if pnl >= 0:
                bar = "█" * bar_len
            else:
                bar = "░" * bar_len

            lines.append(f"  {date}  {pnl_sign}${pnl:>8.2f}  {bar:20s}  Cum: {cum_sign}${running_total:>10.2f}")

    return "\n".join(lines) + "\n"


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
    show_live: bool = True,
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
    ]

    # Add P&L section if live mode is enabled
    if show_live:
        pnl_data = get_pnl_data()
        sections.extend([
            f"{'─' * 62}",
            f"  2. P&L SUMMARY (LIVE)",
            f"{'─' * 62}",
            section_pnl(pnl_data),
        ])
        next_section = 3
    else:
        next_section = 2

    sections.extend([
        f"{'─' * 62}",
        f"  {next_section}. AGENT SIGNALS",
        f"{'─' * 62}",
        section_signal_accuracy(signals),
        f"{'─' * 62}",
        f"  {next_section + 1}. PER-SYMBOL STATS",
        f"{'─' * 62}",
        section_per_symbol(decisions),
        f"{'─' * 62}",
        f"  {next_section + 2}. ORDERS",
        f"{'─' * 62}",
        section_orders(orders),
        f"{'─' * 62}",
        f"  {next_section + 3}. SCORE DISTRIBUTION",
        f"{'─' * 62}",
        section_score_distribution(decisions),
        f"{'─' * 62}",
        f"  {next_section + 4}. RECENT ACTIVITY (last {last})",
        f"{'─' * 62}",
        section_recent(decisions, n=last),
        f"{'#' * 62}\n",
    ])

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
    parser.add_argument(
        "--live",
        action="store_true",
        default=True,
        help="Show live P&L from Alpaca (default: True)",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Skip live P&L section (use if no API keys)",
    )
    args = parser.parse_args()

    show_live = args.live and not args.no_live

    print(render_dashboard(
        journal_dir=args.journal_dir,
        symbol=args.symbol,
        last=args.last,
        show_live=show_live,
    ))


if __name__ == "__main__":
    main()
