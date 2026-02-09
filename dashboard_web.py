"""Live Web Dashboard

Lightweight Flask-based web dashboard for monitoring the trading system
in real time.  Shows positions, signals, regime status, event stream,
and performance metrics.

Usage:
    python dashboard_web.py                     # Start on port 5000
    python dashboard_web.py --port 8080         # Custom port
    python dashboard_web.py --host 0.0.0.0      # Bind to all interfaces

Endpoints:
    GET /                — Main dashboard page (HTML)
    GET /api/status      — System status JSON
    GET /api/positions   — Current positions JSON
    GET /api/signals     — Recent signals JSON
    GET /api/orders      — Recent orders JSON
    GET /api/performance — Performance metrics JSON
    GET /api/events      — Event bus history JSON
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Dashboard data layer (no Flask dependency) ───────────────────


class DashboardData:
    """Collects and serves data for the dashboard without Flask dependency."""

    def __init__(
        self,
        *,
        journal_dir: str | None = None,
        client=None,
    ):
        self.journal_dir = Path(
            journal_dir or os.path.join(os.path.dirname(__file__), "utils", "journal")
        )
        self.client = client  # AlpacaClient (optional)
        self._cache: dict = {}
        self._cache_ttl = 30  # seconds
        self._start_time = time.time()

    def _cached(self, key: str, fn, ttl: int | None = None):
        """Simple TTL cache."""
        ttl = ttl or self._cache_ttl
        entry = self._cache.get(key)
        now = time.time()
        if entry and (now - entry["ts"]) < ttl:
            return entry["data"]
        try:
            data = fn()
        except Exception as e:
            logger.warning("Dashboard data fetch failed (%s): %s", key, e)
            data = entry["data"] if entry else None
        self._cache[key] = {"data": data, "ts": now}
        return data

    # ── Data fetchers ────────────────────────────────────────────

    def get_status(self) -> dict:
        """System status summary."""
        uptime = time.time() - self._start_time
        account = None
        if self.client:
            account = self._cached("account", self.client.get_account, ttl=60)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": round(uptime),
            "account": account,
            "market_open": self._cached(
                "market_open",
                lambda: self.client.is_market_open() if self.client else None,
                ttl=60,
            ),
        }

    def get_positions(self) -> list[dict]:
        if not self.client:
            return []
        return self._cached("positions", self.client.get_positions, ttl=15) or []

    def get_signals(self, limit: int = 50) -> list[dict]:
        return self._read_journal_csv("signals.csv", limit)

    def get_orders(self, limit: int = 50) -> list[dict]:
        return self._read_journal_csv("orders.csv", limit)

    def get_decisions(self, limit: int = 50) -> list[dict]:
        return self._read_journal_csv("decisions.csv", limit)

    def get_performance(self) -> dict:
        """Compute basic performance from orders."""
        orders = self.get_orders(limit=500)
        decisions = self.get_decisions(limit=500)

        filled = [o for o in orders if o.get("status") in ("filled", "dry_run")]
        total = len(filled)
        buys = sum(1 for o in filled if o.get("side") == "buy")
        sells = sum(1 for o in filled if o.get("side") == "sell")

        signals_by_type = {}
        for d in decisions:
            sig = d.get("signal", "HOLD")
            signals_by_type[sig] = signals_by_type.get(sig, 0) + 1

        return {
            "total_orders": total,
            "buys": buys,
            "sells": sells,
            "total_decisions": len(decisions),
            "signals_by_type": signals_by_type,
        }

    def _read_journal_csv(self, filename: str, limit: int) -> list[dict]:
        import csv
        path = self.journal_dir / filename
        if not path.exists():
            return []
        try:
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            return rows[-limit:]
        except Exception:
            return []

    # ── HTML rendering ───────────────────────────────────────────

    def render_html(self) -> str:
        """Render the full dashboard as standalone HTML (no templates)."""
        status = self.get_status()
        positions = self.get_positions()
        signals = self.get_signals(20)
        orders = self.get_orders(20)
        perf = self.get_performance()

        account = status.get("account") or {}
        market = "OPEN" if status.get("market_open") else "CLOSED"

        # Calculate overall P&L
        total_unrealized_pnl = sum(p.get("unrealized_pl", 0) for p in positions)
        total_market_value = sum(abs(p.get("market_value", 0)) for p in positions)
        pnl_class = "positive" if total_unrealized_pnl >= 0 else "negative"

        # Calculate daily P&L (approximate from account)
        equity = account.get('equity', 0)
        last_equity = account.get('last_equity', equity)
        daily_pnl = equity - last_equity if last_equity else 0
        daily_pnl_class = "positive" if daily_pnl >= 0 else "negative"

        # Build HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading Dashboard</title>
<meta http-equiv="refresh" content="30">
<style>
  body {{ font-family: 'Courier New', monospace; background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }}
  h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
  h2 {{ color: #00d4ff; margin-top: 30px; }}
  .card {{ background: #16213e; border-radius: 8px; padding: 15px; margin: 10px 0; border: 1px solid #0f3460; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
  .metric {{ text-align: center; }}
  .metric .value {{ font-size: 24px; font-weight: bold; color: #00d4ff; }}
  .metric .label {{ font-size: 12px; color: #888; margin-top: 5px; }}
  .pnl-card {{ background: #16213e; border-radius: 8px; padding: 20px; margin: 15px 0; border: 2px solid #0f3460; }}
  .pnl-card .value {{ font-size: 36px; font-weight: bold; }}
  .pnl-card .label {{ font-size: 14px; color: #888; margin-top: 5px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
  th {{ text-align: left; padding: 8px; border-bottom: 2px solid #0f3460; color: #00d4ff; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #0f3460; }}
  .buy {{ color: #00ff88; }}
  .sell {{ color: #ff4444; }}
  .hold {{ color: #888; }}
  .positive {{ color: #00ff88; }}
  .negative {{ color: #ff4444; }}
  .status-open {{ color: #00ff88; font-weight: bold; }}
  .status-closed {{ color: #ff4444; font-weight: bold; }}
  .timestamp {{ color: #666; font-size: 11px; }}
</style>
</head>
<body>
<h1>Automated Trading Dashboard</h1>
<p class="timestamp">Last updated: {status['timestamp']} | Market: <span class="status-{'open' if status.get('market_open') else 'closed'}">{market}</span></p>

<div class="grid">
  <div class="pnl-card metric">
    <div class="value {pnl_class}">${total_unrealized_pnl:+,.2f}</div>
    <div class="label">Unrealized P&L (Open Positions)</div>
  </div>
  <div class="pnl-card metric">
    <div class="value {daily_pnl_class}">${daily_pnl:+,.2f}</div>
    <div class="label">Daily P&L</div>
  </div>
</div>

<div class="grid">
  <div class="card metric">
    <div class="value">${equity:,.0f}</div>
    <div class="label">Equity</div>
  </div>
  <div class="card metric">
    <div class="value">${account.get('buying_power', 0):,.0f}</div>
    <div class="label">Buying Power</div>
  </div>
  <div class="card metric">
    <div class="value">${account.get('cash', 0):,.0f}</div>
    <div class="label">Cash</div>
  </div>
  <div class="card metric">
    <div class="value">${total_market_value:,.0f}</div>
    <div class="label">Position Value</div>
  </div>
  <div class="card metric">
    <div class="value">{len(positions)}</div>
    <div class="label">Open Positions</div>
  </div>
  <div class="card metric">
    <div class="value">{perf.get('total_orders', 0)}</div>
    <div class="label">Total Orders</div>
  </div>
</div>

<h2>Positions</h2>
<div class="card">
{self._positions_table(positions)}
</div>

<h2>Recent Signals</h2>
<div class="card">
{self._signals_table(signals)}
</div>

<h2>Recent Orders</h2>
<div class="card">
{self._orders_table(orders)}
</div>

<h2>Signal Distribution</h2>
<div class="card">
<div class="grid">
  {self._signal_dist(perf)}
</div>
</div>

</body>
</html>"""
        return html

    def _positions_table(self, positions: list[dict]) -> str:
        if not positions:
            return "<p>No open positions.</p>"
        rows = ""
        for p in positions:
            pnl = p.get("unrealized_pl", 0)
            cls = "positive" if pnl >= 0 else "negative"
            rows += (
                f"<tr><td>{p['symbol']}</td><td>{p.get('qty', 0)}</td>"
                f"<td>${p.get('current_price', 0):,.2f}</td>"
                f"<td>${p.get('market_value', 0):,.2f}</td>"
                f"<td class='{cls}'>${pnl:+,.2f}</td></tr>\n"
            )
        return (
            "<table><tr><th>Symbol</th><th>Qty</th><th>Price</th>"
            f"<th>Value</th><th>P&L</th></tr>{rows}</table>"
        )

    def _signals_table(self, signals: list[dict]) -> str:
        if not signals:
            return "<p>No signals yet.</p>"
        rows = ""
        for s in reversed(signals[-15:]):
            sig = s.get("signal", "HOLD")
            cls = sig.lower()
            rows += (
                f"<tr><td class='timestamp'>{s.get('timestamp', '')[:19]}</td>"
                f"<td>{s.get('symbol', '')}</td>"
                f"<td>{s.get('agent', '')}</td>"
                f"<td class='{cls}'>{sig}</td>"
                f"<td>{s.get('score', '')}</td></tr>\n"
            )
        return (
            "<table><tr><th>Time</th><th>Symbol</th><th>Agent</th>"
            f"<th>Signal</th><th>Score</th></tr>{rows}</table>"
        )

    def _orders_table(self, orders: list[dict]) -> str:
        if not orders:
            return "<p>No orders yet.</p>"
        rows = ""
        for o in reversed(orders[-15:]):
            side = o.get("side", "")
            cls = "buy" if side == "buy" else "sell"
            rows += (
                f"<tr><td class='timestamp'>{o.get('timestamp', '')[:19]}</td>"
                f"<td>{o.get('symbol', '')}</td>"
                f"<td class='{cls}'>{side.upper()}</td>"
                f"<td>{o.get('qty', '')}</td>"
                f"<td>{o.get('status', '')}</td></tr>\n"
            )
        return (
            "<table><tr><th>Time</th><th>Symbol</th><th>Side</th>"
            f"<th>Qty</th><th>Status</th></tr>{rows}</table>"
        )

    def _signal_dist(self, perf: dict) -> str:
        dist = perf.get("signals_by_type", {})
        cards = ""
        for sig, count in sorted(dist.items()):
            cls = sig.lower()
            cards += (
                f'<div class="card metric">'
                f'<div class="value {cls}">{count}</div>'
                f'<div class="label">{sig}</div></div>\n'
            )
        return cards or "<p>No decisions yet.</p>"


# ── Flask app factory ────────────────────────────────────────────


def create_app(dashboard_data: DashboardData | None = None):
    """Create a Flask app for the web dashboard.

    Returns None if Flask is not installed.
    """
    try:
        from flask import Flask, jsonify, send_file
    except ImportError:
        logger.warning("Flask not installed. Run: pip install flask")
        return None

    app = Flask(__name__)
    data = dashboard_data or DashboardData()

    # Path to agents info page
    agents_info_path = Path(__file__).parent / "agents_info.html"

    @app.route("/")
    def index():
        return data.render_html()

    @app.route("/api/status")
    def api_status():
        return jsonify(data.get_status())

    @app.route("/api/positions")
    def api_positions():
        return jsonify(data.get_positions())

    @app.route("/api/signals")
    def api_signals():
        return jsonify(data.get_signals())

    @app.route("/api/orders")
    def api_orders():
        return jsonify(data.get_orders())

    @app.route("/api/performance")
    def api_performance():
        return jsonify(data.get_performance())

    @app.route("/api/decisions")
    def api_decisions():
        return jsonify(data.get_decisions())

    @app.route("/agents")
    def agents_page():
        if agents_info_path.exists():
            return send_file(agents_info_path)
        return "Agents info page not found", 404

    @app.route("/api/scan")
    def api_scan():
        """Run a quick market scan and return results."""
        from flask import request
        try:
            from scanner import MarketScanner, UNIVERSES
            universe = request.args.get("universe", "DEFAULT")
            top_n = int(request.args.get("top", 10))

            symbols = UNIVERSES.get(universe, UNIVERSES["DEFAULT"])
            scanner = MarketScanner(client=data.client, max_workers=3)
            summary = scanner.scan(symbols, top_n=top_n)

            return jsonify({
                "timestamp": summary.timestamp,
                "market_regime": summary.market_regime,
                "total_scanned": summary.total_scanned,
                "buy_signals": summary.buy_signals,
                "sell_signals": summary.sell_signals,
                "scan_duration_sec": summary.scan_duration_sec,
                "top_buys": [
                    {
                        "symbol": r.symbol,
                        "score": r.combined_score,
                        "ta_score": r.ta_score,
                        "sentiment": r.sentiment_score,
                        "rsi": r.rsi,
                        "price": r.price,
                    }
                    for r in summary.top_buys
                ],
                "top_sells": [
                    {
                        "symbol": r.symbol,
                        "score": r.combined_score,
                        "ta_score": r.ta_score,
                        "sentiment": r.sentiment_score,
                        "rsi": r.rsi,
                        "price": r.price,
                    }
                    for r in summary.top_sells
                ],
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Live Web Dashboard")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--journal-dir", default=None)
    parser.add_argument("--no-broker", action="store_true",
                       help="Run without broker connection (journal data only)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    client = None
    if not args.no_broker:
        try:
            from utils.alpaca_client import AlpacaClient
            client = AlpacaClient()
            logger.info("Connected to Alpaca API")
        except Exception as e:
            logger.warning("Could not connect to broker: %s", e)

    data = DashboardData(
        journal_dir=args.journal_dir,
        client=client,
    )

    app = create_app(data)
    if app is None:
        # Flask not installed — serve a static HTML file
        html = data.render_html()
        output = Path("dashboard_output.html")
        output.write_text(html)
        print(f"Flask not installed. Static dashboard saved to {output}")
        return

    print(f"\nDashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
