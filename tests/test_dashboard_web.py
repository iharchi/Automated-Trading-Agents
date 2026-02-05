"""Tests for the Live Web Dashboard.

All tests use temporary directories and mock objects -- no Flask or broker
connection required.
"""

import csv
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from dashboard_web import DashboardData, create_app


# ── DashboardData.__init__ ───────────────────────────────────────


class TestDashboardDataInit:
    def test_default_journal_dir(self):
        dd = DashboardData()
        # Default should point to <project>/utils/journal
        assert dd.journal_dir.name == "journal"
        assert "utils" in str(dd.journal_dir)

    def test_custom_journal_dir(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        assert dd.journal_dir == tmp_path

    def test_client_defaults_to_none(self):
        dd = DashboardData()
        assert dd.client is None

    def test_client_is_stored(self):
        mock_client = MagicMock()
        dd = DashboardData(client=mock_client)
        assert dd.client is mock_client

    def test_cache_starts_empty(self):
        dd = DashboardData()
        assert dd._cache == {}

    def test_cache_ttl_default(self):
        dd = DashboardData()
        assert dd._cache_ttl == 30


# ── DashboardData._cached ───────────────────────────────────────


class TestCached:
    def test_stores_and_returns_value(self):
        dd = DashboardData()
        result = dd._cached("test_key", lambda: 42)
        assert result == 42
        assert "test_key" in dd._cache

    def test_returns_cached_within_ttl(self):
        dd = DashboardData()
        call_count = {"n": 0}

        def expensive():
            call_count["n"] += 1
            return "data"

        dd._cached("k", expensive, ttl=60)
        dd._cached("k", expensive, ttl=60)
        assert call_count["n"] == 1  # called only once

    def test_refreshes_after_ttl_expires(self):
        dd = DashboardData()
        call_count = {"n": 0}

        def expensive():
            call_count["n"] += 1
            return call_count["n"]

        dd._cached("k", expensive, ttl=1)
        # Manually expire the cache entry
        dd._cache["k"]["ts"] = 0
        dd._cached("k", expensive, ttl=1)
        # After expiry, function should be called again
        assert call_count["n"] == 2

    def test_returns_stale_on_error(self):
        dd = DashboardData()
        dd._cached("k", lambda: "good")
        # Expire the cache entry so it tries to re-fetch
        dd._cache["k"]["ts"] = 0

        def raise_error():
            raise RuntimeError("boom")

        # fn raises, but stale data should be returned
        result = dd._cached("k", raise_error, ttl=1)
        assert result == "good"

    def test_returns_none_on_error_without_cache(self):
        dd = DashboardData()
        result = dd._cached("k", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert result is None

    def test_custom_ttl(self):
        dd = DashboardData()
        dd._cached("k", lambda: "v", ttl=120)
        entry = dd._cache["k"]
        assert entry["data"] == "v"


# ── get_status ───────────────────────────────────────────────────


class TestGetStatus:
    def test_without_client(self):
        dd = DashboardData()
        status = dd.get_status()
        assert "timestamp" in status
        assert "uptime_seconds" in status
        assert status["account"] is None
        assert status["market_open"] is None

    def test_with_client(self):
        mock_client = MagicMock()
        mock_client.get_account.return_value = {"equity": 100000}
        mock_client.is_market_open.return_value = True
        dd = DashboardData(client=mock_client)
        status = dd.get_status()
        assert status["account"] == {"equity": 100000}
        assert status["market_open"] is True

    def test_uptime_increases(self):
        dd = DashboardData()
        dd._start_time = time.time() - 120
        status = dd.get_status()
        assert status["uptime_seconds"] >= 119


# ── get_positions ────────────────────────────────────────────────


class TestGetPositions:
    def test_without_client_returns_empty(self):
        dd = DashboardData()
        assert dd.get_positions() == []

    def test_with_client(self):
        mock_client = MagicMock()
        mock_client.get_positions.return_value = [{"symbol": "AAPL", "qty": 10}]
        dd = DashboardData(client=mock_client)
        positions = dd.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"


# ── get_signals / get_orders / get_decisions ─────────────────────


class TestJournalReaders:
    def _write_csv(self, path: Path, filename: str, headers: list, rows: list):
        filepath = path / filename
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_get_signals_from_csv(self, tmp_path):
        self._write_csv(
            tmp_path, "signals.csv",
            ["timestamp", "symbol", "agent", "signal", "score"],
            [
                {"timestamp": "2025-01-01T00:00:00", "symbol": "AAPL",
                 "agent": "TA", "signal": "BUY", "score": "3"},
                {"timestamp": "2025-01-01T01:00:00", "symbol": "MSFT",
                 "agent": "TA", "signal": "SELL", "score": "-2"},
            ],
        )
        dd = DashboardData(journal_dir=str(tmp_path))
        signals = dd.get_signals()
        assert len(signals) == 2
        assert signals[0]["symbol"] == "AAPL"

    def test_get_signals_missing_file(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        assert dd.get_signals() == []

    def test_get_orders_from_csv(self, tmp_path):
        self._write_csv(
            tmp_path, "orders.csv",
            ["timestamp", "symbol", "side", "qty", "status"],
            [
                {"timestamp": "2025-01-01T00:00:00", "symbol": "AAPL",
                 "side": "buy", "qty": "10", "status": "filled"},
            ],
        )
        dd = DashboardData(journal_dir=str(tmp_path))
        orders = dd.get_orders()
        assert len(orders) == 1
        assert orders[0]["side"] == "buy"

    def test_get_orders_missing_file(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        assert dd.get_orders() == []

    def test_get_decisions_from_csv(self, tmp_path):
        self._write_csv(
            tmp_path, "decisions.csv",
            ["timestamp", "symbol", "signal", "action"],
            [
                {"timestamp": "2025-01-01T00:00:00", "symbol": "AAPL",
                 "signal": "BUY", "action": "execute"},
            ],
        )
        dd = DashboardData(journal_dir=str(tmp_path))
        decisions = dd.get_decisions()
        assert len(decisions) == 1
        assert decisions[0]["signal"] == "BUY"

    def test_get_decisions_missing_file(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        assert dd.get_decisions() == []

    def test_limit_parameter(self, tmp_path):
        rows = [
            {"timestamp": f"2025-01-01T{i:02d}:00:00", "symbol": "X",
             "agent": "TA", "signal": "HOLD", "score": "0"}
            for i in range(10)
        ]
        self._write_csv(
            tmp_path, "signals.csv",
            ["timestamp", "symbol", "agent", "signal", "score"],
            rows,
        )
        dd = DashboardData(journal_dir=str(tmp_path))
        limited = dd.get_signals(limit=3)
        assert len(limited) == 3


# ── get_performance ──────────────────────────────────────────────


class TestGetPerformance:
    def test_empty_performance(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        perf = dd.get_performance()
        assert perf["total_orders"] == 0
        assert perf["buys"] == 0
        assert perf["sells"] == 0
        assert perf["total_decisions"] == 0
        assert perf["signals_by_type"] == {}

    def test_performance_with_data(self, tmp_path):
        # Write orders
        orders_path = tmp_path / "orders.csv"
        with open(orders_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "symbol", "side", "qty", "status"])
            w.writeheader()
            w.writerow({"timestamp": "t1", "symbol": "A", "side": "buy", "qty": "5", "status": "filled"})
            w.writerow({"timestamp": "t2", "symbol": "A", "side": "sell", "qty": "5", "status": "filled"})
            w.writerow({"timestamp": "t3", "symbol": "B", "side": "buy", "qty": "3", "status": "rejected"})

        # Write decisions
        decisions_path = tmp_path / "decisions.csv"
        with open(decisions_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "symbol", "signal", "action"])
            w.writeheader()
            w.writerow({"timestamp": "t1", "symbol": "A", "signal": "BUY", "action": "execute"})
            w.writerow({"timestamp": "t2", "symbol": "A", "signal": "SELL", "action": "execute"})
            w.writerow({"timestamp": "t3", "symbol": "B", "signal": "HOLD", "action": "skip"})

        dd = DashboardData(journal_dir=str(tmp_path))
        perf = dd.get_performance()
        assert perf["total_orders"] == 2  # only filled
        assert perf["buys"] == 1
        assert perf["sells"] == 1
        assert perf["total_decisions"] == 3
        assert perf["signals_by_type"]["BUY"] == 1
        assert perf["signals_by_type"]["SELL"] == 1
        assert perf["signals_by_type"]["HOLD"] == 1


# ── render_html ──────────────────────────────────────────────────


class TestRenderHtml:
    def test_returns_html_string(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        html = dd.render_html()
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "Trading Dashboard" in html

    def test_html_contains_sections(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        html = dd.render_html()
        assert "Positions" in html
        assert "Recent Signals" in html
        assert "Recent Orders" in html
        assert "Signal Distribution" in html


# ── _positions_table ─────────────────────────────────────────────


class TestPositionsTable:
    def test_no_positions(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        html = dd._positions_table([])
        assert "No open positions" in html

    def test_with_positions(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        positions = [
            {"symbol": "AAPL", "qty": 10, "current_price": 150.0,
             "market_value": 1500.0, "unrealized_pl": 50.0},
        ]
        html = dd._positions_table(positions)
        assert "<table>" in html
        assert "AAPL" in html
        assert "positive" in html

    def test_negative_pnl(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        positions = [
            {"symbol": "TSLA", "qty": 5, "current_price": 200.0,
             "market_value": 1000.0, "unrealized_pl": -100.0},
        ]
        html = dd._positions_table(positions)
        assert "negative" in html


# ── _signals_table ───────────────────────────────────────────────


class TestSignalsTable:
    def test_no_signals(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        html = dd._signals_table([])
        assert "No signals yet" in html

    def test_with_signals(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        signals = [
            {"timestamp": "2025-01-01T10:00:00Z", "symbol": "AAPL",
             "agent": "TA", "signal": "BUY", "score": "3"},
        ]
        html = dd._signals_table(signals)
        assert "AAPL" in html
        assert "BUY" in html
        assert "<table>" in html


# ── _orders_table ────────────────────────────────────────────────


class TestOrdersTable:
    def test_no_orders(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        html = dd._orders_table([])
        assert "No orders yet" in html

    def test_with_orders(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        orders = [
            {"timestamp": "2025-01-01T10:00:00Z", "symbol": "MSFT",
             "side": "buy", "qty": "5", "status": "filled"},
        ]
        html = dd._orders_table(orders)
        assert "MSFT" in html
        assert "BUY" in html
        assert "<table>" in html


# ── _signal_dist ─────────────────────────────────────────────────


class TestSignalDist:
    def test_empty_distribution(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        html = dd._signal_dist({"signals_by_type": {}})
        assert "No decisions yet" in html

    def test_with_distribution(self, tmp_path):
        dd = DashboardData(journal_dir=str(tmp_path))
        perf = {"signals_by_type": {"BUY": 5, "SELL": 3, "HOLD": 10}}
        html = dd._signal_dist(perf)
        assert "BUY" in html
        assert "5" in html
        assert "SELL" in html
        assert "HOLD" in html


# ── create_app ───────────────────────────────────────────────────


class TestCreateApp:
    def test_returns_none_if_flask_not_installed(self):
        """Simulates Flask not being installed via ImportError."""
        with patch.dict("sys.modules", {"flask": None}):
            with patch("builtins.__import__", side_effect=_import_raise_for_flask):
                result = create_app()
                assert result is None

    def test_returns_app_if_flask_available(self):
        """Mock Flask to verify create_app returns an app object."""
        mock_flask_module = MagicMock()
        mock_app = MagicMock()
        mock_flask_module.Flask.return_value = mock_app
        mock_app.route = MagicMock(side_effect=lambda path: lambda fn: fn)

        with patch.dict("sys.modules", {"flask": mock_flask_module}):
            # Reload to pick up mocked module
            import importlib
            import dashboard_web
            importlib.reload(dashboard_web)
            app = dashboard_web.create_app()
            assert app is not None

        # Reload again to restore original state
        importlib.reload(dashboard_web)


# ── Helper for import mocking ────────────────────────────────────

_real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__


def _import_raise_for_flask(name, *args, **kwargs):
    if name == "flask":
        raise ImportError("No module named 'flask'")
    return _real_import(name, *args, **kwargs)
