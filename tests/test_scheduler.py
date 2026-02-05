"""Tests for scheduler.py (Unified Pipeline scheduler)"""

from unittest.mock import MagicMock, patch, call
from io import StringIO

import pytest

from scheduler import (
    create_pipeline,
    create_notifier,
    run_cycle,
    run_preflight,
    wait_for_market_open,
)


# ── Helpers ──────────────────────────────────────────────────────


def _mock_pipeline(results=None):
    """Create a mock TradingPipeline."""
    pipeline = MagicMock()
    if results is None:
        results = []
    pipeline.run.return_value = results
    return pipeline


class _FakeResult:
    """Lightweight PipelineResult stand-in."""

    def __init__(
        self, symbol="AAPL", signal="HOLD", executed=False,
        error="", combined_score=0.0, confidence=0.0,
        shares=0, order_status="", regime="", steps=None,
        correlation_filtered=False, risk_approved=False,
        stop_loss=0.0, take_profit=0.0, risk_per_share=0.0,
        kelly_fraction=0.0, order_id="", filled_qty=0,
        filled_avg_price=0.0, price=0.0, atr=0.0,
    ):
        self.symbol = symbol
        self.signal = signal
        self._executed = executed
        self.error = error
        self.combined_score = combined_score
        self.confidence = confidence
        self.shares = shares
        self.order_status = order_status
        self.regime = regime
        self.steps = steps or []
        self.correlation_filtered = correlation_filtered
        self.risk_approved = risk_approved
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.risk_per_share = risk_per_share
        self.kelly_fraction = kelly_fraction
        self.order_id = order_id
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.price = price
        self.atr = atr

    @property
    def executed(self):
        return self._executed


# ── create_pipeline tests ────────────────────────────────────────


class TestCreatePipeline:
    @patch("scheduler.TradingPipeline")
    @patch("scheduler.PositionSizer")
    @patch("scheduler.SignalAggregator")
    def test_creates_pipeline_with_settings(self, MockAgg, MockSizer, MockPipeline):
        client = MagicMock()
        pipeline = create_pipeline(client, dry_run=True)
        MockPipeline.assert_called_once()
        call_kwargs = MockPipeline.call_args[1]
        assert call_kwargs["client"] is client
        assert call_kwargs["dry_run"] is True

    @patch("scheduler.TradingPipeline")
    @patch("scheduler.PositionSizer")
    @patch("scheduler.SignalAggregator")
    def test_passes_event_bus(self, MockAgg, MockSizer, MockPipeline):
        client = MagicMock()
        bus = MagicMock()
        create_pipeline(client, event_bus=bus)
        call_kwargs = MockPipeline.call_args[1]
        assert call_kwargs["event_bus"] is bus

    @patch("scheduler.TradingPipeline")
    @patch("scheduler.PositionSizer")
    @patch("scheduler.SignalAggregator")
    def test_passes_notifier(self, MockAgg, MockSizer, MockPipeline):
        client = MagicMock()
        notifier = MagicMock()
        create_pipeline(client, notifier=notifier)
        call_kwargs = MockPipeline.call_args[1]
        assert call_kwargs["notifier"] is notifier


# ── create_notifier tests ────────────────────────────────────────


class TestCreateNotifier:
    @patch("scheduler.Notifier")
    def test_creates_notifier(self, MockNotifier):
        notifier = create_notifier()
        MockNotifier.assert_called_once()


# ── run_cycle tests ──────────────────────────────────────────────


class TestRunCycle:
    def test_returns_results(self):
        result = _FakeResult(symbol="AAPL", signal="BUY")
        pipeline = _mock_pipeline(results=[result])

        with patch("scheduler.TradingPipeline") as MockTP:
            MockTP.format_result = MagicMock(return_value="formatted result")
            MockTP.format_summary = MagicMock(return_value="summary")
            results = run_cycle(pipeline, ["AAPL"], "1Day", cycle_number=1)

        assert len(results) == 1
        pipeline.run.assert_called_once_with(["AAPL"], timeframe="1Day")

    def test_handles_empty_results(self):
        pipeline = _mock_pipeline(results=[])

        with patch("scheduler.TradingPipeline") as MockTP:
            MockTP.format_result = MagicMock(return_value="")
            MockTP.format_summary = MagicMock(return_value="summary")
            results = run_cycle(pipeline, ["AAPL"], "1Day")

        assert results == []

    def test_handles_pipeline_exception(self):
        pipeline = MagicMock()
        pipeline.run.side_effect = Exception("API timeout")

        results = run_cycle(pipeline, ["AAPL"], "1Day", cycle_number=1)
        assert results == []

    def test_counts_signals(self, capsys):
        results_data = [
            _FakeResult(symbol="AAPL", signal="BUY"),
            _FakeResult(symbol="MSFT", signal="SELL"),
            _FakeResult(symbol="GOOGL", signal="HOLD"),
        ]
        pipeline = _mock_pipeline(results=results_data)

        with patch("scheduler.TradingPipeline") as MockTP:
            MockTP.format_result = MagicMock(return_value="")
            MockTP.format_summary = MagicMock(return_value="")
            run_cycle(pipeline, ["AAPL", "MSFT", "GOOGL"], "1Day", cycle_number=1)

        output = capsys.readouterr().out
        assert "1 BUY" in output
        assert "1 SELL" in output

    def test_counts_executed(self, capsys):
        results_data = [
            _FakeResult(symbol="AAPL", signal="BUY", executed=True),
            _FakeResult(symbol="MSFT", signal="HOLD"),
        ]
        pipeline = _mock_pipeline(results=results_data)

        with patch("scheduler.TradingPipeline") as MockTP:
            MockTP.format_result = MagicMock(return_value="")
            MockTP.format_summary = MagicMock(return_value="")
            run_cycle(pipeline, ["AAPL", "MSFT"], "1Day", cycle_number=2)

        output = capsys.readouterr().out
        assert "1 executed" in output

    def test_counts_errors(self, capsys):
        results_data = [
            _FakeResult(symbol="AAPL", error="TA failed"),
        ]
        pipeline = _mock_pipeline(results=results_data)

        with patch("scheduler.TradingPipeline") as MockTP:
            MockTP.format_result = MagicMock(return_value="")
            MockTP.format_summary = MagicMock(return_value="")
            run_cycle(pipeline, ["AAPL"], "1Day", cycle_number=1)

        output = capsys.readouterr().out
        assert "1 errors" in output

    def test_cycle_number_in_output(self, capsys):
        pipeline = _mock_pipeline(results=[])

        with patch("scheduler.TradingPipeline") as MockTP:
            MockTP.format_result = MagicMock(return_value="")
            MockTP.format_summary = MagicMock(return_value="")
            run_cycle(pipeline, [], "1Day", cycle_number=42)

        output = capsys.readouterr().out
        assert "Cycle #42" in output


# ── run_preflight tests ──────────────────────────────────────────


class TestRunPreflight:
    def test_returns_true_on_all_pass(self, capsys):
        client = MagicMock()

        with patch("scheduler.PreflightCheck") as MockPF:
            mock_report = MagicMock()
            mock_report.passed = True
            mock_report.warnings = []
            mock_instance = MockPF.return_value
            mock_instance.run_all.return_value = mock_report

            with patch("scheduler.PreflightCheck.format_report", return_value="report"):
                result = run_preflight(client, min_buying_power=1000.0)

        assert result is True

    def test_returns_false_on_failure(self, capsys):
        client = MagicMock()

        with patch("scheduler.PreflightCheck") as MockPF:
            mock_report = MagicMock()
            mock_report.passed = False
            mock_report.warnings = []
            mock_report.failures = [MagicMock()]
            mock_instance = MockPF.return_value
            mock_instance.run_all.return_value = mock_report

            with patch("scheduler.PreflightCheck.format_report", return_value="report"):
                result = run_preflight(client)

        assert result is False

    def test_passes_with_warnings(self, capsys):
        client = MagicMock()

        with patch("scheduler.PreflightCheck") as MockPF:
            mock_report = MagicMock()
            mock_report.passed = True
            mock_report.warnings = [MagicMock()]
            mock_instance = MockPF.return_value
            mock_instance.run_all.return_value = mock_report

            with patch("scheduler.PreflightCheck.format_report", return_value="report"):
                result = run_preflight(client)

        assert result is True

    def test_passes_min_buying_power(self):
        client = MagicMock()

        with patch("scheduler.PreflightCheck") as MockPF:
            mock_report = MagicMock()
            mock_report.passed = True
            mock_report.warnings = []
            MockPF.return_value.run_all.return_value = mock_report

            with patch("scheduler.PreflightCheck.format_report", return_value=""):
                run_preflight(client, min_buying_power=5000.0)

        MockPF.assert_called_once_with(
            client,
            min_buying_power=5000.0,
            require_market_open=False,
        )


# ── wait_for_market_open tests ───────────────────────────────────


class TestWaitForMarketOpen:
    def test_returns_true_if_open(self):
        client = MagicMock()
        clock = MagicMock()
        clock.is_open = True
        client.api.get_clock.return_value = clock

        result = wait_for_market_open(client, no_wait=False)
        assert result is True

    def test_returns_false_if_closed_and_no_wait(self):
        client = MagicMock()
        clock = MagicMock()
        clock.is_open = False
        client.api.get_clock.return_value = clock

        result = wait_for_market_open(client, no_wait=True)
        assert result is False

    def test_waits_when_market_closed(self):
        """When market is closed and no-wait is False, it should wait then return."""
        import datetime
        client = MagicMock()
        clock = MagicMock()
        clock.is_open = False
        now = datetime.datetime.now(datetime.timezone.utc)
        clock.timestamp = now
        # Next open in 1 second so the wait loop exits quickly
        clock.next_open = now + datetime.timedelta(seconds=1)
        client.api.get_clock.return_value = clock

        import scheduler
        original_shutdown = scheduler._shutdown
        try:
            scheduler._shutdown = False
            with patch("scheduler.time.sleep"):
                result = wait_for_market_open(client, no_wait=False)
            assert result is True
        finally:
            scheduler._shutdown = original_shutdown


# ── main() argument parsing (smoke tests) ────────────────────────


class TestMainArgParsing:
    @patch("scheduler.sys.exit")
    @patch("scheduler.AlpacaClient")
    @patch("scheduler.run_preflight", return_value=False)
    def test_exits_on_preflight_failure(self, mock_pf, mock_client, mock_exit):
        """If preflight fails, main should call sys.exit(1)."""
        import scheduler
        with patch("sys.argv", ["scheduler.py", "--once"]):
            scheduler.main()
        mock_exit.assert_called_with(1)

    @patch("scheduler.run_cycle", return_value=[])
    @patch("scheduler.create_pipeline")
    @patch("scheduler.run_preflight", return_value=True)
    @patch("scheduler.AlpacaClient")
    def test_single_pass_mode(self, mock_client, mock_pf, mock_create, mock_cycle):
        """--once should run a single cycle and return."""
        import scheduler
        with patch("sys.argv", ["scheduler.py", "--once", "--skip-preflight"]):
            scheduler.main()
        mock_cycle.assert_called_once()

    @patch("scheduler.run_cycle", return_value=[])
    @patch("scheduler.create_pipeline")
    @patch("scheduler.AlpacaClient")
    def test_skip_preflight(self, mock_client, mock_create, mock_cycle, capsys):
        """--skip-preflight should skip health checks."""
        import scheduler
        with patch("sys.argv", ["scheduler.py", "--once", "--skip-preflight"]):
            scheduler.main()
        output = capsys.readouterr().out
        assert "SKIP" in output

    @patch("scheduler.run_cycle", return_value=[])
    @patch("scheduler.create_pipeline")
    @patch("scheduler.AlpacaClient")
    def test_auto_trade_mode(self, mock_client, mock_create, mock_cycle, capsys):
        """--auto-trade should show AUTO-TRADE mode."""
        import scheduler
        with patch("sys.argv", ["scheduler.py", "--once", "--auto-trade", "--skip-preflight"]):
            scheduler.main()
        output = capsys.readouterr().out
        assert "AUTO-TRADE" in output

    @patch("scheduler.run_cycle", return_value=[])
    @patch("scheduler.create_pipeline")
    @patch("scheduler.AlpacaClient")
    def test_dry_run_mode(self, mock_client, mock_create, mock_cycle, capsys):
        """Default mode should be DRY-RUN."""
        import scheduler
        with patch("sys.argv", ["scheduler.py", "--once", "--skip-preflight"]):
            scheduler.main()
        output = capsys.readouterr().out
        assert "DRY-RUN" in output

    @patch("scheduler.run_cycle", return_value=[])
    @patch("scheduler.create_pipeline")
    @patch("scheduler.AlpacaClient")
    def test_custom_symbols(self, mock_client, mock_create, mock_cycle, capsys):
        """Positional symbols should appear in output."""
        import scheduler
        with patch("sys.argv", ["scheduler.py", "AAPL", "TSLA", "--once", "--skip-preflight"]):
            scheduler.main()
        mock_cycle.assert_called_once()
        call_args = mock_cycle.call_args
        assert "AAPL" in call_args[0][1]
        assert "TSLA" in call_args[0][1]

    @patch("scheduler.run_cycle", return_value=[])
    @patch("scheduler.create_pipeline")
    @patch("scheduler.AlpacaClient")
    def test_custom_interval(self, mock_client, mock_create, mock_cycle, capsys):
        """--interval should set the interval."""
        import scheduler
        with patch("sys.argv", ["scheduler.py", "--once", "--skip-preflight", "--interval", "30"]):
            scheduler.main()
        output = capsys.readouterr().out
        assert "30 min" in output

    @patch("scheduler.EventBus")
    @patch("scheduler.run_cycle", return_value=[])
    @patch("scheduler.create_pipeline")
    @patch("scheduler.AlpacaClient")
    def test_show_events_flag(self, mock_client, mock_create, mock_cycle, mock_bus, capsys):
        """--show-events should not crash."""
        import scheduler
        with patch("sys.argv", ["scheduler.py", "--once", "--skip-preflight", "--show-events"]):
            # Mock bus instance
            bus_instance = MagicMock()
            bus_instance.get_history.return_value = []
            mock_bus.return_value = bus_instance
            scheduler.main()
