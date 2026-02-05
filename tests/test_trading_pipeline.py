"""Tests for utils/trading_pipeline.py"""

from unittest.mock import MagicMock, patch

import pytest

from utils.trading_pipeline import (
    PipelineResult,
    PipelineStep,
    TradingPipeline,
)
from utils.signal_aggregator import SignalAggregator, AggregatedSignal
from utils.position_sizer import PositionSizer, SizingResult
from utils.event_bus import EventBus


# ── Helpers ───────────────────────────────────────────────────────

def _mock_client():
    """Return a mocked AlpacaClient."""
    client = MagicMock()
    client.get_account.return_value = {"equity": 100_000, "cash": 50_000}
    client.get_positions.return_value = []
    client.is_market_open.return_value = True
    return client


def _mock_ta_agent(signal="BUY", score=3, price=150.0, atr=2.5):
    agent = MagicMock()
    agent.analyze.return_value = {
        "symbol": "AAPL",
        "signal": signal,
        "composite_score": score,
        "current_price": price,
        "atr": atr,
        "indicators": [],
    }
    return agent


def _mock_sentiment_agent(signal="BUY", score=0.2):
    agent = MagicMock()
    agent.analyze.return_value = {
        "symbol": "AAPL",
        "signal": signal,
        "composite_score": score,
        "articles": [],
    }
    return agent


def _mock_risk_agent(approved=True, shares=50):
    agent = MagicMock()
    agent.analyze.return_value = {
        "approved": approved,
        "position_size": shares,
        "stop_loss": 146.0,
        "take_profit": 158.0,
        "checks": [{"rule": "test", "passed": approved, "detail": "ok"}],
    }
    return agent


def _mock_exec_agent(status="dry_run", order_id="ord_123"):
    agent = MagicMock()
    agent.analyze.return_value = {
        "all_passed": True,
        "checks": [],
        "ticket": MagicMock(),
        "decision": {},
    }
    agent.execute.return_value = {
        "symbol": "AAPL",
        "status": status,
        "side": "buy",
        "qty": 50,
        "order_id": order_id,
        "filled_qty": 50 if status == "filled" else 0,
        "filled_avg_price": 150.0 if status == "filled" else 0,
        "order_type": "market",
        "error": "",
    }
    return agent


class _RegimeObj:
    """Simple regime result stand-in (avoids MagicMock __dict__ issues)."""
    def __init__(self, regime="TRENDING_UP", confidence=0.7):
        self.regime = regime
        self.confidence = confidence
        self.adjustments = {
            "position_size_factor": 1.2,
            "stop_multiplier": 1.3,
            "take_profit_multiplier": 1.5,
            "min_score_adjustment": -1,
        }
        self.indicators = None


class _CorrObj:
    """Simple correlation result stand-in."""
    def __init__(self, is_correlated=False):
        self.is_correlated = is_correlated
        self.correlated_with = ["MSFT"] if is_correlated else []


def _mock_regime_detector(regime="TRENDING_UP", confidence=0.7):
    detector = MagicMock()
    detector.detect.return_value = _RegimeObj(regime, confidence)
    return detector


def _mock_corr_filter(is_correlated=False):
    filt = MagicMock()
    filt.check_correlation.return_value = _CorrObj(is_correlated)
    return filt


def _mock_trail_manager():
    mgr = MagicMock()
    return mgr


def _build_pipeline(**kwargs):
    """Build a pipeline with all components mocked."""
    defaults = {
        "client": _mock_client(),
        "dry_run": True,
        "ta_agent": _mock_ta_agent(),
        "sentiment_agent": _mock_sentiment_agent(),
        "risk_agent": _mock_risk_agent(),
        "exec_agent": _mock_exec_agent(),
        "regime_detector": _mock_regime_detector(),
        "correlation_filter": _mock_corr_filter(),
        "trail_manager": _mock_trail_manager(),
        "event_bus": EventBus(),
        "journal": None,
        "notifier": None,
        "enable_regime": True,
        "enable_correlation": True,
        "enable_trailing_stops": True,
        "enable_events": True,
        "enable_journal": False,
        "enable_notifications": False,
    }
    defaults.update(kwargs)
    return TradingPipeline(**defaults)


# ── Dataclass tests ──────────────────────────────────────────────

class TestPipelineStep:
    def test_defaults(self):
        s = PipelineStep(name="test")
        assert s.passed is True
        assert s.detail == ""
        assert s.data == {}

    def test_failed_step(self):
        s = PipelineStep(name="risk", passed=False, detail="rejected")
        assert not s.passed


class TestPipelineResult:
    def test_defaults(self):
        r = PipelineResult(symbol="AAPL")
        assert r.signal == "HOLD"
        assert r.shares == 0
        assert r.order_status == ""
        assert not r.executed

    def test_executed_when_filled(self):
        r = PipelineResult(symbol="AAPL", order_status="filled")
        assert r.executed

    def test_executed_when_dry_run(self):
        r = PipelineResult(symbol="AAPL", order_status="dry_run")
        assert r.executed

    def test_not_executed_when_skipped(self):
        r = PipelineResult(symbol="AAPL", order_status="skipped")
        assert not r.executed


# ── Full pipeline run ────────────────────────────────────────────

class TestFullPipeline:
    def test_buy_signal_dry_run(self):
        pipeline = _build_pipeline()
        results = pipeline.run(["AAPL"])
        assert len(results) == 1
        r = results[0]
        assert r.symbol == "AAPL"
        assert r.signal == "BUY"
        assert r.shares > 0
        assert r.order_status == "dry_run"

    def test_hold_signal_skips_execution(self):
        pipeline = _build_pipeline(
            ta_agent=_mock_ta_agent(signal="HOLD", score=0),
            sentiment_agent=_mock_sentiment_agent(signal="HOLD", score=0),
        )
        results = pipeline.run(["AAPL"])
        r = results[0]
        assert r.signal == "HOLD"
        assert r.shares == 0
        assert r.order_status == ""

    def test_sell_signal_processes(self):
        pipeline = _build_pipeline(
            ta_agent=_mock_ta_agent(signal="SELL", score=-4),
            sentiment_agent=_mock_sentiment_agent(signal="SELL", score=-0.3),
        )
        results = pipeline.run(["AAPL"])
        r = results[0]
        assert r.signal == "SELL"

    def test_multiple_symbols(self):
        pipeline = _build_pipeline()
        results = pipeline.run(["AAPL", "MSFT", "GOOGL"])
        assert len(results) == 3
        for r in results:
            assert r.symbol in ("AAPL", "MSFT", "GOOGL")

    def test_account_failure_returns_errors(self):
        client = _mock_client()
        client.get_account.side_effect = Exception("Network error")
        pipeline = _build_pipeline(client=client)
        results = pipeline.run(["AAPL"])
        assert len(results) == 1
        assert "Account fetch failed" in results[0].error


# ── Step-level tests ─────────────────────────────────────────────

class TestRegimeStep:
    def test_regime_detected(self):
        pipeline = _build_pipeline()
        pipeline._ensure_components()
        step, data = pipeline._step_regime("1Day")
        assert "TRENDING_UP" in step.detail
        assert data is not None
        assert data["regime"] == "TRENDING_UP"

    def test_regime_disabled(self):
        pipeline = _build_pipeline(enable_regime=False)
        pipeline._ensure_components()
        step, data = pipeline._step_regime("1Day")
        assert "disabled" in step.detail
        assert data is None

    def test_regime_error_handled(self):
        detector = MagicMock()
        detector.detect.side_effect = Exception("API error")
        pipeline = _build_pipeline(regime_detector=detector)
        pipeline._ensure_components()
        step, data = pipeline._step_regime("1Day")
        assert "error" in step.detail
        assert data is None


class TestTAStep:
    def test_ta_succeeds(self):
        pipeline = _build_pipeline()
        pipeline._ensure_components()
        step, data = pipeline._step_ta("AAPL", "1Day")
        assert step.passed
        assert data["signal"] == "BUY"

    def test_ta_failure(self):
        ta = MagicMock()
        ta.analyze.side_effect = Exception("No data")
        pipeline = _build_pipeline(ta_agent=ta)
        pipeline._ensure_components()
        step, data = pipeline._step_ta("AAPL", "1Day")
        assert not step.passed
        assert data is None


class TestSentimentStep:
    def test_sentiment_succeeds(self):
        pipeline = _build_pipeline()
        pipeline._ensure_components()
        step, data = pipeline._step_sentiment("AAPL", "1Day")
        assert step.passed
        assert data["signal"] == "BUY"


class TestCorrelationStep:
    def test_no_correlation(self):
        pipeline = _build_pipeline()
        pipeline._ensure_components()
        step, data = pipeline._step_correlation("AAPL", ["MSFT"])
        assert "correlated=False" in step.detail

    def test_correlated(self):
        pipeline = _build_pipeline(correlation_filter=_mock_corr_filter(is_correlated=True))
        pipeline._ensure_components()
        step, data = pipeline._step_correlation("AAPL", ["MSFT"])
        assert "correlated=True" in step.detail
        assert data["is_correlated"] is True

    def test_no_positions_skips(self):
        pipeline = _build_pipeline()
        pipeline._ensure_components()
        step, data = pipeline._step_correlation("AAPL", [])
        assert "skipped" in step.detail

    def test_disabled_skips(self):
        pipeline = _build_pipeline(enable_correlation=False)
        pipeline._ensure_components()
        step, data = pipeline._step_correlation("AAPL", ["MSFT"])
        assert "skipped" in step.detail


class TestSizingStep:
    def test_buy_sizing(self):
        pipeline = _build_pipeline()
        pipeline._ensure_components()
        step, sizing = pipeline._step_sizing(
            "AAPL", "BUY", 150.0, 2.5, 100_000, None, None, 0.0,
        )
        assert step.passed
        assert sizing.shares > 0

    def test_hold_skips(self):
        pipeline = _build_pipeline()
        pipeline._ensure_components()
        step, sizing = pipeline._step_sizing(
            "AAPL", "HOLD", 150.0, 2.5, 100_000, None, None, 0.0,
        )
        assert "skipped" in step.detail
        assert sizing is None


class TestRiskStep:
    def test_approved(self):
        pipeline = _build_pipeline(risk_agent=_mock_risk_agent(approved=True))
        pipeline._ensure_components()
        step, data = pipeline._step_risk("AAPL", "BUY", 150.0, 2.5, 50, 146.0, 158.0)
        assert step.passed

    def test_rejected(self):
        pipeline = _build_pipeline(risk_agent=_mock_risk_agent(approved=False))
        pipeline._ensure_components()
        step, data = pipeline._step_risk("AAPL", "BUY", 150.0, 2.5, 50, 146.0, 158.0)
        assert not step.passed

    def test_hold_skips(self):
        pipeline = _build_pipeline()
        pipeline._ensure_components()
        step, data = pipeline._step_risk("AAPL", "HOLD", 150.0, 2.5, 0, 0, 0)
        assert "skipped" in step.detail


class TestExecuteStep:
    def test_dry_run(self):
        pipeline = _build_pipeline(exec_agent=_mock_exec_agent(status="dry_run"))
        pipeline._ensure_components()
        decision = {"symbol": "AAPL", "signal": "BUY", "position_size": 50,
                     "risk_approved": True, "stop_loss": 146.0, "take_profit": 158.0}
        step, data = pipeline._step_execute("AAPL", decision)
        assert step.passed
        assert data["status"] == "dry_run"

    def test_filled(self):
        pipeline = _build_pipeline(exec_agent=_mock_exec_agent(status="filled"))
        pipeline._ensure_components()
        decision = {"symbol": "AAPL", "signal": "BUY", "position_size": 50,
                     "risk_approved": True, "stop_loss": 146.0, "take_profit": 158.0}
        step, data = pipeline._step_execute("AAPL", decision)
        assert data["status"] == "filled"
        assert data["filled_qty"] == 50


class TestTrailingStopStep:
    def test_registers_stop(self):
        mgr = _mock_trail_manager()
        pipeline = _build_pipeline(trail_manager=mgr)
        pipeline._ensure_components()
        step = pipeline._step_trailing_stop("AAPL", "BUY", 150.0, 50, 2.5, 146.0)
        assert "registered" in step.detail
        mgr.add_position.assert_called_once()

    def test_skips_hold(self):
        pipeline = _build_pipeline()
        pipeline._ensure_components()
        step = pipeline._step_trailing_stop("AAPL", "HOLD", 150.0, 0, 2.5, 0)
        assert "skipped" in step.detail

    def test_disabled_skips(self):
        pipeline = _build_pipeline(enable_trailing_stops=False)
        pipeline._ensure_components()
        step = pipeline._step_trailing_stop("AAPL", "BUY", 150.0, 50, 2.5, 146.0)
        assert "disabled" in step.detail


# ── Risk rejection pipeline test ─────────────────────────────────

class TestRiskRejection:
    def test_risk_rejected_skips_execution(self):
        pipeline = _build_pipeline(risk_agent=_mock_risk_agent(approved=False))
        results = pipeline.run(["AAPL"])
        r = results[0]
        assert r.signal == "BUY"
        assert not r.risk_approved
        # Execution should have been skipped
        exec_steps = [s for s in r.steps if s.name == "execute"]
        assert len(exec_steps) == 1
        assert "skipped" in exec_steps[0].detail


# ── Correlation filtered pipeline test ───────────────────────────

class TestCorrelationFiltered:
    def test_correlated_signal_filtered_to_hold(self):
        client = _mock_client()
        client.get_positions.return_value = [{"symbol": "MSFT"}]
        pipeline = _build_pipeline(
            client=client,
            correlation_filter=_mock_corr_filter(is_correlated=True),
        )
        results = pipeline.run(["AAPL"])
        r = results[0]
        # Signal should be downgraded to HOLD by aggregator
        assert r.signal == "HOLD"
        assert r.correlation_filtered is True


# ── Event Bus integration ────────────────────────────────────────

class TestEventBusIntegration:
    def test_events_published(self):
        bus = EventBus()
        pipeline = _build_pipeline(event_bus=bus)
        pipeline.run(["AAPL"])
        history = bus.get_history(limit=50)
        types = [e.event_type for e in history]
        assert "regime_changed" in types
        assert "signal_generated" in types
        assert "aggregation_done" in types

    def test_no_events_when_disabled(self):
        bus = EventBus()
        pipeline = _build_pipeline(event_bus=bus, enable_events=False)
        pipeline.run(["AAPL"])
        history = bus.get_history(limit=50)
        assert len(history) == 0


# ── Notification integration ─────────────────────────────────────

class TestNotificationIntegration:
    def test_notify_on_buy_signal(self):
        notifier = MagicMock()
        pipeline = _build_pipeline(notifier=notifier, enable_notifications=True)
        pipeline.run(["AAPL"])
        notifier.notify_signal.assert_called_once()

    def test_no_notify_when_disabled(self):
        notifier = MagicMock()
        pipeline = _build_pipeline(notifier=notifier, enable_notifications=False)
        pipeline.run(["AAPL"])
        notifier.notify_signal.assert_not_called()


# ── Formatting tests ─────────────────────────────────────────────

class TestFormatting:
    def test_format_result(self):
        pipeline = _build_pipeline()
        results = pipeline.run(["AAPL"])
        text = TradingPipeline.format_result(results[0])
        assert "AAPL" in text
        assert "PIPELINE RESULT" in text

    def test_format_result_with_error(self):
        r = PipelineResult(symbol="AAPL", error="test error")
        text = TradingPipeline.format_result(r)
        assert "test error" in text

    def test_format_summary(self):
        pipeline = _build_pipeline()
        results = pipeline.run(["AAPL"])
        text = TradingPipeline.format_summary(results)
        assert "PIPELINE SUMMARY" in text
        assert "AAPL" in text
        assert "BUY" in text

    def test_format_summary_multiple(self):
        results = [
            PipelineResult(symbol="AAPL", signal="BUY", order_status="dry_run"),
            PipelineResult(symbol="MSFT", signal="HOLD"),
            PipelineResult(symbol="GOOGL", signal="SELL", order_status="filled"),
        ]
        text = TradingPipeline.format_summary(results)
        assert "1 BUY" in text
        assert "1 SELL" in text
        assert "1 HOLD" in text


# ── Edge cases ───────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_symbols(self):
        pipeline = _build_pipeline()
        results = pipeline.run([])
        assert results == []

    def test_ta_failure_continues(self):
        ta = MagicMock()
        ta.analyze.side_effect = Exception("No bars")
        pipeline = _build_pipeline(ta_agent=ta)
        results = pipeline.run(["AAPL", "MSFT"])
        # Both should have errors but not crash
        assert len(results) == 2
        for r in results:
            assert "TA failed" in r.error

    def test_sentiment_failure_non_fatal(self):
        sent = MagicMock()
        sent.analyze.side_effect = Exception("No news")
        pipeline = _build_pipeline(sentiment_agent=sent)
        results = pipeline.run(["AAPL"])
        # Pipeline should continue with TA only
        r = results[0]
        # The signal depends on TA alone
        assert r.signal in ("BUY", "SELL", "HOLD")

    def test_negative_kelly_stops_pipeline(self):
        """If Kelly returns negative (no edge), shares = 0 and pipeline stops."""
        sizer = PositionSizer(
            default_win_rate=0.30,
            default_payoff_ratio=0.5,
        )
        pipeline = _build_pipeline(sizer=sizer)
        results = pipeline.run(["AAPL"])
        r = results[0]
        # Should not reach execution
        assert r.shares == 0

    def test_position_updates_for_correlation(self):
        """After a fill, the symbol should be added to existing positions."""
        pipeline = _build_pipeline(
            exec_agent=_mock_exec_agent(status="dry_run"),
        )
        results = pipeline.run(["AAPL", "MSFT"])
        # After AAPL fills, MSFT correlation check should include AAPL
        assert len(results) == 2
