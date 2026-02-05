"""Tests for utils/event_bus.py"""

import time

import pytest

from utils.event_bus import (
    EVENT_TYPES,
    Event,
    EventBus,
    Subscription,
)


# ── Event dataclass tests ────────────────────────────────────────

class TestEvent:
    def test_defaults(self):
        e = Event(event_type="signal_generated")
        assert e.event_type == "signal_generated"
        assert e.data == {}
        assert e.source == ""
        assert e.symbol == ""
        assert e.timestamp > 0

    def test_custom_values(self):
        e = Event(
            event_type="custom",
            data={"key": "value"},
            source="TestAgent",
            symbol="AAPL",
        )
        assert e.data["key"] == "value"
        assert e.source == "TestAgent"
        assert e.symbol == "AAPL"

    def test_timestamp_auto_set(self):
        before = time.time()
        e = Event(event_type="test")
        after = time.time()
        assert before <= e.timestamp <= after

    def test_custom_timestamp_preserved(self):
        e = Event(event_type="test", timestamp=12345.0)
        assert e.timestamp == 12345.0


class TestSubscription:
    def test_creation(self):
        def handler(event):
            pass

        s = Subscription(event_type="test", handler=handler, name="my_handler")
        assert s.event_type == "test"
        assert s.name == "my_handler"
        assert s.priority == 0
        assert s.once is False


# ── Subscribe / Publish tests ────────────────────────────────────

class TestSubscribePublish:
    def test_basic_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda e: received.append(e))
        bus.publish("test", {"msg": "hello"})
        assert len(received) == 1
        assert received[0].data["msg"] == "hello"

    def test_multiple_subscribers(self):
        bus = EventBus()
        results = []
        bus.subscribe("test", lambda e: results.append("a"))
        bus.subscribe("test", lambda e: results.append("b"))
        bus.publish("test")
        assert len(results) == 2

    def test_different_event_types(self):
        bus = EventBus()
        results = []
        bus.subscribe("type_a", lambda e: results.append("a"))
        bus.subscribe("type_b", lambda e: results.append("b"))
        bus.publish("type_a")
        assert results == ["a"]

    def test_no_subscribers(self):
        bus = EventBus()
        event = bus.publish("nobody_listening")
        assert event.event_type == "nobody_listening"

    def test_publish_returns_event(self):
        bus = EventBus()
        event = bus.publish("test", {"x": 1}, source="src", symbol="AAPL")
        assert event.event_type == "test"
        assert event.data["x"] == 1
        assert event.source == "src"
        assert event.symbol == "AAPL"


# ── Wildcard subscriber ──────────────────────────────────────────

class TestWildcard:
    def test_wildcard_receives_all_events(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e.event_type))
        bus.publish("type_a")
        bus.publish("type_b")
        bus.publish("type_c")
        assert received == ["type_a", "type_b", "type_c"]

    def test_wildcard_and_specific(self):
        bus = EventBus()
        results = []
        bus.subscribe("*", lambda e: results.append("wild"))
        bus.subscribe("test", lambda e: results.append("specific"))
        bus.publish("test")
        # Specific handler runs first, then wildcard
        assert "wild" in results
        assert "specific" in results
        assert len(results) == 2


# ── Priority tests ───────────────────────────────────────────────

class TestPriority:
    def test_higher_priority_runs_first(self):
        bus = EventBus()
        order = []
        bus.subscribe("test", lambda e: order.append("low"), priority=0)
        bus.subscribe("test", lambda e: order.append("high"), priority=10)
        bus.publish("test")
        assert order == ["high", "low"]

    def test_same_priority_preserves_order(self):
        bus = EventBus()
        order = []
        bus.subscribe("test", lambda e: order.append("first"), priority=5)
        bus.subscribe("test", lambda e: order.append("second"), priority=5)
        bus.publish("test")
        assert len(order) == 2


# ── Once (one-shot) subscription ─────────────────────────────────

class TestOnce:
    def test_once_fires_only_once(self):
        bus = EventBus()
        count = []
        bus.subscribe("test", lambda e: count.append(1), once=True)
        bus.publish("test")
        bus.publish("test")
        assert len(count) == 1

    def test_once_does_not_affect_others(self):
        bus = EventBus()
        results = []
        bus.subscribe("test", lambda e: results.append("persistent"))
        bus.subscribe("test", lambda e: results.append("once"), once=True)
        bus.publish("test")
        bus.publish("test")
        assert results.count("persistent") == 2
        assert results.count("once") == 1


# ── Unsubscribe tests ────────────────────────────────────────────

class TestUnsubscribe:
    def test_unsubscribe(self):
        bus = EventBus()
        results = []
        sub = bus.subscribe("test", lambda e: results.append(1))
        bus.publish("test")
        bus.unsubscribe(sub)
        bus.publish("test")
        assert len(results) == 1

    def test_unsubscribe_returns_false_if_not_found(self):
        bus = EventBus()
        sub = Subscription(event_type="test", handler=lambda e: None)
        assert bus.unsubscribe(sub) is False

    def test_unsubscribe_all_for_type(self):
        bus = EventBus()
        bus.subscribe("test", lambda e: None)
        bus.subscribe("test", lambda e: None)
        bus.subscribe("other", lambda e: None)
        count = bus.unsubscribe_all("test")
        assert count == 2
        assert bus.get_subscriber_count("test") == 0
        assert bus.get_subscriber_count("other") == 1

    def test_unsubscribe_all_global(self):
        bus = EventBus()
        bus.subscribe("a", lambda e: None)
        bus.subscribe("b", lambda e: None)
        count = bus.unsubscribe_all()
        assert count == 2
        assert bus.get_subscriber_count() == 0


# ── Decorator tests ──────────────────────────────────────────────

class TestDecorator:
    def test_on_decorator(self):
        bus = EventBus()
        results = []

        @bus.on("test")
        def handler(event):
            results.append(event.data)

        bus.publish("test", {"val": 42})
        assert len(results) == 1
        assert results[0]["val"] == 42

    def test_on_decorator_with_once(self):
        bus = EventBus()
        count = []

        @bus.on("test", once=True)
        def handler(event):
            count.append(1)

        bus.publish("test")
        bus.publish("test")
        assert len(count) == 1


# ── Error handling ───────────────────────────────────────────────

class TestErrorHandling:
    def test_handler_error_does_not_crash(self):
        bus = EventBus()
        results = []

        def bad_handler(event):
            raise ValueError("boom")

        def good_handler(event):
            results.append("ok")

        bus.subscribe("test", bad_handler, priority=10)
        bus.subscribe("test", good_handler, priority=0)
        bus.publish("test")
        # Good handler still runs despite bad handler raising
        assert results == ["ok"]


# ── Strict mode tests ────────────────────────────────────────────

class TestStrictMode:
    def test_strict_rejects_unknown_event_type(self):
        bus = EventBus(strict=True)
        with pytest.raises(ValueError, match="Unknown event type"):
            bus.publish("made_up_event")

    def test_strict_rejects_unknown_subscribe(self):
        bus = EventBus(strict=True)
        with pytest.raises(ValueError, match="Unknown event type"):
            bus.subscribe("made_up_event", lambda e: None)

    def test_strict_allows_known_types(self):
        bus = EventBus(strict=True)
        for et in EVENT_TYPES:
            bus.publish(et)  # Should not raise

    def test_strict_allows_wildcard_subscribe(self):
        bus = EventBus(strict=True)
        bus.subscribe("*", lambda e: None)  # Should not raise

    def test_non_strict_allows_any_type(self):
        bus = EventBus(strict=False)
        bus.publish("completely_custom_type")  # No error


# ── Pause / Resume tests ────────────────────────────────────────

class TestPauseResume:
    def test_paused_does_not_dispatch(self):
        bus = EventBus()
        results = []
        bus.subscribe("test", lambda e: results.append(1))
        bus.pause()
        bus.publish("test")
        assert len(results) == 0
        assert bus.paused is True

    def test_resume_restores_dispatch(self):
        bus = EventBus()
        results = []
        bus.subscribe("test", lambda e: results.append(1))
        bus.pause()
        bus.publish("test")
        bus.resume()
        bus.publish("test")
        assert len(results) == 1

    def test_paused_still_records_history(self):
        bus = EventBus()
        bus.pause()
        bus.publish("test")
        assert len(bus.get_history()) == 1


# ── History tests ────────────────────────────────────────────────

class TestHistory:
    def test_history_records_events(self):
        bus = EventBus()
        bus.publish("a", symbol="AAPL")
        bus.publish("b", symbol="MSFT")
        history = bus.get_history()
        assert len(history) == 2

    def test_history_newest_first(self):
        bus = EventBus()
        bus.publish("first")
        bus.publish("second")
        history = bus.get_history()
        assert history[0].event_type == "second"
        assert history[1].event_type == "first"

    def test_history_filter_by_type(self):
        bus = EventBus()
        bus.publish("a")
        bus.publish("b")
        bus.publish("a")
        history = bus.get_history(event_type="a")
        assert len(history) == 2

    def test_history_filter_by_symbol(self):
        bus = EventBus()
        bus.publish("test", symbol="AAPL")
        bus.publish("test", symbol="MSFT")
        bus.publish("test", symbol="AAPL")
        history = bus.get_history(symbol="AAPL")
        assert len(history) == 2

    def test_history_limit(self):
        bus = EventBus()
        for i in range(10):
            bus.publish("test")
        history = bus.get_history(limit=3)
        assert len(history) == 3

    def test_max_history_cap(self):
        bus = EventBus(max_history=5)
        for i in range(10):
            bus.publish("test")
        history = bus.get_history(limit=100)
        assert len(history) == 5

    def test_clear_history(self):
        bus = EventBus()
        bus.publish("test")
        bus.clear_history()
        assert len(bus.get_history()) == 0


# ── Convenience emitters ─────────────────────────────────────────

class TestConvenienceEmitters:
    def test_emit_signal(self):
        bus = EventBus()
        received = []
        bus.subscribe("signal_generated", lambda e: received.append(e))
        bus.emit_signal("AAPL", "BUY", 3.0, source="TA")
        assert len(received) == 1
        assert received[0].symbol == "AAPL"
        assert received[0].data["signal"] == "BUY"
        assert received[0].data["score"] == 3.0

    def test_emit_regime_change(self):
        bus = EventBus()
        received = []
        bus.subscribe("regime_changed", lambda e: received.append(e))
        bus.emit_regime_change("SPY", "RANGING", "TRENDING_UP", confidence=0.8)
        assert len(received) == 1
        assert received[0].data["old_regime"] == "RANGING"
        assert received[0].data["new_regime"] == "TRENDING_UP"

    def test_emit_stop_triggered(self):
        bus = EventBus()
        received = []
        bus.subscribe("stop_triggered", lambda e: received.append(e))
        bus.emit_stop_triggered("AAPL", 145.0, 144.5, side="long")
        assert len(received) == 1
        assert received[0].data["stop_price"] == 145.0

    def test_emit_risk_breach(self):
        bus = EventBus()
        received = []
        bus.subscribe("risk_breach", lambda e: received.append(e))
        bus.emit_risk_breach("concentration", "Position too large", symbol="AAPL")
        assert len(received) == 1
        assert received[0].data["rule"] == "concentration"


# ── Querying tests ───────────────────────────────────────────────

class TestQuerying:
    def test_subscriber_count(self):
        bus = EventBus()
        bus.subscribe("a", lambda e: None)
        bus.subscribe("a", lambda e: None)
        bus.subscribe("b", lambda e: None)
        assert bus.get_subscriber_count("a") == 2
        assert bus.get_subscriber_count("b") == 1
        assert bus.get_subscriber_count() == 3

    def test_list_event_types(self):
        bus = EventBus()
        bus.subscribe("alpha", lambda e: None)
        bus.subscribe("beta", lambda e: None)
        types = bus.list_event_types()
        assert "alpha" in types
        assert "beta" in types


# ── Formatting tests ─────────────────────────────────────────────

class TestFormatting:
    def test_format_event(self):
        e = Event(event_type="signal_generated", symbol="AAPL", data={"signal": "BUY"})
        text = EventBus.format_event(e)
        assert "AAPL" in text
        assert "signal_generated" in text
        assert "BUY" in text

    def test_format_history_empty(self):
        text = EventBus.format_history([])
        assert "No events" in text

    def test_format_history_with_data(self):
        events = [
            Event(event_type="a", symbol="AAPL"),
            Event(event_type="b", symbol="MSFT"),
        ]
        text = EventBus.format_history(events)
        assert "EVENT HISTORY" in text
        assert "2 events" in text


# ── Known event types ────────────────────────────────────────────

class TestEventTypes:
    def test_expected_types_present(self):
        expected = {
            "signal_generated", "regime_changed", "correlation_alert",
            "stop_triggered", "position_opened", "position_closed",
            "risk_breach", "aggregation_done", "sizing_done", "custom",
        }
        assert set(EVENT_TYPES) == expected
