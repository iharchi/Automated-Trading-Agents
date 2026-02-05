"""Alert / Event Bus

Lightweight publish-subscribe event system for inter-component communication.
Components can publish events and subscribe handlers that react to them,
enabling loose coupling between agents, utilities, and the main pipeline.

Supported events:
    signal_generated  — When any agent produces a BUY/SELL/HOLD signal
    regime_changed    — When market regime shifts (e.g. RANGING → TRENDING_UP)
    correlation_alert — When a new trade would breach correlation limits
    stop_triggered    — When a trailing stop is hit
    position_opened   — When a new position is entered
    position_closed   — When a position is exited
    risk_breach       — When a portfolio risk limit is approached or exceeded
    aggregation_done  — When signal aggregation completes for a symbol
    sizing_done       — When position sizing completes
    custom            — User-defined event type

Usage:
    bus = EventBus()
    bus.subscribe("signal_generated", my_handler)
    bus.publish("signal_generated", {"symbol": "AAPL", "signal": "BUY"})

    # Async-style callback:
    @bus.on("regime_changed")
    def on_regime_change(event):
        print(f"Regime changed to {event.data['new_regime']}")
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Well-known event types ────────────────────────────────────────

EVENT_TYPES = [
    "signal_generated",
    "regime_changed",
    "correlation_alert",
    "stop_triggered",
    "position_opened",
    "position_closed",
    "risk_breach",
    "aggregation_done",
    "sizing_done",
    "custom",
]

Handler = Callable[["Event"], None]


@dataclass
class Event:
    """An event published on the bus."""

    event_type: str
    data: dict = field(default_factory=dict)
    source: str = ""          # Component that published the event
    symbol: str = ""          # Associated ticker (if any)
    timestamp: float = 0.0    # Unix timestamp (auto-set if 0)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class Subscription:
    """A registered handler for a specific event type."""

    event_type: str
    handler: Handler
    name: str = ""            # Optional label for debugging
    priority: int = 0         # Higher = executed first
    once: bool = False        # If True, auto-unsubscribe after first call


class EventBus:
    """Lightweight synchronous pub/sub event bus."""

    def __init__(self, *, strict: bool = False, max_history: int = 100):
        """
        Args:
            strict: If True, only allow well-known event types.
            max_history: Number of recent events to retain in history.
        """
        self._subscribers: dict[str, list[Subscription]] = defaultdict(list)
        self._history: list[Event] = []
        self._max_history = max_history
        self._strict = strict
        self._paused = False

    # ── Subscribe ─────────────────────────────────────────────────

    def subscribe(
        self,
        event_type: str,
        handler: Handler,
        *,
        name: str = "",
        priority: int = 0,
        once: bool = False,
    ) -> Subscription:
        """Register a handler for an event type.

        Args:
            event_type: Event type to listen for, or "*" for all events.
            handler: Callable that receives an Event.
            name: Optional label for debugging.
            priority: Higher priority handlers run first.
            once: Automatically unsubscribe after first invocation.

        Returns:
            The Subscription object (can be used for unsubscribe).
        """
        if self._strict and event_type not in EVENT_TYPES and event_type != "*":
            raise ValueError(
                f"Unknown event type '{event_type}'. "
                f"Known types: {EVENT_TYPES}"
            )

        sub = Subscription(
            event_type=event_type,
            handler=handler,
            name=name or getattr(handler, "__name__", ""),
            priority=priority,
            once=once,
        )
        self._subscribers[event_type].append(sub)
        # Sort by priority (descending)
        self._subscribers[event_type].sort(key=lambda s: s.priority, reverse=True)
        return sub

    def on(self, event_type: str, *, priority: int = 0, once: bool = False):
        """Decorator form of subscribe.

        Usage:
            @bus.on("signal_generated")
            def handle_signal(event):
                ...
        """
        def decorator(fn: Handler) -> Handler:
            self.subscribe(event_type, fn, priority=priority, once=once)
            return fn
        return decorator

    def unsubscribe(self, subscription: Subscription) -> bool:
        """Remove a subscription.

        Returns:
            True if the subscription was found and removed.
        """
        subs = self._subscribers.get(subscription.event_type, [])
        try:
            subs.remove(subscription)
            return True
        except ValueError:
            return False

    def unsubscribe_all(self, event_type: str | None = None) -> int:
        """Remove all handlers for an event type (or all types if None).

        Returns:
            Number of subscriptions removed.
        """
        if event_type is None:
            count = sum(len(subs) for subs in self._subscribers.values())
            self._subscribers.clear()
            return count
        else:
            count = len(self._subscribers.get(event_type, []))
            self._subscribers[event_type] = []
            return count

    # ── Publish ───────────────────────────────────────────────────

    def publish(
        self,
        event_type: str,
        data: dict | None = None,
        *,
        source: str = "",
        symbol: str = "",
    ) -> Event:
        """Publish an event to all subscribers.

        Args:
            event_type: The event type string.
            data: Event payload dictionary.
            source: Name of the publishing component.
            symbol: Associated ticker symbol (if any).

        Returns:
            The Event that was published.
        """
        if self._strict and event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown event type '{event_type}'")

        event = Event(
            event_type=event_type,
            data=data or {},
            source=source,
            symbol=symbol,
        )

        # Store in history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        if self._paused:
            logger.debug("EventBus paused, event queued: %s", event_type)
            return event

        # Dispatch to type-specific handlers
        self._dispatch(event_type, event)

        # Dispatch to wildcard handlers
        if event_type != "*":
            self._dispatch("*", event)

        return event

    def _dispatch(self, event_type: str, event: Event) -> None:
        """Dispatch event to subscribers, handling errors and once-subs."""
        subs = self._subscribers.get(event_type, [])
        to_remove: list[Subscription] = []

        for sub in subs:
            try:
                sub.handler(event)
            except Exception:
                logger.exception(
                    "Error in handler '%s' for event '%s'",
                    sub.name, event.event_type,
                )
            if sub.once:
                to_remove.append(sub)

        for sub in to_remove:
            try:
                subs.remove(sub)
            except ValueError:
                pass

    # ── Pause / Resume ────────────────────────────────────────────

    def pause(self) -> None:
        """Pause event dispatch (events still recorded in history)."""
        self._paused = True

    def resume(self) -> None:
        """Resume event dispatch."""
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    # ── Querying ──────────────────────────────────────────────────

    def get_history(
        self,
        event_type: str | None = None,
        symbol: str | None = None,
        limit: int = 50,
    ) -> list[Event]:
        """Return recent events, optionally filtered.

        Args:
            event_type: Filter by type (None = all).
            symbol: Filter by symbol (None = all).
            limit: Max events to return.

        Returns:
            List of matching events (newest first).
        """
        events = reversed(self._history)
        results = []
        for e in events:
            if event_type and e.event_type != event_type:
                continue
            if symbol and e.symbol != symbol:
                continue
            results.append(e)
            if len(results) >= limit:
                break
        return results

    def get_subscriber_count(self, event_type: str | None = None) -> int:
        """Return the number of active subscriptions."""
        if event_type is None:
            return sum(len(subs) for subs in self._subscribers.values())
        return len(self._subscribers.get(event_type, []))

    def list_event_types(self) -> list[str]:
        """Return event types that have active subscribers."""
        return [k for k, v in self._subscribers.items() if v]

    def clear_history(self) -> None:
        """Clear the event history."""
        self._history.clear()

    # ── Convenience publishers ────────────────────────────────────

    def emit_signal(
        self,
        symbol: str,
        signal: str,
        score: float,
        source: str = "",
        **extra,
    ) -> Event:
        """Publish a signal_generated event."""
        return self.publish(
            "signal_generated",
            {"signal": signal, "score": score, **extra},
            source=source,
            symbol=symbol,
        )

    def emit_regime_change(
        self,
        symbol: str,
        old_regime: str,
        new_regime: str,
        confidence: float = 0.0,
        **extra,
    ) -> Event:
        """Publish a regime_changed event."""
        return self.publish(
            "regime_changed",
            {
                "old_regime": old_regime,
                "new_regime": new_regime,
                "confidence": confidence,
                **extra,
            },
            source="MarketRegimeDetector",
            symbol=symbol,
        )

    def emit_stop_triggered(
        self,
        symbol: str,
        stop_price: float,
        current_price: float,
        side: str = "long",
        **extra,
    ) -> Event:
        """Publish a stop_triggered event."""
        return self.publish(
            "stop_triggered",
            {
                "stop_price": stop_price,
                "current_price": current_price,
                "side": side,
                **extra,
            },
            source="TrailingStopManager",
            symbol=symbol,
        )

    def emit_risk_breach(
        self,
        rule: str,
        detail: str,
        symbol: str = "",
        **extra,
    ) -> Event:
        """Publish a risk_breach event."""
        return self.publish(
            "risk_breach",
            {"rule": rule, "detail": detail, **extra},
            source="RiskManagement",
            symbol=symbol,
        )

    # ── Formatting ────────────────────────────────────────────────

    @staticmethod
    def format_event(event: Event) -> str:
        """Format a single event for display."""
        ts = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
        symbol_str = f" [{event.symbol}]" if event.symbol else ""
        source_str = f" ({event.source})" if event.source else ""
        lines = [
            f"  {ts}{symbol_str} {event.event_type}{source_str}",
        ]
        for k, v in event.data.items():
            lines.append(f"    {k}: {v}")
        return "\n".join(lines)

    @staticmethod
    def format_history(events: list[Event]) -> str:
        """Format a list of events for display."""
        if not events:
            return "\n  No events.\n"

        lines = [
            f"\n{'=' * 62}",
            f"  EVENT HISTORY  ({len(events)} events)",
            f"{'=' * 62}",
        ]
        for event in events:
            lines.append(EventBus.format_event(event))
            lines.append(f"  {'~' * 58}")
        lines.append(f"{'=' * 62}\n")
        return "\n".join(lines)
