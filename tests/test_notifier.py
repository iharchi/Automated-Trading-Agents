"""Unit tests for the Notification System.

Tests cover:
    - Event filtering (by type and score threshold)
    - NotificationEvent creation and serialization
    - Convenience methods (notify_signal, notify_order_*, notify_summary)
    - Channel dispatch (mocked)
    - Color helpers for Slack/Discord
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

from utils.notifier import Notifier, NotificationEvent


class TestNotificationEvent(unittest.TestCase):
    """Tests for NotificationEvent dataclass."""

    def test_event_creation(self):
        event = NotificationEvent(
            event_type="signal",
            symbol="AAPL",
            title="BUY Signal",
            message="Generated a buy signal",
            data={"score": 0.5},
        )
        self.assertEqual(event.event_type, "signal")
        self.assertEqual(event.symbol, "AAPL")
        self.assertIsInstance(event.timestamp, datetime)

    def test_event_to_dict(self):
        event = NotificationEvent(
            event_type="order_filled",
            symbol="TSLA",
            title="Order Filled",
            message="Bought 10 shares",
            data={"qty": 10, "price": 250.0},
        )
        d = event.to_dict()
        self.assertEqual(d["event_type"], "order_filled")
        self.assertEqual(d["symbol"], "TSLA")
        self.assertEqual(d["data"]["qty"], 10)
        self.assertIn("timestamp", d)


class TestEventFiltering(unittest.TestCase):
    """Tests for event filtering logic."""

    def test_filter_by_event_type(self):
        notifier = Notifier(enabled_events=["signal", "order_filled"])

        signal_event = NotificationEvent(
            event_type="signal", symbol="AAPL", title="", message="",
            data={"score": 0.5},
        )
        self.assertTrue(notifier._should_notify(signal_event))

        # summary is not in enabled_events
        summary_event = NotificationEvent(
            event_type="summary", symbol="PORTFOLIO", title="", message="",
        )
        self.assertFalse(notifier._should_notify(summary_event))

    def test_filter_by_min_score(self):
        notifier = Notifier(
            enabled_events=["signal"],
            min_signal_score=0.3,
        )

        # Score 0.5 >= 0.3, should pass
        high_score = NotificationEvent(
            event_type="signal", symbol="AAPL", title="", message="",
            data={"score": 0.5},
        )
        self.assertTrue(notifier._should_notify(high_score))

        # Score 0.2 < 0.3, should be filtered
        low_score = NotificationEvent(
            event_type="signal", symbol="AAPL", title="", message="",
            data={"score": 0.2},
        )
        self.assertFalse(notifier._should_notify(low_score))

    def test_negative_score_uses_absolute_value(self):
        notifier = Notifier(
            enabled_events=["signal"],
            min_signal_score=0.3,
        )

        # Score -0.5, abs = 0.5 >= 0.3
        negative_score = NotificationEvent(
            event_type="signal", symbol="AAPL", title="", message="",
            data={"score": -0.5},
        )
        self.assertTrue(notifier._should_notify(negative_score))

    def test_non_signal_events_ignore_score_filter(self):
        notifier = Notifier(
            enabled_events=["order_filled"],
            min_signal_score=0.5,
        )

        # order_filled has no score, should pass
        order_event = NotificationEvent(
            event_type="order_filled", symbol="AAPL", title="", message="",
        )
        self.assertTrue(notifier._should_notify(order_event))


class TestConvenienceMethods(unittest.TestCase):
    """Tests for convenience notification methods."""

    def setUp(self):
        # Create notifier with all channels disabled (console only)
        self.notifier = Notifier()

    def test_notify_signal(self):
        with patch.object(self.notifier, "_send_console") as mock_console:
            result = self.notifier.notify_signal(
                symbol="AAPL",
                signal="BUY",
                score=0.45,
                price=150.0,
            )
            self.assertIn("console", result)
            mock_console.assert_called_once()
            event = mock_console.call_args[0][0]
            self.assertEqual(event.event_type, "signal")
            self.assertEqual(event.data["signal"], "BUY")

    def test_notify_order_placed(self):
        with patch.object(self.notifier, "_send_console") as mock_console:
            result = self.notifier.notify_order_placed(
                symbol="TSLA",
                side="buy",
                qty=10,
                order_type="market",
                order_id="abc123",
            )
            mock_console.assert_called_once()
            event = mock_console.call_args[0][0]
            self.assertEqual(event.event_type, "order_placed")
            self.assertEqual(event.data["qty"], 10)

    def test_notify_order_filled(self):
        with patch.object(self.notifier, "_send_console") as mock_console:
            result = self.notifier.notify_order_filled(
                symbol="MSFT",
                side="sell",
                qty=5,
                avg_price=350.0,
            )
            mock_console.assert_called_once()
            event = mock_console.call_args[0][0]
            self.assertEqual(event.event_type, "order_filled")
            self.assertEqual(event.data["avg_price"], 350.0)

    def test_notify_order_failed(self):
        with patch.object(self.notifier, "_send_console") as mock_console:
            result = self.notifier.notify_order_failed(
                symbol="GOOGL",
                side="buy",
                qty=2,
                error="Insufficient funds",
            )
            mock_console.assert_called_once()
            event = mock_console.call_args[0][0]
            self.assertEqual(event.event_type, "order_failed")
            self.assertIn("Insufficient funds", event.data["error"])

    def test_notify_summary(self):
        with patch.object(self.notifier, "_send_console") as mock_console:
            result = self.notifier.notify_summary(
                title="Daily Summary",
                message="5 signals, 2 orders placed",
                total_signals=5,
                total_orders=2,
            )
            mock_console.assert_called_once()
            event = mock_console.call_args[0][0]
            self.assertEqual(event.event_type, "summary")
            self.assertEqual(event.symbol, "PORTFOLIO")


class TestChannelDispatch(unittest.TestCase):
    """Tests for multi-channel dispatch."""

    def test_disabled_channels_not_called(self):
        notifier = Notifier(
            email_enabled=False,
            slack_enabled=False,
            discord_enabled=False,
            webhook_enabled=False,
        )

        with patch.object(notifier, "_send_email") as mock_email, \
             patch.object(notifier, "_send_slack") as mock_slack, \
             patch.object(notifier, "_send_discord") as mock_discord, \
             patch.object(notifier, "_send_webhook") as mock_webhook:

            notifier.notify_signal("AAPL", "BUY", 0.5, 150.0)

            mock_email.assert_not_called()
            mock_slack.assert_not_called()
            mock_discord.assert_not_called()
            mock_webhook.assert_not_called()

    def test_enabled_channels_called(self):
        notifier = Notifier(
            slack_enabled=True,
            slack_webhook_url="https://hooks.slack.com/test",
            discord_enabled=True,
            discord_webhook_url="https://discord.com/api/webhooks/test",
        )

        with patch.object(notifier, "_send_slack", return_value=True) as mock_slack, \
             patch.object(notifier, "_send_discord", return_value=True) as mock_discord:

            result = notifier.notify_signal("AAPL", "BUY", 0.5, 150.0)

            mock_slack.assert_called_once()
            mock_discord.assert_called_once()
            self.assertTrue(result.get("slack"))
            self.assertTrue(result.get("discord"))


class TestColorHelpers(unittest.TestCase):
    """Tests for color helper methods."""

    def test_buy_signal_green(self):
        event = NotificationEvent(
            event_type="signal", symbol="AAPL", title="", message="",
            data={"signal": "BUY"},
        )
        color = Notifier._get_color_for_event(event)
        self.assertEqual(color, "#4CAF50")  # green

    def test_sell_signal_red(self):
        event = NotificationEvent(
            event_type="signal", symbol="AAPL", title="", message="",
            data={"signal": "SELL"},
        )
        color = Notifier._get_color_for_event(event)
        self.assertEqual(color, "#F44336")  # red

    def test_order_filled_green(self):
        event = NotificationEvent(
            event_type="order_filled", symbol="AAPL", title="", message="",
        )
        color = Notifier._get_color_for_event(event)
        self.assertEqual(color, "#4CAF50")  # green

    def test_order_failed_red(self):
        event = NotificationEvent(
            event_type="order_failed", symbol="AAPL", title="", message="",
        )
        color = Notifier._get_color_for_event(event)
        self.assertEqual(color, "#F44336")  # red

    def test_color_int_conversion(self):
        event = NotificationEvent(
            event_type="order_filled", symbol="AAPL", title="", message="",
        )
        color_int = Notifier._get_color_int_for_event(event)
        self.assertEqual(color_int, 0x4CAF50)


class TestWebhookPosting(unittest.TestCase):
    """Tests for HTTP POST helper."""

    def test_post_json_success(self):
        with patch("utils.notifier.urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = Notifier._post_json(
                "https://example.com/webhook",
                {"test": "data"},
            )
            self.assertTrue(result)

    def test_post_json_failure(self):
        with patch("utils.notifier.urllib.request.urlopen") as mock_urlopen:
            from urllib.error import HTTPError
            mock_urlopen.side_effect = HTTPError(
                "https://example.com", 500, "Server Error", {}, None
            )

            result = Notifier._post_json(
                "https://example.com/webhook",
                {"test": "data"},
            )
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
