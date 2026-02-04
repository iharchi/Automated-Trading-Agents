"""Notification System

Multi-channel alerting for trading signals and order events.

Supported channels:
    - Console (logging) — always enabled
    - Email (SMTP)
    - Slack (webhook)
    - Discord (webhook)
    - Generic webhook (custom HTTP POST)

Events:
    - signal: A BUY/SELL signal was generated
    - order_placed: An order was submitted to Alpaca
    - order_filled: An order was filled
    - order_failed: An order submission failed
    - summary: Daily/periodic summary

Configuration via config.yaml + env vars (secrets).
"""

import json
import logging
import smtplib
import ssl
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Literal

logger = logging.getLogger(__name__)

EventType = Literal["signal", "order_placed", "order_filled", "order_failed", "summary"]


@dataclass
class NotificationEvent:
    """Represents a notification to be sent."""

    event_type: EventType
    symbol: str
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "symbol": self.symbol,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


class Notifier:
    """Multi-channel notification manager."""

    def __init__(
        self,
        *,
        # Email config
        email_enabled: bool = False,
        smtp_host: str = "",
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_password: str = "",
        email_from: str = "",
        email_to: list[str] | None = None,
        # Slack config
        slack_enabled: bool = False,
        slack_webhook_url: str = "",
        # Discord config
        discord_enabled: bool = False,
        discord_webhook_url: str = "",
        # Generic webhook config
        webhook_enabled: bool = False,
        webhook_url: str = "",
        webhook_headers: dict | None = None,
        # Filtering
        enabled_events: list[EventType] | None = None,
        min_signal_score: float = 0.0,
    ):
        # Email
        self.email_enabled = email_enabled
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.email_from = email_from
        self.email_to = email_to or []

        # Slack
        self.slack_enabled = slack_enabled
        self.slack_webhook_url = slack_webhook_url

        # Discord
        self.discord_enabled = discord_enabled
        self.discord_webhook_url = discord_webhook_url

        # Generic webhook
        self.webhook_enabled = webhook_enabled
        self.webhook_url = webhook_url
        self.webhook_headers = webhook_headers or {}

        # Filtering
        self.enabled_events: set[EventType] = set(
            enabled_events or ["signal", "order_placed", "order_filled", "order_failed", "summary"]
        )
        self.min_signal_score = min_signal_score

    # ── Event filtering ───────────────────────────────────────

    def _should_notify(self, event: NotificationEvent) -> bool:
        """Check if this event should trigger notifications."""
        if event.event_type not in self.enabled_events:
            return False

        # For signals, check minimum score threshold
        if event.event_type == "signal":
            score = abs(event.data.get("score", 0))
            if score < self.min_signal_score:
                return False

        return True

    # ── Console (always on) ───────────────────────────────────

    @staticmethod
    def _send_console(event: NotificationEvent) -> None:
        """Log notification to console."""
        logger.info(
            "[%s] %s: %s — %s",
            event.event_type.upper(),
            event.symbol,
            event.title,
            event.message,
        )

    # ── Email ─────────────────────────────────────────────────

    def _send_email(self, event: NotificationEvent) -> bool:
        """Send notification via SMTP email."""
        if not self.email_enabled or not self.email_to:
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[Trading Alert] {event.title}"
            msg["From"] = self.email_from
            msg["To"] = ", ".join(self.email_to)

            # Plain text body
            text_body = f"""
Trading Alert: {event.title}

Symbol: {event.symbol}
Event: {event.event_type}
Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

{event.message}

Data:
{json.dumps(event.data, indent=2)}
"""
            # HTML body
            html_body = f"""
<html>
<body>
<h2>Trading Alert: {event.title}</h2>
<table>
<tr><td><strong>Symbol:</strong></td><td>{event.symbol}</td></tr>
<tr><td><strong>Event:</strong></td><td>{event.event_type}</td></tr>
<tr><td><strong>Time:</strong></td><td>{event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
</table>
<p>{event.message}</p>
<pre>{json.dumps(event.data, indent=2)}</pre>
</body>
</html>
"""
            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.email_from, self.email_to, msg.as_string())

            logger.debug("Email sent for %s", event.symbol)
            return True
        except Exception as e:
            logger.error("Email notification failed: %s", e)
            return False

    # ── Slack ─────────────────────────────────────────────────

    def _send_slack(self, event: NotificationEvent) -> bool:
        """Send notification to Slack via webhook."""
        if not self.slack_enabled or not self.slack_webhook_url:
            return False

        try:
            # Build Slack message with blocks for better formatting
            color = self._get_color_for_event(event)
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "blocks": [
                            {
                                "type": "header",
                                "text": {
                                    "type": "plain_text",
                                    "text": f"{event.title}",
                                },
                            },
                            {
                                "type": "section",
                                "fields": [
                                    {"type": "mrkdwn", "text": f"*Symbol:*\n{event.symbol}"},
                                    {"type": "mrkdwn", "text": f"*Event:*\n{event.event_type}"},
                                    {
                                        "type": "mrkdwn",
                                        "text": f"*Time:*\n{event.timestamp.strftime('%H:%M:%S')}",
                                    },
                                ],
                            },
                            {
                                "type": "section",
                                "text": {"type": "mrkdwn", "text": event.message},
                            },
                        ],
                    }
                ]
            }

            return self._post_json(self.slack_webhook_url, payload)
        except Exception as e:
            logger.error("Slack notification failed: %s", e)
            return False

    # ── Discord ───────────────────────────────────────────────

    def _send_discord(self, event: NotificationEvent) -> bool:
        """Send notification to Discord via webhook."""
        if not self.discord_enabled or not self.discord_webhook_url:
            return False

        try:
            color = self._get_color_int_for_event(event)
            payload = {
                "embeds": [
                    {
                        "title": event.title,
                        "description": event.message,
                        "color": color,
                        "fields": [
                            {"name": "Symbol", "value": event.symbol, "inline": True},
                            {"name": "Event", "value": event.event_type, "inline": True},
                            {
                                "name": "Time",
                                "value": event.timestamp.strftime("%H:%M:%S"),
                                "inline": True,
                            },
                        ],
                        "timestamp": event.timestamp.isoformat(),
                    }
                ]
            }

            return self._post_json(self.discord_webhook_url, payload)
        except Exception as e:
            logger.error("Discord notification failed: %s", e)
            return False

    # ── Generic webhook ───────────────────────────────────────

    def _send_webhook(self, event: NotificationEvent) -> bool:
        """Send notification to a generic webhook endpoint."""
        if not self.webhook_enabled or not self.webhook_url:
            return False

        try:
            payload = event.to_dict()
            return self._post_json(self.webhook_url, payload, self.webhook_headers)
        except Exception as e:
            logger.error("Webhook notification failed: %s", e)
            return False

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _post_json(url: str, payload: dict, extra_headers: dict | None = None) -> bool:
        """POST JSON to a URL."""
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status in (200, 201, 204)
        except urllib.error.HTTPError as e:
            logger.error("Webhook HTTP error: %s %s", e.code, e.reason)
            return False
        except urllib.error.URLError as e:
            logger.error("Webhook URL error: %s", e.reason)
            return False

    @staticmethod
    def _get_color_for_event(event: NotificationEvent) -> str:
        """Get hex color for Slack attachment based on event type."""
        colors = {
            "signal": "#2196F3",       # blue
            "order_placed": "#FF9800", # orange
            "order_filled": "#4CAF50", # green
            "order_failed": "#F44336", # red
            "summary": "#9C27B0",      # purple
        }
        # Override for BUY/SELL signals
        if event.event_type == "signal":
            signal = event.data.get("signal", "")
            if signal == "BUY":
                return "#4CAF50"  # green
            elif signal == "SELL":
                return "#F44336"  # red
        return colors.get(event.event_type, "#607D8B")

    @staticmethod
    def _get_color_int_for_event(event: NotificationEvent) -> int:
        """Get integer color for Discord embed."""
        hex_color = Notifier._get_color_for_event(event)
        return int(hex_color.lstrip("#"), 16)

    # ── Public API ────────────────────────────────────────────

    def notify(self, event: NotificationEvent) -> dict[str, bool]:
        """Send notification across all enabled channels.

        Returns dict of channel → success status.
        """
        results = {"console": True}

        if not self._should_notify(event):
            logger.debug("Event filtered out: %s %s", event.event_type, event.symbol)
            return results

        # Always log to console
        self._send_console(event)

        # Send to enabled channels
        if self.email_enabled:
            results["email"] = self._send_email(event)

        if self.slack_enabled:
            results["slack"] = self._send_slack(event)

        if self.discord_enabled:
            results["discord"] = self._send_discord(event)

        if self.webhook_enabled:
            results["webhook"] = self._send_webhook(event)

        return results

    def notify_signal(
        self,
        symbol: str,
        signal: str,
        score: float,
        price: float,
        agent: str = "Portfolio Manager",
        **extra_data,
    ) -> dict[str, bool]:
        """Convenience method for signal notifications."""
        event = NotificationEvent(
            event_type="signal",
            symbol=symbol,
            title=f"{signal} Signal: {symbol}",
            message=f"{agent} generated a {signal} signal for {symbol} at ${price:.2f} (score: {score:.2f})",
            data={
                "signal": signal,
                "score": score,
                "price": price,
                "agent": agent,
                **extra_data,
            },
        )
        return self.notify(event)

    def notify_order_placed(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "market",
        order_id: str = "",
        **extra_data,
    ) -> dict[str, bool]:
        """Convenience method for order placed notifications."""
        event = NotificationEvent(
            event_type="order_placed",
            symbol=symbol,
            title=f"Order Placed: {side.upper()} {qty} {symbol}",
            message=f"Submitted {order_type} order to {side} {qty} shares of {symbol}",
            data={
                "side": side,
                "qty": qty,
                "order_type": order_type,
                "order_id": order_id,
                **extra_data,
            },
        )
        return self.notify(event)

    def notify_order_filled(
        self,
        symbol: str,
        side: str,
        qty: int,
        avg_price: float,
        order_id: str = "",
        **extra_data,
    ) -> dict[str, bool]:
        """Convenience method for order filled notifications."""
        event = NotificationEvent(
            event_type="order_filled",
            symbol=symbol,
            title=f"Order Filled: {side.upper()} {qty} {symbol} @ ${avg_price:.2f}",
            message=f"Filled {side} order for {qty} shares of {symbol} at ${avg_price:.2f}",
            data={
                "side": side,
                "qty": qty,
                "avg_price": avg_price,
                "order_id": order_id,
                **extra_data,
            },
        )
        return self.notify(event)

    def notify_order_failed(
        self,
        symbol: str,
        side: str,
        qty: int,
        error: str,
        **extra_data,
    ) -> dict[str, bool]:
        """Convenience method for order failed notifications."""
        event = NotificationEvent(
            event_type="order_failed",
            symbol=symbol,
            title=f"Order Failed: {side.upper()} {qty} {symbol}",
            message=f"Failed to {side} {qty} shares of {symbol}: {error}",
            data={
                "side": side,
                "qty": qty,
                "error": error,
                **extra_data,
            },
        )
        return self.notify(event)

    def notify_summary(
        self,
        title: str,
        message: str,
        **extra_data,
    ) -> dict[str, bool]:
        """Convenience method for summary notifications."""
        event = NotificationEvent(
            event_type="summary",
            symbol="PORTFOLIO",
            title=title,
            message=message,
            data=extra_data,
        )
        return self.notify(event)
