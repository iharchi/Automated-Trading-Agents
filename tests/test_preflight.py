"""Tests for utils/preflight.py"""

import time
from unittest.mock import MagicMock, patch

import pytest

from utils.preflight import CheckResult, PreflightCheck, PreflightReport


# ── CheckResult tests ────────────────────────────────────────────


class TestCheckResult:
    def test_defaults(self):
        c = CheckResult(name="test")
        assert c.name == "test"
        assert c.passed is True
        assert c.detail == ""
        assert c.warning is False

    def test_failed_check(self):
        c = CheckResult(name="api", passed=False, detail="timeout")
        assert c.passed is False
        assert c.detail == "timeout"

    def test_warning_check(self):
        c = CheckResult(name="bp", passed=True, warning=True, detail="low")
        assert c.passed is True
        assert c.warning is True


# ── PreflightReport tests ────────────────────────────────────────


class TestPreflightReport:
    def test_empty_report_passes(self):
        report = PreflightReport()
        assert report.passed is True
        assert report.failures == []
        assert report.warnings == []

    def test_auto_timestamp(self):
        before = time.time()
        report = PreflightReport()
        after = time.time()
        assert before <= report.timestamp <= after

    def test_all_pass(self):
        report = PreflightReport(checks=[
            CheckResult(name="a", passed=True),
            CheckResult(name="b", passed=True),
        ])
        assert report.passed is True
        assert len(report.failures) == 0

    def test_one_failure(self):
        report = PreflightReport(checks=[
            CheckResult(name="a", passed=True),
            CheckResult(name="b", passed=False, detail="boom"),
        ])
        assert report.passed is False
        assert len(report.failures) == 1
        assert report.failures[0].name == "b"

    def test_warning_does_not_fail(self):
        report = PreflightReport(checks=[
            CheckResult(name="a", passed=True, warning=True),
        ])
        assert report.passed is True
        assert len(report.warnings) == 1

    def test_duration_ms(self):
        report = PreflightReport(duration_ms=42.5)
        assert report.duration_ms == 42.5


# ── PreflightCheck — no client ───────────────────────────────────


class TestPreflightNoClient:
    def test_no_client_api_fails(self):
        pf = PreflightCheck(client=None)
        result = pf.check_api_connectivity()
        assert result.passed is False
        assert "No client" in result.detail

    def test_no_client_account_status_fails(self):
        pf = PreflightCheck(client=None)
        result = pf.check_account_status()
        assert result.passed is False

    def test_no_client_buying_power_fails(self):
        pf = PreflightCheck(client=None)
        result = pf.check_buying_power()
        assert result.passed is False

    def test_no_client_market_clock_passes_if_not_required(self):
        pf = PreflightCheck(client=None, require_market_open=False)
        result = pf.check_market_clock()
        assert result.passed is True

    def test_no_client_market_clock_fails_if_required(self):
        pf = PreflightCheck(client=None, require_market_open=True)
        result = pf.check_market_clock()
        assert result.passed is False

    def test_no_client_data_fails(self):
        pf = PreflightCheck(client=None)
        result = pf.check_data_availability()
        assert result.passed is False

    def test_no_client_run_all(self):
        """run_all with no client should fail (API down path)."""
        pf = PreflightCheck(client=None)
        report = pf.run_all()
        assert report.passed is False
        assert len(report.checks) == 6  # config + api + 4 skipped


# ── PreflightCheck — mocked client ───────────────────────────────


def _mock_client(
    account=None,
    clock_open=True,
    bars_count=5,
    api_error=False,
):
    """Create a mock AlpacaClient."""
    client = MagicMock()

    if api_error:
        client.get_account.side_effect = Exception("Connection refused")
    else:
        if account is None:
            account = {
                "id": "test-123",
                "status": "ACTIVE",
                "cash": 50000.0,
                "portfolio_value": 100000.0,
                "buying_power": 100000.0,
                "equity": 100000.0,
                "currency": "USD",
            }
        client.get_account.return_value = account

    # Clock
    clock = MagicMock()
    clock.is_open = clock_open
    if not clock_open:
        import datetime
        clock.next_open = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=12)
        clock.timestamp = datetime.datetime.now(datetime.timezone.utc)
    client.api = MagicMock()
    client.api.get_clock.return_value = clock

    # Bars
    import pandas as pd
    if bars_count > 0:
        client.get_bars.return_value = pd.DataFrame({"close": [100.0] * bars_count})
    else:
        client.get_bars.return_value = pd.DataFrame()

    return client


class TestPreflightApiConnectivity:
    def test_successful_connection(self):
        client = _mock_client()
        pf = PreflightCheck(client)
        result = pf.check_api_connectivity()
        assert result.passed is True
        assert "test-123" in result.detail

    def test_api_error(self):
        client = _mock_client(api_error=True)
        pf = PreflightCheck(client)
        result = pf.check_api_connectivity()
        assert result.passed is False
        assert "unreachable" in result.detail.lower() or "Connection refused" in result.detail


class TestPreflightAccountStatus:
    def test_active_account(self):
        client = _mock_client()
        account = client.get_account()
        pf = PreflightCheck(client)
        result = pf.check_account_status(account)
        assert result.passed is True
        assert "ACTIVE" in result.detail

    def test_inactive_account(self):
        client = _mock_client(account={
            "id": "test-123", "status": "SUSPENDED",
            "cash": 0, "portfolio_value": 0,
            "buying_power": 0, "equity": 0, "currency": "USD",
        })
        account = client.get_account()
        pf = PreflightCheck(client)
        result = pf.check_account_status(account)
        assert result.passed is False
        assert "SUSPENDED" in result.detail

    def test_account_fetch_without_cached(self):
        """When no account passed, it fetches internally."""
        client = _mock_client()
        pf = PreflightCheck(client)
        result = pf.check_account_status()
        assert result.passed is True

    def test_account_fetch_error(self):
        client = _mock_client(api_error=True)
        pf = PreflightCheck(client)
        result = pf.check_account_status()
        assert result.passed is False


class TestPreflightBuyingPower:
    def test_sufficient_buying_power(self):
        client = _mock_client()
        pf = PreflightCheck(client, min_buying_power=1000.0)
        result = pf.check_buying_power()
        assert result.passed is True
        assert "$100,000" in result.detail

    def test_insufficient_buying_power(self):
        account = {
            "id": "x", "status": "ACTIVE",
            "cash": 500, "portfolio_value": 500,
            "buying_power": 500, "equity": 500, "currency": "USD",
        }
        client = _mock_client(account=account)
        pf = PreflightCheck(client, min_buying_power=1000.0)
        result = pf.check_buying_power()
        assert result.passed is False

    def test_low_buying_power_warning(self):
        """BP above min but below 5x min → warning."""
        account = {
            "id": "x", "status": "ACTIVE",
            "cash": 2000, "portfolio_value": 2000,
            "buying_power": 2000, "equity": 2000, "currency": "USD",
        }
        client = _mock_client(account=account)
        pf = PreflightCheck(client, min_buying_power=1000.0)
        result = pf.check_buying_power()
        assert result.passed is True
        assert result.warning is True

    def test_high_buying_power_no_warning(self):
        """BP well above 5x min → no warning."""
        client = _mock_client()  # 100k buying power
        pf = PreflightCheck(client, min_buying_power=1000.0)
        result = pf.check_buying_power()
        assert result.passed is True
        assert result.warning is False


class TestPreflightMarketClock:
    def test_market_open(self):
        client = _mock_client(clock_open=True)
        pf = PreflightCheck(client)
        result = pf.check_market_clock()
        assert result.passed is True
        assert "OPEN" in result.detail

    def test_market_closed_not_required(self):
        client = _mock_client(clock_open=False)
        pf = PreflightCheck(client, require_market_open=False)
        result = pf.check_market_clock()
        assert result.passed is True
        assert result.warning is True
        assert "CLOSED" in result.detail

    def test_market_closed_required(self):
        client = _mock_client(clock_open=False)
        pf = PreflightCheck(client, require_market_open=True)
        result = pf.check_market_clock()
        assert result.passed is False

    def test_clock_api_error(self):
        client = _mock_client()
        client.api.get_clock.side_effect = Exception("API error")
        pf = PreflightCheck(client)
        result = pf.check_market_clock()
        assert result.passed is False
        assert "error" in result.detail.lower()


class TestPreflightDataAvailability:
    def test_data_available(self):
        client = _mock_client(bars_count=5)
        pf = PreflightCheck(client, reference_symbol="SPY")
        result = pf.check_data_availability()
        assert result.passed is True
        assert "5 bars" in result.detail

    def test_no_data(self):
        client = _mock_client(bars_count=0)
        pf = PreflightCheck(client)
        result = pf.check_data_availability()
        assert result.passed is False

    def test_data_fetch_error(self):
        client = _mock_client()
        client.get_bars.side_effect = Exception("timeout")
        pf = PreflightCheck(client)
        result = pf.check_data_availability()
        assert result.passed is False
        assert "timeout" in result.detail


class TestPreflightConfig:
    def test_valid_config(self):
        pf = PreflightCheck(client=None)
        with patch("config.settings.Settings") as MockSettings:
            MockSettings.validate.return_value = True
            MockSettings.DEFAULT_SYMBOLS = ["AAPL", "MSFT"]
            result = pf.check_config()
        assert result.name == "config"
        assert result.passed is True
        assert "2 symbols" in result.detail

    def test_config_check_returns_check_result(self):
        pf = PreflightCheck(client=None)
        result = pf.check_config()
        assert isinstance(result, CheckResult)
        assert result.name == "config"


# ── run_all integration ──────────────────────────────────────────


class TestRunAll:
    def test_all_pass(self):
        client = _mock_client()
        pf = PreflightCheck(client, min_buying_power=100.0)
        # Patch config check to always pass
        with patch.object(pf, "check_config", return_value=CheckResult(name="config", passed=True, detail="OK")):
            report = pf.run_all()
        assert report.passed is True
        assert len(report.checks) == 6
        assert report.duration_ms >= 0

    def test_api_down_skips_remaining(self):
        client = _mock_client(api_error=True)
        pf = PreflightCheck(client)
        with patch.object(pf, "check_config", return_value=CheckResult(name="config", passed=True)):
            report = pf.run_all()
        # Should have config + api_connectivity + 4 skipped
        assert len(report.checks) == 6
        assert report.passed is False
        # Verify skipped checks
        skipped = [c for c in report.checks if "Skipped" in c.detail]
        assert len(skipped) == 4

    def test_reuses_account_fetch(self):
        """Verify get_account is called only once for efficiency."""
        client = _mock_client()
        pf = PreflightCheck(client, min_buying_power=100.0)
        with patch.object(pf, "check_config", return_value=CheckResult(name="config", passed=True)):
            report = pf.run_all()
        # get_account called once in check_api_connectivity + once in run_all for sharing
        # But the shared account is passed to check_account_status and check_buying_power
        # so those don't call get_account again.
        # Total: check_api_connectivity calls it once, run_all calls it once = 2
        assert client.get_account.call_count == 2

    def test_report_with_warnings(self):
        account = {
            "id": "x", "status": "ACTIVE",
            "cash": 2000, "portfolio_value": 2000,
            "buying_power": 2000, "equity": 2000, "currency": "USD",
        }
        client = _mock_client(account=account, clock_open=False)
        pf = PreflightCheck(client, min_buying_power=1000.0)
        with patch.object(pf, "check_config", return_value=CheckResult(name="config", passed=True)):
            report = pf.run_all()
        assert report.passed is True
        assert len(report.warnings) >= 1  # buying power warning + market closed warning


# ── Formatting ───────────────────────────────────────────────────


class TestFormatting:
    def test_format_all_pass(self):
        report = PreflightReport(
            checks=[
                CheckResult(name="api", passed=True, detail="Connected"),
                CheckResult(name="account", passed=True, detail="ACTIVE"),
            ],
            duration_ms=123,
        )
        text = PreflightCheck.format_report(report)
        assert "PRE-FLIGHT" in text
        assert "PASS" in text
        assert "ALL CHECKS PASSED" in text
        assert "123ms" in text

    def test_format_with_failure(self):
        report = PreflightReport(
            checks=[
                CheckResult(name="api", passed=False, detail="timeout"),
            ],
        )
        text = PreflightCheck.format_report(report)
        assert "FAIL" in text
        assert "FAILED" in text

    def test_format_with_warning(self):
        report = PreflightReport(
            checks=[
                CheckResult(name="bp", passed=True, warning=True, detail="low"),
            ],
        )
        text = PreflightCheck.format_report(report)
        assert "WARN" in text
        assert "warning" in text.lower()
