"""Pre-flight Health Checks

Runs a series of checks before the trading scheduler starts its first cycle
to ensure all systems are operational.

Checks:
    1. API Connectivity   — Can we reach the Alpaca API?
    2. Account Status     — Is the account ACTIVE?
    3. Buying Power       — Is buying power above a minimum threshold?
    4. Market Clock       — Is the market currently open (or when does it open)?
    5. Data Availability  — Can we fetch bars for a reference symbol?
    6. Config Validation  — Are required config values present?

Usage:
    from utils.preflight import PreflightCheck
    pf = PreflightCheck(client)
    report = pf.run_all()
    if not report.passed:
        print(PreflightCheck.format_report(report))
        sys.exit(1)
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result from a single health check."""

    name: str
    passed: bool = True
    detail: str = ""
    warning: bool = False  # Passed but with a caveat


@dataclass
class PreflightReport:
    """Aggregated results from all pre-flight checks."""

    checks: list[CheckResult] = field(default_factory=list)
    timestamp: float = 0.0
    duration_ms: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def passed(self) -> bool:
        """All critical checks passed (warnings are OK)."""
        return all(c.passed for c in self.checks)

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.warning]

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


class PreflightCheck:
    """Pre-flight health check suite for the trading system."""

    def __init__(
        self,
        client=None,
        *,
        min_buying_power: float = 1000.0,
        reference_symbol: str = "SPY",
        require_market_open: bool = False,
    ):
        """
        Args:
            client: AlpacaClient instance (or None to skip API checks).
            min_buying_power: Minimum buying power to pass (default $1000).
            reference_symbol: Symbol to test data retrieval.
            require_market_open: If True, fail when market is closed.
        """
        self.client = client
        self.min_buying_power = min_buying_power
        self.reference_symbol = reference_symbol
        self.require_market_open = require_market_open

    # ── Individual checks ─────────────────────────────────────────

    def check_api_connectivity(self) -> CheckResult:
        """Check that we can reach the Alpaca API."""
        if self.client is None:
            return CheckResult(
                name="api_connectivity",
                passed=False,
                detail="No client provided",
            )
        try:
            account = self.client.get_account()
            return CheckResult(
                name="api_connectivity",
                passed=True,
                detail=f"Connected (account={account.get('id', 'unknown')})",
            )
        except Exception as e:
            return CheckResult(
                name="api_connectivity",
                passed=False,
                detail=f"API unreachable: {e}",
            )

    def check_account_status(self, account: dict | None = None) -> CheckResult:
        """Check that the account is ACTIVE."""
        if self.client is None:
            return CheckResult(
                name="account_status",
                passed=False,
                detail="No client provided",
            )
        try:
            if account is None:
                account = self.client.get_account()
            status = account.get("status", "UNKNOWN")
            ok = status == "ACTIVE"
            return CheckResult(
                name="account_status",
                passed=ok,
                detail=f"Status={status}",
            )
        except Exception as e:
            return CheckResult(
                name="account_status",
                passed=False,
                detail=f"Could not fetch account: {e}",
            )

    def check_buying_power(self, account: dict | None = None) -> CheckResult:
        """Check that buying power meets the minimum threshold."""
        if self.client is None:
            return CheckResult(
                name="buying_power",
                passed=False,
                detail="No client provided",
            )
        try:
            if account is None:
                account = self.client.get_account()
            bp = account.get("buying_power", 0)
            ok = bp >= self.min_buying_power
            warning = ok and bp < self.min_buying_power * 5  # Low but acceptable
            return CheckResult(
                name="buying_power",
                passed=ok,
                detail=f"${bp:,.2f} (min=${self.min_buying_power:,.2f})",
                warning=warning,
            )
        except Exception as e:
            return CheckResult(
                name="buying_power",
                passed=False,
                detail=f"Could not fetch account: {e}",
            )

    def check_market_clock(self) -> CheckResult:
        """Check market open/close status."""
        if self.client is None:
            return CheckResult(
                name="market_clock",
                passed=not self.require_market_open,
                detail="No client provided",
            )
        try:
            clock = self.client.api.get_clock()
            is_open = clock.is_open

            if is_open:
                return CheckResult(
                    name="market_clock",
                    passed=True,
                    detail="Market is OPEN",
                )
            else:
                next_open = clock.next_open
                now = clock.timestamp
                wait = (next_open - now).total_seconds()
                hours = int(wait // 3600)
                minutes = int((wait % 3600) // 60)
                return CheckResult(
                    name="market_clock",
                    passed=not self.require_market_open,
                    detail=f"Market CLOSED — opens in {hours}h {minutes}m",
                    warning=True,
                )
        except Exception as e:
            return CheckResult(
                name="market_clock",
                passed=False,
                detail=f"Clock API error: {e}",
            )

    def check_data_availability(self) -> CheckResult:
        """Check that we can fetch historical data."""
        if self.client is None:
            return CheckResult(
                name="data_availability",
                passed=False,
                detail="No client provided",
            )
        try:
            bars = self.client.get_bars(
                self.reference_symbol, timeframe="1Day", limit=5,
            )
            rows = len(bars)
            ok = rows > 0
            return CheckResult(
                name="data_availability",
                passed=ok,
                detail=f"{self.reference_symbol}: {rows} bars fetched",
            )
        except Exception as e:
            return CheckResult(
                name="data_availability",
                passed=False,
                detail=f"Data fetch failed: {e}",
            )

    def check_config(self) -> CheckResult:
        """Check that essential configuration values are set."""
        from config.settings import Settings

        issues = []
        if not Settings.validate():
            issues.append("Alpaca API keys missing or placeholder")
        if not Settings.DEFAULT_SYMBOLS:
            issues.append("No default symbols configured")

        if issues:
            return CheckResult(
                name="config",
                passed=False,
                detail="; ".join(issues),
            )
        return CheckResult(
            name="config",
            passed=True,
            detail=f"OK ({len(Settings.DEFAULT_SYMBOLS)} symbols configured)",
        )

    # ── Run all checks ────────────────────────────────────────────

    def run_all(self) -> PreflightReport:
        """Run all pre-flight checks and return a report.

        Optimised to reuse the account fetch across multiple checks.
        """
        start = time.time()
        report = PreflightReport()

        # Config check first (no API needed)
        report.checks.append(self.check_config())

        # API connectivity
        api_check = self.check_api_connectivity()
        report.checks.append(api_check)

        if not api_check.passed:
            # Skip remaining API-dependent checks
            report.checks.append(
                CheckResult(name="account_status", passed=False, detail="Skipped (API down)")
            )
            report.checks.append(
                CheckResult(name="buying_power", passed=False, detail="Skipped (API down)")
            )
            report.checks.append(
                CheckResult(name="market_clock", passed=False, detail="Skipped (API down)")
            )
            report.checks.append(
                CheckResult(name="data_availability", passed=False, detail="Skipped (API down)")
            )
            report.duration_ms = (time.time() - start) * 1000
            return report

        # Fetch account once and share across checks
        account = None
        try:
            account = self.client.get_account()
        except Exception:
            pass

        report.checks.append(self.check_account_status(account))
        report.checks.append(self.check_buying_power(account))
        report.checks.append(self.check_market_clock())
        report.checks.append(self.check_data_availability())

        report.duration_ms = (time.time() - start) * 1000
        return report

    # ── Formatting ────────────────────────────────────────────────

    @staticmethod
    def format_report(report: PreflightReport) -> str:
        """Human-readable formatting of a preflight report."""
        lines = [
            f"\n{'=' * 62}",
            f"  PRE-FLIGHT HEALTH CHECK",
            f"{'=' * 62}",
        ]

        for check in report.checks:
            if check.passed and not check.warning:
                icon = "PASS"
            elif check.passed and check.warning:
                icon = "WARN"
            else:
                icon = "FAIL"
            lines.append(f"  [{icon}] {check.name:22s} {check.detail}")

        lines.append(f"{'~' * 62}")

        if report.passed:
            status = "ALL CHECKS PASSED"
            if report.warnings:
                status += f" ({len(report.warnings)} warning(s))"
        else:
            status = f"FAILED ({len(report.failures)} check(s) failed)"

        lines.append(f"  Result: {status}")
        lines.append(f"  Duration: {report.duration_ms:.0f}ms")
        lines.append(f"{'=' * 62}\n")
        return "\n".join(lines)
