"""Market-Hours Scheduler (Unified Pipeline)

Runs the Unified Trading Pipeline on a configurable interval during
market hours.  Sleeps between cycles and stops when the market closes.

Usage:
    python scheduler.py                             # defaults: 15 min interval, dry-run
    python scheduler.py AAPL TSLA --interval 30     # 30 min interval
    python scheduler.py --once                      # single pass then exit
    python scheduler.py --auto-trade                # enable live paper-trade execution
    python scheduler.py --skip-preflight            # skip pre-flight health checks

The scheduler:
    1. Runs pre-flight health checks (API, account, buying power, data).
    2. Checks if the market is open (via Alpaca clock API).
    3. If open, runs the Unified Trading Pipeline for all symbols.
    4. Logs all signals, decisions, and orders via the pipeline's journal.
    5. Publishes events to the Event Bus throughout.
    6. Sleeps for --interval minutes and repeats.
    7. If the market is closed, calculates time until next open and
       waits (or exits if --no-wait is set).
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone

from config.settings import Settings
from utils.alpaca_client import AlpacaClient
from utils.event_bus import EventBus
from utils.notifier import Notifier
from utils.position_sizer import PositionSizer
from utils.preflight import PreflightCheck
from utils.signal_aggregator import SignalAggregator
from utils.trading_pipeline import TradingPipeline

logger = logging.getLogger("scheduler")

# Graceful shutdown
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received — finishing current cycle...")
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── Pipeline factory ─────────────────────────────────────────────

def create_pipeline(
    client: AlpacaClient,
    *,
    dry_run: bool = True,
    notifier: Notifier | None = None,
    event_bus: EventBus | None = None,
) -> TradingPipeline:
    """Build a TradingPipeline wired from Settings."""
    aggregator = SignalAggregator(
        weights={
            "ta": Settings.AGG_TA_WEIGHT,
            "sentiment": Settings.AGG_SENTIMENT_WEIGHT,
            "mtf": Settings.AGG_MTF_WEIGHT,
            "regime": Settings.AGG_REGIME_WEIGHT,
        },
        buy_threshold=Settings.AGG_BUY_THRESHOLD,
        sell_threshold=Settings.AGG_SELL_THRESHOLD,
        min_sources=Settings.AGG_MIN_SOURCES,
        regime_adaptive=Settings.AGG_REGIME_ADAPTIVE,
        agreement_bonus=Settings.AGG_AGREEMENT_BONUS,
    )
    sizer = PositionSizer(
        kelly_factor=Settings.SIZER_KELLY_FACTOR,
        max_position_pct=Settings.SIZER_MAX_POSITION_PCT,
        max_portfolio_heat=Settings.SIZER_MAX_PORTFOLIO_HEAT,
        atr_risk_mult=Settings.SIZER_ATR_RISK_MULT,
        take_profit_ratio=Settings.SIZER_TP_RATIO,
        default_win_rate=Settings.SIZER_DEFAULT_WIN_RATE,
        default_payoff_ratio=Settings.SIZER_DEFAULT_PAYOFF,
        vol_target=Settings.SIZER_VOL_TARGET,
    )

    return TradingPipeline(
        client=client,
        dry_run=dry_run,
        aggregator=aggregator,
        sizer=sizer,
        event_bus=event_bus,
        notifier=notifier,
        enable_regime=Settings.PIPELINE_ENABLE_REGIME,
        enable_correlation=Settings.PIPELINE_ENABLE_CORRELATION,
        enable_trailing_stops=Settings.PIPELINE_ENABLE_TRAILING,
        enable_events=Settings.PIPELINE_ENABLE_EVENTS,
        enable_journal=Settings.PIPELINE_ENABLE_JOURNAL,
        enable_notifications=Settings.PIPELINE_ENABLE_NOTIFY,
    )


def create_notifier() -> Notifier:
    """Build a Notifier from Settings."""
    return Notifier(
        email_enabled=Settings.NOTIFY_EMAIL_ENABLED,
        smtp_host=Settings.NOTIFY_SMTP_HOST,
        smtp_port=Settings.NOTIFY_SMTP_PORT,
        smtp_user=Settings.NOTIFY_SMTP_USER,
        smtp_password=Settings.NOTIFY_SMTP_PASSWORD,
        email_from=Settings.NOTIFY_EMAIL_FROM,
        email_to=Settings.NOTIFY_EMAIL_TO,
        slack_enabled=Settings.NOTIFY_SLACK_ENABLED,
        slack_webhook_url=Settings.NOTIFY_SLACK_WEBHOOK,
        discord_enabled=Settings.NOTIFY_DISCORD_ENABLED,
        discord_webhook_url=Settings.NOTIFY_DISCORD_WEBHOOK,
        webhook_enabled=Settings.NOTIFY_WEBHOOK_ENABLED,
        webhook_url=Settings.NOTIFY_WEBHOOK_URL,
        enabled_events=Settings.NOTIFY_ENABLED_EVENTS,
        min_signal_score=Settings.NOTIFY_MIN_SIGNAL_SCORE,
    )


# ── Cycle runner ─────────────────────────────────────────────────

def run_cycle(
    pipeline: TradingPipeline,
    symbols: list[str],
    timeframe: str,
    cycle_number: int = 0,
) -> list:
    """Run one full analysis + trade cycle via the unified pipeline.

    Returns:
        List of PipelineResult objects.
    """
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(f"\n--- Cycle #{cycle_number} start: {now} ---")

    try:
        results = pipeline.run(symbols, timeframe=timeframe)

        # Print per-symbol results
        for r in results:
            print(TradingPipeline.format_result(r))

        # Print summary table
        print(TradingPipeline.format_summary(results))

        # Stats
        buys = sum(1 for r in results if r.signal == "BUY")
        sells = sum(1 for r in results if r.signal == "SELL")
        executed = sum(1 for r in results if r.executed)
        errors = sum(1 for r in results if r.error)

        print(
            f"--- Cycle #{cycle_number} complete: "
            f"{buys} BUY / {sells} SELL / {executed} executed / "
            f"{errors} errors ---\n"
        )

        return results

    except Exception as e:
        logger.error("Cycle %d failed: %s", cycle_number, e)
        print(f"--- Cycle #{cycle_number} FAILED: {e} ---\n")
        return []


# ── Market hours wait ────────────────────────────────────────────

def wait_for_market_open(client: AlpacaClient, no_wait: bool) -> bool:
    """Wait until market opens. Returns False if --no-wait and market is closed."""
    clock = client.api.get_clock()
    if clock.is_open:
        return True

    if no_wait:
        logger.info("Market is closed and --no-wait is set. Exiting.")
        return False

    next_open = clock.next_open
    now = clock.timestamp
    wait_seconds = (next_open - now).total_seconds()

    if wait_seconds > 0:
        hours = int(wait_seconds // 3600)
        minutes = int((wait_seconds % 3600) // 60)
        logger.info(
            "Market closed. Next open in %dh %dm. Waiting...", hours, minutes
        )
        # Sleep in small increments so we can catch shutdown signals
        while wait_seconds > 0 and not _shutdown:
            time.sleep(min(wait_seconds, 60))
            wait_seconds -= 60

    return not _shutdown


# ── Pre-flight ───────────────────────────────────────────────────

def run_preflight(
    client: AlpacaClient,
    *,
    min_buying_power: float = 1000.0,
    require_market_open: bool = False,
) -> bool:
    """Run pre-flight health checks. Returns True if all pass."""
    pf = PreflightCheck(
        client,
        min_buying_power=min_buying_power,
        require_market_open=require_market_open,
    )
    report = pf.run_all()
    print(PreflightCheck.format_report(report))

    if not report.passed:
        logger.error("Pre-flight checks FAILED. Aborting scheduler.")
        return False

    if report.warnings:
        logger.warning(
            "Pre-flight passed with %d warning(s).", len(report.warnings),
        )

    return True


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Market-Hours Scheduler (Unified Pipeline)")
    parser.add_argument("symbols", nargs="*", help="Ticker symbols")
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Minutes between analysis cycles (default from config)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Exit if market is closed instead of waiting",
    )
    parser.add_argument(
        "--auto-trade",
        action="store_true",
        help="Enable paper-trade order execution (default is dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run pipeline without placing real orders (default)",
    )
    parser.add_argument(
        "--timeframe",
        default="1Day",
        choices=["1Min", "5Min", "15Min", "1Hour", "1Day"],
        help="Bar timeframe (default: 1Day)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip pre-flight health checks",
    )
    parser.add_argument(
        "--min-buying-power",
        type=float,
        default=None,
        help="Minimum buying power for preflight (default from config)",
    )
    parser.add_argument(
        "--show-events",
        action="store_true",
        help="Show event bus history after each cycle",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, Settings.LOG_LEVEL, logging.INFO),
        format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve settings
    symbols = args.symbols or Settings.DEFAULT_SYMBOLS
    interval = args.interval if args.interval is not None else Settings.SCHED_INTERVAL
    dry_run = not args.auto_trade  # auto-trade overrides dry-run
    min_bp = args.min_buying_power if args.min_buying_power is not None else Settings.PREFLIGHT_MIN_BUYING_POWER
    no_wait = args.no_wait or Settings.SCHED_NO_WAIT

    mode = "AUTO-TRADE" if args.auto_trade else "DRY-RUN"
    print(f"\n{'=' * 62}")
    print(f"  AUTOMATED TRADING SCHEDULER")
    print(f"{'=' * 62}")
    print(f"  Mode         : {mode}")
    print(f"  Symbols      : {', '.join(symbols)}")
    print(f"  Interval     : {interval} min")
    print(f"  Timeframe    : {args.timeframe}")
    print(f"  Regime       : {'ON' if Settings.PIPELINE_ENABLE_REGIME else 'OFF'}")
    print(f"  Correlation  : {'ON' if Settings.PIPELINE_ENABLE_CORRELATION else 'OFF'}")
    print(f"  Sizer        : Kelly ({Settings.SIZER_KELLY_FACTOR:.0%})")
    print(f"  Trail stops  : {'ON' if Settings.PIPELINE_ENABLE_TRAILING else 'OFF'}")
    print(f"  Journal      : {'ON' if Settings.PIPELINE_ENABLE_JOURNAL else 'OFF'}")
    print(f"  Notifications: {'ON' if Settings.PIPELINE_ENABLE_NOTIFY else 'OFF'}")
    print(f"  Schedule     : {'single pass' if args.once else 'continuous'}")
    print(f"{'=' * 62}\n")

    # Initialise client
    client = AlpacaClient()

    # ── Pre-flight checks ────────────────────────────────────
    if not args.skip_preflight:
        if not run_preflight(
            client,
            min_buying_power=min_bp,
            require_market_open=False,
        ):
            sys.exit(1)
    else:
        print("  [SKIP] Pre-flight checks skipped.\n")

    # ── Build pipeline components ────────────────────────────
    notifier_instance = create_notifier() if Settings.PIPELINE_ENABLE_NOTIFY else None
    bus = EventBus(
        strict=Settings.EVENTBUS_STRICT,
        max_history=Settings.EVENTBUS_MAX_HISTORY,
    ) if Settings.PIPELINE_ENABLE_EVENTS else None

    pipeline = create_pipeline(
        client,
        dry_run=dry_run,
        notifier=notifier_instance,
        event_bus=bus,
    )

    # ── Single-pass mode ─────────────────────────────────────
    if args.once:
        results = run_cycle(pipeline, symbols, args.timeframe, cycle_number=1)
        if bus and args.show_events:
            print(EventBus.format_history(bus.get_history(limit=50)))
        return

    # ── Continuous loop ──────────────────────────────────────
    cycle = 0
    while not _shutdown:
        if not wait_for_market_open(client, no_wait):
            break

        if _shutdown:
            break

        cycle += 1
        results = run_cycle(pipeline, symbols, args.timeframe, cycle_number=cycle)

        if bus and args.show_events:
            print(EventBus.format_history(bus.get_history(limit=30)))

        if _shutdown:
            break

        # Check if market is still open before sleeping
        if not client.is_market_open():
            logger.info("Market has closed. Cycle complete for today.")
            if no_wait:
                break
            continue  # will wait_for_market_open on next iteration

        logger.info("Sleeping %d minutes until next cycle...", interval)
        sleep_seconds = interval * 60
        while sleep_seconds > 0 and not _shutdown:
            time.sleep(min(sleep_seconds, 10))
            sleep_seconds -= 10

    print(f"\nScheduler stopped after {cycle} cycle(s).")


if __name__ == "__main__":
    main()
