"""Market-Hours Scheduler (Unified Pipeline)

Runs the Unified Trading Pipeline on a configurable interval during
market hours.  Sleeps between cycles and stops when the market closes.

Usage:
    python scheduler.py                             # defaults: 15 min interval, dry-run
    python scheduler.py AAPL TSLA --interval 30     # 30 min interval
    python scheduler.py --once                      # single pass then exit
    python scheduler.py --auto-trade                # enable live paper-trade execution
    python scheduler.py --skip-preflight            # skip pre-flight health checks
    python scheduler.py --daemon                    # run as background daemon
    python scheduler.py --scan                      # use scanner mode instead of fixed symbols

The scheduler:
    1. Runs pre-flight health checks (API, account, buying power, data).
    2. Checks if the market is open (via Alpaca clock API).
    3. If open, runs the Unified Trading Pipeline for all symbols.
    4. Logs all signals, decisions, and orders via the pipeline's journal.
    5. Publishes events to the Event Bus throughout.
    6. Sleeps for --interval minutes and repeats.
    7. If the market is closed, calculates time until next open and
       waits (or exits if --no-wait is set).

Daemon mode:
    Run as a background process that automatically starts trading at market
    open and stops at market close. Use --daemon flag to enable.

    To start:  python scheduler.py --daemon --auto-trade
    To stop:   python scheduler.py --stop
    To status: python scheduler.py --status
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config.settings import Settings
from utils.alpaca_client import AlpacaClient
from utils.event_bus import EventBus
from utils.notifier import Notifier
from utils.position_sizer import PositionSizer
from utils.preflight import PreflightCheck
from utils.signal_aggregator import SignalAggregator
from utils.trading_pipeline import TradingPipeline
from agents.scanner_agent import ScannerAgent, UNIVERSES
from utils.finviz_scanner import FinvizScanner
from utils.news_analyzer import NewsAnalyzer
from utils.fast_pipeline import FastPipeline, QueuedTrade
from utils.scalping_manager import ScalpingManager, ScalpPosition

logger = logging.getLogger("scheduler")

# PID file for daemon mode
PID_FILE = Path(__file__).parent / ".scheduler.pid"

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


def run_scan_cycle(
    client: AlpacaClient,
    universe: str,
    dry_run: bool,
    cycle_number: int = 0,
) -> dict:
    """Run one scan cycle using the Scanner Agent.

    Returns:
        Scan summary dict.
    """
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(f"\n--- Scan Cycle #{cycle_number} start: {now} ---")

    try:
        scanner = ScannerAgent(
            client=client,
            auto_trade=not dry_run,
            dry_run=dry_run,
            max_workers=8,
            max_trades=5,
            min_score=0.05,
            top_n=15,
        )

        # Use the scan method with execute=True
        summary = scanner.scan(universe=universe, execute=not dry_run)

        print(ScannerAgent.format_summary(summary))

        # Extract stats from summary
        buy_signals = summary.get("buy_signals", 0)
        sell_signals = summary.get("sell_signals", 0)
        trades_executed = summary.get("trades_executed", 0)

        print(
            f"--- Scan Cycle #{cycle_number} complete: "
            f"{buy_signals} BUY / {sell_signals} SELL / "
            f"{trades_executed} executed ---\n"
        )

        return summary

    except Exception as e:
        logger.error("Scan cycle %d failed: %s", cycle_number, e)
        print(f"--- Scan Cycle #{cycle_number} FAILED: {e} ---\n")
        return {}


def run_finviz_cycle(
    pipeline: TradingPipeline,
    screen: str,
    timeframe: str,
    cycle_number: int = 0,
    limit: int = 15,
    check_news: bool = False,
    require_news: bool = False,
) -> list:
    """Run one cycle using Finviz scanner to find opportunities.

    Args:
        pipeline: Trading pipeline to run
        screen: Finviz screen to use
        timeframe: Bar timeframe
        cycle_number: Cycle counter
        limit: Max stocks to scan
        check_news: Whether to analyze news for each stock
        require_news: Only trade stocks with recent news/catalyst

    Returns:
        List of PipelineResult objects.
    """
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(f"\n--- Finviz Cycle #{cycle_number} start: {now} ---")
    print(f"    Screen: {screen}")
    if check_news:
        print(f"    News Check: ON (require: {require_news})")

    try:
        # Initialize Finviz scanner
        finviz = FinvizScanner(min_price=5.0, max_price=500.0)

        # Get stocks from Finviz
        if screen == "BUY_CANDIDATES":
            stocks = finviz.get_buy_candidates(limit=limit)
        else:
            stocks = finviz.get_screen(screen, limit=limit)

        if not stocks:
            print(f"  No stocks found from Finviz {screen} screen")
            return []

        # Extract symbols
        symbols = [s.symbol for s in stocks]
        print(f"  Finviz found: {', '.join(symbols)}")

        # Print Finviz scanner results
        print(FinvizScanner.format_results(stocks))

        # Analyze news if requested
        news_results = {}
        if check_news:
            print("\n  Analyzing news for candidates...")
            news_analyzer = NewsAnalyzer()
            news_results = news_analyzer.analyze_batch(symbols, days=2)
            print(NewsAnalyzer.format_results(news_results))

            # Filter to only stocks with news/catalysts if required
            if require_news:
                filtered_symbols = []
                for symbol in symbols:
                    analysis = news_results.get(symbol)
                    if analysis and analysis.has_news:
                        # Prefer stocks with catalysts or positive sentiment
                        if analysis.has_catalyst or analysis.sentiment_score > 0:
                            filtered_symbols.append(symbol)

                if not filtered_symbols:
                    print("  No stocks with news catalysts found.")
                    return []

                print(f"  Filtered to {len(filtered_symbols)} stocks with news: {', '.join(filtered_symbols)}")
                symbols = filtered_symbols

        # Run pipeline on these symbols
        results = pipeline.run(symbols, timeframe=timeframe)

        # Print per-symbol results with news info
        for r in results:
            result_str = TradingPipeline.format_result(r)
            # Add news info if available
            if r.symbol in news_results:
                news = news_results[r.symbol]
                if news.has_news:
                    catalyst_str = f" [{news.catalyst_type}]" if news.has_catalyst else ""
                    news_str = f"  NEWS: {news.sentiment} ({news.sentiment_score:+.2f}){catalyst_str}"
                    result_str += f"\n{news_str}"
            print(result_str)

        # Print summary table
        print(TradingPipeline.format_summary(results))

        # Stats
        buys = sum(1 for r in results if r.signal == "BUY")
        sells = sum(1 for r in results if r.signal == "SELL")
        executed = sum(1 for r in results if r.executed)
        errors = sum(1 for r in results if r.error)
        with_news = sum(1 for s in symbols if s in news_results and news_results[s].has_news)

        print(
            f"--- Finviz Cycle #{cycle_number} complete: "
            f"{buys} BUY / {sells} SELL / {executed} executed / "
            f"{errors} errors"
            + (f" / {with_news} with news ---\n" if check_news else " ---\n")
        )

        return results

    except ImportError as e:
        logger.error("Finviz not available: %s", e)
        print("  ERROR: finvizfinance not installed. Run: pip install finvizfinance")
        return []
    except Exception as e:
        logger.error("Finviz cycle %d failed: %s", cycle_number, e)
        print(f"--- Finviz Cycle #{cycle_number} FAILED: {e} ---\n")
        return []


def run_scalp_cycle(
    fast_pipeline: FastPipeline,
    symbols: list[str],
    timeframe: str,
    cycle_number: int = 0,
    execute: bool = True,
    scalp_manager: ScalpingManager | None = None,
    validate_symbols: bool = True,
    min_price: float = 1.0,
    min_volume: int = 10000,
) -> list:
    """Run one fast scalping cycle with parallel processing.

    Args:
        fast_pipeline: FastPipeline instance
        symbols: Stock tickers to analyze
        timeframe: Bar timeframe (1Min, 5Min)
        cycle_number: Cycle counter
        execute: Whether to execute trades
        scalp_manager: Optional ScalpingManager for position tracking
        validate_symbols: Whether to filter out non-tradeable symbols first
        min_price: Minimum stock price (default $1)
        min_volume: Minimum daily volume (default 10k shares)

    Returns:
        List of FastResult objects.
    """
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(f"\n--- Scalp Cycle #{cycle_number} start: {now} ---")
    print(f"    Timeframe: {timeframe} | Symbols: {len(symbols)}")

    try:
        results = fast_pipeline.run_fast(
            symbols,
            timeframe=timeframe,
            execute=execute,
            validate_symbols=validate_symbols,
            min_price=min_price,
            min_volume=min_volume,
        )

        # Print results
        print(FastPipeline.format_results(results))

        # Track executed trades in scalp manager
        if scalp_manager and execute:
            for r in results:
                if r.executed and r.signal in ("BUY", "SELL"):
                    side = "long" if r.signal == "BUY" else "short"
                    scalp_manager.add_position(
                        symbol=r.symbol,
                        side=side,
                        entry_price=r.price,
                        qty=r.shares,
                        stop_loss=r.stop_loss,
                        take_profit=r.take_profit,
                    )

            # Show open positions
            if scalp_manager.get_all_positions():
                print(scalp_manager.format_positions())

        # Stats
        buys = sum(1 for r in results if r.signal == "BUY")
        sells = sum(1 for r in results if r.signal == "SELL")
        executed = sum(1 for r in results if r.executed)
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

        print(
            f"--- Scalp Cycle #{cycle_number} complete: "
            f"{buys} BUY / {sells} SELL / {executed} executed / "
            f"avg {avg_latency:.0f}ms ---\n"
        )

        return results

    except Exception as e:
        logger.error("Scalp cycle %d failed: %s", cycle_number, e)
        print(f"--- Scalp Cycle #{cycle_number} FAILED: {e} ---\n")
        return []


def run_premarket_scan(
    fast_pipeline: FastPipeline,
    symbols: list[str],
    finviz_screen: str | None = None,
    max_trades: int = 5,
) -> list[QueuedTrade]:
    """Run pre-market scan and queue trades for market open.

    Args:
        fast_pipeline: FastPipeline instance
        symbols: Stock tickers to analyze (or use Finviz)
        finviz_screen: Optional Finviz screen to get symbols
        max_trades: Maximum trades to queue

    Returns:
        List of QueuedTrade objects.
    """
    print(f"\n{'=' * 62}")
    print(f"  PRE-MARKET SCAN")
    print(f"{'=' * 62}")

    try:
        # Get symbols from Finviz if specified
        if finviz_screen:
            finviz = FinvizScanner(min_price=5.0, max_price=500.0)
            if finviz_screen == "BUY_CANDIDATES":
                stocks = finviz.get_buy_candidates(limit=20)
            else:
                stocks = finviz.get_screen(finviz_screen, limit=20)

            if stocks:
                symbols = [s.symbol for s in stocks]
                print(f"  Finviz {finviz_screen}: {len(symbols)} stocks")
                print(FinvizScanner.format_results(stocks))

        if not symbols:
            print("  No symbols to scan")
            return []

        print(f"  Scanning {len(symbols)} symbols...")

        # Run pre-market scan
        queue = fast_pipeline.premarket_scan(symbols, timeframe="1Day")

        # Limit queue size
        queue = queue[:max_trades]

        if queue:
            print(FastPipeline.format_queue(queue))
            print(f"  {len(queue)} trades queued for market open")
        else:
            print("  No actionable signals found")

        print(f"{'=' * 62}\n")
        return queue

    except Exception as e:
        logger.error("Pre-market scan failed: %s", e)
        print(f"  ERROR: {e}")
        return []


def execute_queue_at_market_open(
    fast_pipeline: FastPipeline,
    queue: list[QueuedTrade],
    client: AlpacaClient,
    no_wait: bool = False,
) -> list[QueuedTrade]:
    """Wait for market open and execute queued trades immediately.

    Args:
        fast_pipeline: FastPipeline instance
        queue: Trades to execute
        client: Alpaca client
        no_wait: Exit if market is closed

    Returns:
        List of executed trades.
    """
    if not queue:
        print("No trades in queue")
        return []

    # Check if market is already open
    clock = client.api.get_clock()
    if clock.is_open:
        print("Market is OPEN - executing queued trades immediately!")
        return fast_pipeline.execute_queue_at_open(queue)

    if no_wait:
        print("Market is closed and --no-wait is set. Queue saved for later.")
        return []

    # Wait for market open
    next_open = clock.next_open
    now = clock.timestamp
    wait_seconds = (next_open - now).total_seconds()

    if wait_seconds > 0:
        hours = int(wait_seconds // 3600)
        minutes = int((wait_seconds % 3600) // 60)
        print(f"\n  Market opens in {hours}h {minutes}m")
        print(f"  {len(queue)} trades queued for immediate execution at open")
        print(f"  Waiting...")

    # Use the pipeline's wait and execute
    return fast_pipeline.wait_and_execute_at_open(queue)


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


def minutes_until_close(client: AlpacaClient) -> float:
    """Get minutes until market close."""
    clock = client.api.get_clock()
    if not clock.is_open:
        return 0
    now = clock.timestamp
    close_time = clock.next_close
    return (close_time - now).total_seconds() / 60


def close_all_positions(client: AlpacaClient, dry_run: bool = False) -> list[dict]:
    """Close all open positions before end of day.

    Returns:
        List of closed position results.
    """
    positions = client.get_positions()
    results = []

    if not positions:
        logger.info("No positions to close.")
        return results

    print(f"\n{'=' * 60}")
    print(f"  END OF DAY - CLOSING ALL POSITIONS")
    print(f"{'=' * 60}")

    for pos in positions:
        symbol = pos["symbol"]
        qty = pos["qty"]
        pnl = pos.get("unrealized_pl", 0)

        result = {
            "symbol": symbol,
            "qty": qty,
            "pnl": pnl,
            "status": "pending",
        }

        if dry_run:
            result["status"] = "dry_run"
            logger.info("[DRY RUN] Would close %s: %d shares (P&L: $%.2f)", symbol, qty, pnl)
        else:
            try:
                order = client.api.close_position(symbol)
                result["status"] = "closed"
                result["order_id"] = order.id
                logger.info("Closed %s: %d shares (P&L: $%.2f) - Order: %s",
                           symbol, qty, pnl, order.id)
            except Exception as e:
                result["status"] = "failed"
                result["error"] = str(e)
                logger.error("Failed to close %s: %s", symbol, e)

        results.append(result)
        print(f"  {symbol:8s} {qty:>6d} shares  P&L: ${pnl:+,.2f}  [{result['status']}]")

    total_pnl = sum(r["pnl"] for r in results)
    print(f"{'─' * 60}")
    print(f"  Total P&L: ${total_pnl:+,.2f}")
    print(f"{'=' * 60}\n")

    return results


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


# ── Daemon helpers ───────────────────────────────────────────────

def daemonize():
    """Fork process to run as daemon (Unix only)."""
    if sys.platform == "win32":
        logger.error("Daemon mode not supported on Windows")
        sys.exit(1)

    # First fork
    try:
        pid = os.fork()
        if pid > 0:
            print(f"Scheduler started as daemon (PID: {pid})")
            sys.exit(0)
    except OSError as e:
        logger.error("Fork #1 failed: %s", e)
        sys.exit(1)

    # Decouple from parent
    os.setsid()
    os.umask(0)

    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        logger.error("Fork #2 failed: %s", e)
        sys.exit(1)

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()

    # Write PID file
    PID_FILE.write_text(str(os.getpid()))
    logger.info("Daemon started with PID %d", os.getpid())


def check_daemon_status() -> tuple[bool, int | None]:
    """Check if daemon is running. Returns (running, pid)."""
    if not PID_FILE.exists():
        return False, None

    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, 0)  # Check if process exists
        return True, pid
    except OSError:
        # Stale PID file
        PID_FILE.unlink()
        return False, None


def stop_daemon() -> bool:
    """Stop running daemon."""
    running, pid = check_daemon_status()
    if not running:
        print("Scheduler is not running")
        return False

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent stop signal to scheduler (PID: {pid})")
        # Wait for process to stop
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except OSError:
                PID_FILE.unlink() if PID_FILE.exists() else None
                print("Scheduler stopped")
                return True
        print("Warning: Scheduler may still be running")
        return True
    except OSError as e:
        print(f"Error stopping scheduler: {e}")
        return False


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
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as background daemon process",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check if scheduler daemon is running",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop running scheduler daemon",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Use scanner mode to find trading opportunities",
    )
    parser.add_argument(
        "--universe",
        default="VOLATILE",
        choices=["DEFAULT", "TECH", "VOLATILE", "SP500_TOP", "DIVIDEND", "ALL"],
        help="Stock universe for scan mode (default: VOLATILE)",
    )
    parser.add_argument(
        "--close-eod",
        action="store_true",
        default=True,
        help="Close all positions before end of day (default: True)",
    )
    parser.add_argument(
        "--no-close-eod",
        action="store_true",
        help="Don't close positions at end of day (hold overnight)",
    )
    parser.add_argument(
        "--eod-minutes",
        type=int,
        default=15,
        help="Minutes before market close to close all positions (default: 15)",
    )
    parser.add_argument(
        "--finviz",
        action="store_true",
        help="Use Finviz screener to find trading opportunities",
    )
    parser.add_argument(
        "--finviz-screen",
        default="OVERSOLD",
        choices=[
            "TOP_GAINERS", "TOP_LOSERS", "UNUSUAL_VOLUME", "NEW_HIGH", "NEW_LOW",
            "OVERSOLD", "OVERBOUGHT", "BREAKOUT", "SMA_CROSS_UP", "SMA_CROSS_DOWN",
            "GOLDEN_CROSS", "DEATH_CROSS", "UPGRADES", "DOWNGRADES", "BUY_CANDIDATES"
        ],
        help="Finviz screen to use (default: OVERSOLD)",
    )
    parser.add_argument(
        "--check-news",
        action="store_true",
        help="Analyze news for each stock (identify catalysts)",
    )
    parser.add_argument(
        "--require-news",
        action="store_true",
        help="Only trade stocks with recent news/catalyst (implies --check-news)",
    )
    # Scalping / HFT mode
    parser.add_argument(
        "--scalp",
        action="store_true",
        help="Enable fast scalping mode (parallel processing, 1Min bars, tighter stops)",
    )
    parser.add_argument(
        "--scalp-interval",
        type=int,
        default=30,
        help="Seconds between scalp cycles (default: 30)",
    )
    parser.add_argument(
        "--scalp-workers",
        type=int,
        default=12,
        help="Parallel worker threads for scalping (default: 12)",
    )
    # Pre-market scan mode
    parser.add_argument(
        "--premarket",
        action="store_true",
        help="Run pre-market scan and queue trades for market open",
    )
    parser.add_argument(
        "--execute-at-open",
        action="store_true",
        help="Execute queued trades immediately when market opens (use with --premarket)",
    )
    parser.add_argument(
        "--max-queued-trades",
        type=int,
        default=5,
        help="Maximum trades to queue for market open (default: 5)",
    )
    # Symbol validation options
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip symbol validation (faster but may cause failed trades)",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=1.0,
        help="Minimum stock price filter (default: $1.00)",
    )
    parser.add_argument(
        "--min-volume",
        type=int,
        default=10000,
        help="Minimum daily volume filter (default: 10,000 shares)",
    )
    args = parser.parse_args()

    # Handle daemon management commands first
    if args.status:
        running, pid = check_daemon_status()
        if running:
            print(f"Scheduler is running (PID: {pid})")
        else:
            print("Scheduler is not running")
        return

    if args.stop:
        stop_daemon()
        return

    # Start as daemon if requested
    if args.daemon:
        running, pid = check_daemon_status()
        if running:
            print(f"Scheduler already running (PID: {pid})")
            sys.exit(1)
        daemonize()

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
    close_eod = args.close_eod and not args.no_close_eod
    eod_minutes = args.eod_minutes

    # Override timeframe for scalping if not explicitly set
    if args.scalp and args.timeframe == "1Day":
        args.timeframe = "1Min"  # Default to 1Min for scalping

    # Handle --require-news implying --check-news
    check_news = args.check_news or args.require_news
    require_news = args.require_news

    # Determine scan mode for banner
    if args.premarket:
        scan_mode = "PRE-MARKET SCAN"
        if args.finviz:
            scan_mode += f" + FINVIZ ({args.finviz_screen})"
    elif args.scalp:
        scan_mode = "SCALPING/HFT"
        if args.finviz:
            scan_mode += f" + FINVIZ ({args.finviz_screen})"
    elif args.finviz:
        scan_mode = f"FINVIZ ({args.finviz_screen})"
    elif args.scan:
        scan_mode = f"SCANNER ({args.universe})"
    else:
        scan_mode = "FIXED SYMBOLS"

    # Adjust interval for scalping
    if args.scalp:
        interval = args.scalp_interval

    print(f"\n{'=' * 62}")
    print(f"  AUTOMATED TRADING SCHEDULER")
    print(f"{'=' * 62}")
    print(f"  Mode         : {mode}")
    print(f"  Scan Mode    : {scan_mode}")
    if args.scalp:
        print(f"  Scalp Workers: {args.scalp_workers} threads")
        print(f"  Interval     : {interval} sec")
        validate_str = "OFF" if args.no_validate else f"ON (>=${args.min_price:.2f}, vol>{args.min_volume:,})"
        print(f"  Validation   : {validate_str}")
    else:
        print(f"  Interval     : {interval} min")
    if not args.finviz and not args.scan and not args.premarket:
        print(f"  Symbols      : {', '.join(symbols)}")
    print(f"  Timeframe    : {args.timeframe}")
    if check_news:
        news_mode = "REQUIRED" if require_news else "ON"
        print(f"  News Check   : {news_mode}")
    print(f"  Close EOD    : {'ON' if close_eod else 'OFF'} ({eod_minutes} min before close)")
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

    # ── Build fast pipeline for scalping/premarket ─────────
    fast_pipeline = None
    scalp_manager = None
    if args.scalp or args.premarket:
        fast_pipeline = FastPipeline(
            client=client,
            dry_run=dry_run,
            max_workers=args.scalp_workers,
            atr_multiplier_sl=1.0,  # Tighter stop loss
            atr_multiplier_tp=1.5,  # Smaller profit target
            max_position_pct=0.02,  # Smaller positions
            min_score=0.1,
            # Scalping-specific
            use_scalp_indicators=True,
            profit_target_pct=0.5,  # 0.5% profit target
            stop_loss_pct=0.25,  # 0.25% stop loss
        )
        # Create scalping position manager
        scalp_manager = ScalpingManager(
            client=client,
            check_interval=1.0,  # Check every 1 second
            max_hold_minutes=30,  # Max 30 min hold time
            trailing_lock_pct=0.3,  # Lock 30% of gains
            dry_run=dry_run,
        )
        scalp_manager.start_monitoring()
        print("  Scalp Manager: Position monitoring active")

    # ── Pre-market scan mode ─────────────────────────────────
    if args.premarket:
        # Get symbols to scan
        scan_symbols = symbols if symbols else []
        finviz_screen = args.finviz_screen if args.finviz else None

        queue = run_premarket_scan(
            fast_pipeline,
            scan_symbols,
            finviz_screen=finviz_screen,
            max_trades=args.max_queued_trades,
        )

        if args.execute_at_open and queue:
            executed = execute_queue_at_market_open(
                fast_pipeline, queue, client, no_wait=no_wait
            )
            print(f"\nExecuted {len([t for t in executed if t.executed])} trades at market open")

        return

    # ── Single-pass mode ─────────────────────────────────────
    if args.once:
        if args.scalp:
            # Get symbols from Finviz if specified
            scan_symbols = symbols
            if args.finviz:
                finviz = FinvizScanner(min_price=5.0, max_price=500.0)
                stocks = finviz.get_screen(args.finviz_screen, limit=20)
                if stocks:
                    scan_symbols = [s.symbol for s in stocks]
            run_scalp_cycle(
                fast_pipeline, scan_symbols, args.timeframe,
                cycle_number=1, execute=not dry_run,
                scalp_manager=scalp_manager,
                validate_symbols=not args.no_validate,
                min_price=args.min_price,
                min_volume=args.min_volume,
            )
        elif args.finviz:
            run_finviz_cycle(
                pipeline, args.finviz_screen, args.timeframe,
                cycle_number=1, check_news=check_news, require_news=require_news
            )
        elif args.scan:
            run_scan_cycle(client, args.universe, dry_run, cycle_number=1)
        else:
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

        if args.scalp:
            # Get symbols from Finviz if specified
            scan_symbols = symbols
            if args.finviz:
                try:
                    finviz = FinvizScanner(min_price=5.0, max_price=500.0)
                    stocks = finviz.get_screen(args.finviz_screen, limit=20)
                    if stocks:
                        scan_symbols = [s.symbol for s in stocks]
                except Exception as e:
                    logger.warning("Finviz scan failed, using default symbols: %s", e)
            run_scalp_cycle(
                fast_pipeline, scan_symbols, args.timeframe,
                cycle_number=cycle, execute=not dry_run,
                scalp_manager=scalp_manager,
                validate_symbols=not args.no_validate,
                min_price=args.min_price,
                min_volume=args.min_volume,
            )

            # Show scalp stats periodically
            if cycle % 10 == 0 and scalp_manager:
                print(scalp_manager.format_stats())

        elif args.finviz:
            run_finviz_cycle(
                pipeline, args.finviz_screen, args.timeframe,
                cycle_number=cycle, check_news=check_news, require_news=require_news
            )
        elif args.scan:
            run_scan_cycle(client, args.universe, dry_run, cycle_number=cycle)
        else:
            results = run_cycle(pipeline, symbols, args.timeframe, cycle_number=cycle)
            if bus and args.show_events:
                print(EventBus.format_history(bus.get_history(limit=30)))

        if _shutdown:
            break

        # Check if we need to close all positions before end of day
        if close_eod and client.is_market_open():
            mins_to_close = minutes_until_close(client)
            if mins_to_close <= eod_minutes and mins_to_close > 0:
                logger.info("Market closes in %.1f minutes - closing all positions", mins_to_close)
                close_all_positions(client, dry_run=dry_run)
                logger.info("End of day position close complete. Waiting for market close.")
                # Wait for market to actually close
                while client.is_market_open() and not _shutdown:
                    time.sleep(30)
                if no_wait:
                    break
                continue

        # Check if market is still open before sleeping
        if not client.is_market_open():
            logger.info("Market has closed. Cycle complete for today.")
            if no_wait:
                break
            continue  # will wait_for_market_open on next iteration

        # Scalp mode uses seconds, regular mode uses minutes
        if args.scalp:
            sleep_seconds = interval  # Already in seconds for scalping
            logger.info("Sleeping %d seconds until next scalp cycle...", interval)
        else:
            sleep_seconds = interval * 60
            logger.info("Sleeping %d minutes until next cycle...", interval)

        while sleep_seconds > 0 and not _shutdown:
            time.sleep(min(sleep_seconds, 10))
            sleep_seconds -= 10

    # Cleanup scalp manager
    if scalp_manager:
        print("\nClosing scalp positions...")
        scalp_manager.exit_all(reason="shutdown")
        scalp_manager.stop_monitoring()
        print(scalp_manager.format_stats())

    print(f"\nScheduler stopped after {cycle} cycle(s).")

    # Cleanup PID file on exit
    if PID_FILE.exists():
        PID_FILE.unlink()


if __name__ == "__main__":
    main()
