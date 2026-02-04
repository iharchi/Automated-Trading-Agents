"""Market-Hours Scheduler

Runs the Portfolio Manager pipeline on a configurable interval during
market hours.  Sleeps between cycles and stops when the market closes.

Usage:
    python scheduler.py                             # defaults: 15 min interval
    python scheduler.py AAPL TSLA --interval 30     # 30 min interval
    python scheduler.py --once                      # single pass then exit

The scheduler:
    1. Checks if the market is open (via Alpaca clock API).
    2. If open, runs the full pipeline for each symbol.
    3. Logs all signals, decisions, and orders to the trade journal.
    4. Sleeps for --interval minutes and repeats.
    5. If the market is closed, calculates time until next open and
       waits (or exits if --no-wait is set).
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone

from config.settings import Settings
from agents.portfolio_manager_agent import PortfolioManagerAgent
from utils.alpaca_client import AlpacaClient
from utils.trade_journal import TradeJournal

logger = logging.getLogger("scheduler")

# Graceful shutdown
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received — finishing current cycle...")
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def run_cycle(
    pm: PortfolioManagerAgent,
    symbols: list[str],
    journal: TradeJournal,
    auto_trade: bool,
    timeframe: str,
) -> None:
    """Run one full analysis + optional trade cycle."""
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(f"\n--- Cycle start: {now} ---")

    for symbol in symbols:
        try:
            decision = pm.analyze(symbol, timeframe=timeframe)
            print(PortfolioManagerAgent.format_analysis(decision))

            # Log individual agent signals
            for sig in decision.get("agent_signals", []):
                s = sig if isinstance(sig, dict) else sig.__dict__
                journal.log_signal(
                    symbol=symbol,
                    agent=s.get("agent", ""),
                    signal=s.get("signal", "HOLD"),
                    score=s.get("normalised_score", 0),
                    price=decision.get("current_price", 0),
                )

            # Log the combined decision
            journal.log_decision(decision)

            # Execute if auto-trade is on
            if auto_trade:
                result = pm.execute(symbol, decision)
                order = result.get("order")
                if order and order not in (None, "skipped_no_position"):
                    print(f"  Order placed: {order}")
                    journal.log_order(
                        symbol=symbol,
                        side="buy" if decision.get("signal") == "BUY" else "sell",
                        qty=decision.get("position_size", 0),
                        order_result=order,
                        reason=f"signal={decision.get('signal')}",
                    )
                elif decision.get("signal") != "HOLD":
                    journal.log_order(
                        symbol=symbol,
                        side="buy" if decision.get("signal") == "BUY" else "sell",
                        qty=decision.get("position_size", 0),
                        order_result="skipped",
                        reason="auto_trade=off or risk_rejected",
                    )

        except Exception as e:
            logger.error("Error processing %s: %s", symbol, e)

    print(f"--- Cycle complete ---\n")


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


def main():
    parser = argparse.ArgumentParser(description="Market-Hours Scheduler")
    parser.add_argument("symbols", nargs="*", help="Ticker symbols")
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Minutes between analysis cycles (default: 15)",
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
        help="Enable paper-trade order execution",
    )
    parser.add_argument(
        "--timeframe",
        default="1Day",
        choices=["1Min", "5Min", "15Min", "1Hour", "1Day"],
        help="Bar timeframe (default: 1Day)",
    )
    parser.add_argument(
        "--ta-weight",
        type=float,
        default=0.65,
        help="Weight for TA signals (default: 0.65)",
    )
    parser.add_argument(
        "--sentiment-weight",
        type=float,
        default=0.35,
        help="Weight for Sentiment signals (default: 0.35)",
    )
    parser.add_argument(
        "--journal-dir",
        default=None,
        help="Directory for journal CSV files (default: ./utils/journal/)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, Settings.LOG_LEVEL, logging.INFO),
        format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    symbols = args.symbols or Settings.DEFAULT_SYMBOLS
    journal = TradeJournal(args.journal_dir) if args.journal_dir else TradeJournal()

    print(f"\nScheduler")
    print(f"Symbols      : {', '.join(symbols)}")
    print(f"Interval     : {args.interval} min")
    print(f"Auto-trade   : {'ON' if args.auto_trade else 'OFF'}")
    print(f"Journal      : {journal.journal_dir}")
    print(f"Mode         : {'single pass' if args.once else 'continuous'}\n")

    client = AlpacaClient()
    pm = PortfolioManagerAgent(
        client=client,
        ta_weight=args.ta_weight,
        sentiment_weight=args.sentiment_weight,
        auto_trade=args.auto_trade,
    )

    if args.once:
        run_cycle(pm, symbols, journal, args.auto_trade, args.timeframe)
        return

    # ── Continuous loop ──────────────────────────────────────
    while not _shutdown:
        if not wait_for_market_open(client, args.no_wait):
            break

        if _shutdown:
            break

        run_cycle(pm, symbols, journal, args.auto_trade, args.timeframe)

        if _shutdown:
            break

        # Check if market is still open before sleeping
        if not client.is_market_open():
            logger.info("Market has closed. Cycle complete for today.")
            if args.no_wait:
                break
            continue  # will wait_for_market_open on next iteration

        logger.info("Sleeping %d minutes until next cycle...", args.interval)
        sleep_seconds = args.interval * 60
        while sleep_seconds > 0 and not _shutdown:
            time.sleep(min(sleep_seconds, 10))
            sleep_seconds -= 10

    print("\nScheduler stopped.")


if __name__ == "__main__":
    main()
