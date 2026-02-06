#!/usr/bin/env python3
"""Portfolio Guardian CLI

Runs the Portfolio Guardian agent to actively monitor and protect the portfolio.

Features:
    - Take-profit enforcement (default 10%)
    - Trailing stops (default 5% from peak)
    - Stop-loss enforcement
    - Daily loss limit monitoring
    - Maximum drawdown protection
    - Sector balance monitoring

Usage:
    # Monitor mode (no auto-close)
    python3 guardian.py --monitor

    # Active mode (auto-close positions hitting targets)
    python3 guardian.py --active

    # With custom settings
    python3 guardian.py --active --take-profit 0.10 --trailing-stop 0.05 --interval 30

    # Single check (no loop)
    python3 guardian.py --once
"""

import argparse
import logging
import sys

from utils.alpaca_client import AlpacaClient
from agents.portfolio_guardian_agent import PortfolioGuardianAgent
from utils.profit_monitor import ProfitMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Guardian - Active portfolio protection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor only mode (no auto-close)",
    )
    mode_group.add_argument(
        "--active",
        action="store_true",
        help="Active mode (auto-close positions hitting targets)",
    )

    # Settings
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.10,
        help="Take profit percentage (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--trailing-stop",
        type=float,
        default=0.05,
        help="Trailing stop percentage from peak (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.05,
        help="Stop loss percentage (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--daily-loss-limit",
        type=float,
        default=0.03,
        help="Daily loss limit percentage (default: 0.03 = 3%%)",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.10,
        help="Maximum drawdown percentage (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--max-sector",
        type=float,
        default=0.35,
        help="Maximum sector allocation (default: 0.35 = 35%%)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Seconds between checks (default: 30)",
    )

    # Other options
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (no continuous loop)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (no actual trades)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine auto_trade mode
    auto_trade = args.active and not args.dry_run

    print("\n" + "=" * 70)
    print("  PORTFOLIO GUARDIAN")
    print("=" * 70)
    print(f"  Mode: {'ACTIVE' if auto_trade else 'MONITOR ONLY'}")
    print(f"  Dry Run: {args.dry_run}")
    print(f"  Settings:")
    print(f"    Take Profit:     {args.take_profit:.0%}")
    print(f"    Trailing Stop:   {args.trailing_stop:.0%}")
    print(f"    Stop Loss:       {args.stop_loss:.0%}")
    print(f"    Daily Loss Limit:{args.daily_loss_limit:.0%}")
    print(f"    Max Drawdown:    {args.max_drawdown:.0%}")
    print(f"    Max Sector:      {args.max_sector:.0%}")
    if not args.once:
        print(f"    Check Interval:  {args.interval}s")
    print("=" * 70 + "\n")

    # Initialize
    client = AlpacaClient()
    guardian = PortfolioGuardianAgent(
        client=client,
        take_profit_pct=args.take_profit,
        trailing_stop_pct=args.trailing_stop,
        stop_loss_pct=args.stop_loss,
        daily_loss_limit_pct=args.daily_loss_limit,
        max_drawdown_pct=args.max_drawdown,
        max_sector_pct=args.max_sector,
        auto_trade=auto_trade,
        dry_run=args.dry_run,
    )

    if args.once:
        # Single check
        analysis = guardian.analyze()
        results = guardian.execute(analysis=analysis)
        print(PortfolioGuardianAgent.format_analysis(analysis))

        if results.get("actions_taken"):
            print("\nActions taken:")
            for action in results["actions_taken"]:
                print(f"  {action['symbol']}: {action['reason']} - {action['status']}")
    else:
        # Continuous mode
        try:
            guardian.run_continuous(interval=args.interval)
        except KeyboardInterrupt:
            print("\nGuardian stopped by user")
            sys.exit(0)


if __name__ == "__main__":
    main()
