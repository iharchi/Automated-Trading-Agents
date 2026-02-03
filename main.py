#!/usr/bin/env python3
"""Entry point for the Automated Trading Agents system.

Usage:
    python main.py                       # Analyse default symbols
    python main.py AAPL TSLA MSFT        # Analyse specific symbols
    python main.py --test                 # Run connection test only
    python main.py --auto-trade           # Enable paper-trade execution
"""

import argparse
import logging
import sys

from config.settings import Settings
from agents.technical_analysis_agent import TechnicalAnalysisAgent


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, Settings.LOG_LEVEL, logging.INFO),
        format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def run_connection_test():
    from tests.test_connection import test_connection

    return test_connection()


def main():
    parser = argparse.ArgumentParser(description="Automated Trading Agents")
    parser.add_argument("symbols", nargs="*", help="Ticker symbols to analyse")
    parser.add_argument("--test", action="store_true", help="Run connection test")
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
    args = parser.parse_args()

    setup_logging()

    if args.test:
        success = run_connection_test()
        sys.exit(0 if success else 1)

    symbols = args.symbols or Settings.DEFAULT_SYMBOLS

    print(f"\nTrading mode : {Settings.TRADING_MODE}")
    print(f"Symbols      : {', '.join(symbols)}")
    print(f"Timeframe    : {args.timeframe}")
    print(f"Auto-trade   : {'ON' if args.auto_trade else 'OFF'}\n")

    agent = TechnicalAnalysisAgent(auto_trade=args.auto_trade)

    for symbol in symbols:
        try:
            analysis = agent.analyze(symbol, timeframe=args.timeframe)
            print(TechnicalAnalysisAgent.format_analysis(analysis))

            if args.auto_trade:
                result = agent.execute(symbol, analysis)
                if result.get("order") and result["order"] not in (
                    None,
                    "skipped_no_position",
                ):
                    print(f"  Order placed: {result['order']}")
        except Exception as e:
            logging.getLogger("main").error("Error analysing %s: %s", symbol, e)


if __name__ == "__main__":
    main()
