#!/usr/bin/env python3
"""Entry point for the Automated Trading Agents system.

Usage:
    python main.py                         # Run all agents on default symbols
    python main.py AAPL TSLA MSFT          # Analyse specific symbols
    python main.py --agent ta              # Technical Analysis only
    python main.py --agent sentiment       # Sentiment Analysis only
    python main.py --test                  # Run connection test only
    python main.py --auto-trade            # Enable paper-trade execution
"""

import argparse
import logging
import sys

from config.settings import Settings
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.sentiment_analysis_agent import SentimentAnalysisAgent

AGENTS = {
    "ta": {
        "class": TechnicalAnalysisAgent,
        "label": "Technical Analysis",
    },
    "sentiment": {
        "class": SentimentAnalysisAgent,
        "label": "Sentiment Analysis",
    },
}


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, Settings.LOG_LEVEL, logging.INFO),
        format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def run_connection_test():
    from tests.test_connection import test_connection

    return test_connection()


def run_agent(agent, symbol: str, auto_trade: bool, **kwargs) -> None:
    """Run a single agent on a symbol and print results."""
    analysis = agent.analyze(symbol, **kwargs)
    print(agent.format_analysis(analysis))

    if auto_trade:
        result = agent.execute(symbol, analysis, **kwargs)
        order = result.get("order")
        if order and order not in (None, "skipped_no_position"):
            print(f"  Order placed: {order}")


def main():
    parser = argparse.ArgumentParser(description="Automated Trading Agents")
    parser.add_argument("symbols", nargs="*", help="Ticker symbols to analyse")
    parser.add_argument("--test", action="store_true", help="Run connection test")
    parser.add_argument(
        "--agent",
        choices=list(AGENTS.keys()),
        default=None,
        help="Run a specific agent (default: all agents)",
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
        help="Bar timeframe for Technical Analysis (default: 1Day)",
    )
    args = parser.parse_args()

    setup_logging()

    if args.test:
        success = run_connection_test()
        sys.exit(0 if success else 1)

    symbols = args.symbols or Settings.DEFAULT_SYMBOLS

    # Decide which agents to run
    if args.agent:
        selected = {args.agent: AGENTS[args.agent]}
    else:
        selected = AGENTS

    agent_names = ", ".join(v["label"] for v in selected.values())
    print(f"\nTrading mode : {Settings.TRADING_MODE}")
    print(f"Agents       : {agent_names}")
    print(f"Symbols      : {', '.join(symbols)}")
    print(f"Timeframe    : {args.timeframe}")
    print(f"Auto-trade   : {'ON' if args.auto_trade else 'OFF'}\n")

    # Instantiate selected agents
    agents = {
        key: info["class"](auto_trade=args.auto_trade)
        for key, info in selected.items()
    }

    for symbol in symbols:
        for key, agent in agents.items():
            try:
                run_agent(
                    agent,
                    symbol,
                    auto_trade=args.auto_trade,
                    timeframe=args.timeframe,
                )
            except Exception as e:
                logging.getLogger("main").error(
                    "Error running %s on %s: %s", AGENTS[key]["label"], symbol, e
                )


if __name__ == "__main__":
    main()
