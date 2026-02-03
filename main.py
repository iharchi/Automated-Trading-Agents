#!/usr/bin/env python3
"""Entry point for the Automated Trading Agents system.

Usage:
    python main.py                         # Full pipeline (TA + Sentiment + Risk)
    python main.py AAPL TSLA MSFT          # Analyse specific symbols
    python main.py --agent ta              # Technical Analysis only
    python main.py --agent sentiment       # Sentiment Analysis only
    python main.py --agent risk            # TA + Risk (no sentiment)
    python main.py --test                  # Run connection test only
    python main.py --auto-trade            # Enable paper-trade execution
    python main.py --ta-weight 0.7 --sentiment-weight 0.3   # Custom weights
"""

import argparse
import logging
import sys

from config.settings import Settings
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.sentiment_analysis_agent import SentimentAnalysisAgent
from agents.risk_management_agent import RiskManagementAgent
from agents.portfolio_manager_agent import PortfolioManagerAgent

# Individual agents for standalone mode
STANDALONE_AGENTS = {
    "ta": {
        "class": TechnicalAnalysisAgent,
        "label": "Technical Analysis",
    },
    "sentiment": {
        "class": SentimentAnalysisAgent,
        "label": "Sentiment Analysis",
    },
    "risk": {
        "class": None,  # risk requires TA data, handled specially
        "label": "Risk Management",
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


def run_standalone(agent, symbol: str, **kwargs) -> None:
    """Run a single standalone agent and print its output."""
    analysis = agent.analyze(symbol, **kwargs)
    print(agent.format_analysis(analysis))


def run_standalone_risk(symbol: str, auto_trade: bool, **kwargs) -> None:
    """Run TA + Risk without the full portfolio manager."""
    ta_agent = TechnicalAnalysisAgent(auto_trade=False)
    risk_agent = RiskManagementAgent(auto_trade=auto_trade)

    ta_result = ta_agent.analyze(symbol, **kwargs)
    print(TechnicalAnalysisAgent.format_analysis(ta_result))

    signal = ta_result.get("signal", "HOLD")
    if signal == "HOLD":
        print(f"  [{symbol}] Signal is HOLD — skipping risk assessment.\n")
        return

    proposed_side = "buy" if signal == "BUY" else "sell"
    risk_result = risk_agent.analyze(
        symbol,
        proposed_side=proposed_side,
        price=ta_result.get("current_price", 0),
        atr=ta_result.get("atr", 0),
    )
    print(RiskManagementAgent.format_analysis(risk_result))

    if auto_trade and risk_result.get("approved"):
        exec_result = risk_agent.execute(symbol, risk_result)
        order = exec_result.get("order")
        if order and order not in (None, "skipped_no_position"):
            print(f"  Order placed: {order}")


def main():
    parser = argparse.ArgumentParser(description="Automated Trading Agents")
    parser.add_argument("symbols", nargs="*", help="Ticker symbols to analyse")
    parser.add_argument("--test", action="store_true", help="Run connection test")
    parser.add_argument(
        "--agent",
        choices=list(STANDALONE_AGENTS.keys()),
        default=None,
        help="Run a single agent instead of the full pipeline",
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
    parser.add_argument(
        "--ta-weight",
        type=float,
        default=0.65,
        help="Weight for Technical Analysis signals (default: 0.65)",
    )
    parser.add_argument(
        "--sentiment-weight",
        type=float,
        default=0.35,
        help="Weight for Sentiment Analysis signals (default: 0.35)",
    )
    args = parser.parse_args()

    setup_logging()

    if args.test:
        success = run_connection_test()
        sys.exit(0 if success else 1)

    symbols = args.symbols or Settings.DEFAULT_SYMBOLS

    # ── Standalone agent mode ────────────────────────────────
    if args.agent:
        label = STANDALONE_AGENTS[args.agent]["label"]
        print(f"\nTrading mode : {Settings.TRADING_MODE}")
        print(f"Agent        : {label} (standalone)")
        print(f"Symbols      : {', '.join(symbols)}")
        print(f"Timeframe    : {args.timeframe}")
        print(f"Auto-trade   : {'ON' if args.auto_trade else 'OFF'}\n")

        for symbol in symbols:
            try:
                if args.agent == "risk":
                    run_standalone_risk(
                        symbol,
                        auto_trade=args.auto_trade,
                        timeframe=args.timeframe,
                    )
                else:
                    agent_cls = STANDALONE_AGENTS[args.agent]["class"]
                    agent = agent_cls(auto_trade=args.auto_trade)
                    run_standalone(agent, symbol, timeframe=args.timeframe)
            except Exception as e:
                logging.getLogger("main").error(
                    "Error running %s on %s: %s", label, symbol, e,
                )
        return

    # ── Full pipeline via Portfolio Manager ──────────────────
    print(f"\nTrading mode : {Settings.TRADING_MODE}")
    print(f"Pipeline     : Portfolio Manager (TA + Sentiment + Risk)")
    print(f"Weights      : TA={args.ta_weight:.0%}  Sentiment={args.sentiment_weight:.0%}")
    print(f"Symbols      : {', '.join(symbols)}")
    print(f"Timeframe    : {args.timeframe}")
    print(f"Auto-trade   : {'ON' if args.auto_trade else 'OFF'}\n")

    pm = PortfolioManagerAgent(
        ta_weight=args.ta_weight,
        sentiment_weight=args.sentiment_weight,
        auto_trade=args.auto_trade,
    )

    for symbol in symbols:
        try:
            decision = pm.analyze(symbol, timeframe=args.timeframe)
            print(PortfolioManagerAgent.format_analysis(decision))

            if args.auto_trade:
                result = pm.execute(symbol, decision)
                order = result.get("order")
                if order and order not in (None, "skipped_no_position"):
                    print(f"  Order placed: {order}")
        except Exception as e:
            logging.getLogger("main").error(
                "Error in portfolio pipeline for %s: %s", symbol, e,
            )


if __name__ == "__main__":
    main()
