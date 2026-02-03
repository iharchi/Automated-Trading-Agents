#!/usr/bin/env python3
"""Entry point for the Automated Trading Agents system.

Usage:
    python main.py                         # Run full pipeline on default symbols
    python main.py AAPL TSLA MSFT          # Analyse specific symbols
    python main.py --agent ta              # Technical Analysis only
    python main.py --agent sentiment       # Sentiment Analysis only
    python main.py --agent risk            # Risk Management only (needs --price/--atr)
    python main.py --test                  # Run connection test only
    python main.py --auto-trade            # Enable paper-trade execution
"""

import argparse
import logging
import sys

from config.settings import Settings
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.sentiment_analysis_agent import SentimentAnalysisAgent
from agents.risk_management_agent import RiskManagementAgent

# Signal-generating agents (run independently per symbol)
SIGNAL_AGENTS = {
    "ta": {
        "class": TechnicalAnalysisAgent,
        "label": "Technical Analysis",
    },
    "sentiment": {
        "class": SentimentAnalysisAgent,
        "label": "Sentiment Analysis",
    },
}

ALL_AGENT_KEYS = list(SIGNAL_AGENTS.keys()) + ["risk"]


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, Settings.LOG_LEVEL, logging.INFO),
        format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def run_connection_test():
    from tests.test_connection import test_connection

    return test_connection()


def run_pipeline(
    symbol: str,
    signal_agents: dict,
    risk_agent: RiskManagementAgent | None,
    auto_trade: bool,
    **kwargs,
) -> None:
    """Run signal agents, then pass results through risk management."""
    ta_analysis = None

    # 1. Run each signal agent
    for key, agent in signal_agents.items():
        try:
            analysis = agent.analyze(symbol, **kwargs)
            print(agent.format_analysis(analysis))

            # Keep TA results for the risk agent
            if key == "ta":
                ta_analysis = analysis
        except Exception as e:
            logging.getLogger("main").error(
                "Error running %s on %s: %s",
                SIGNAL_AGENTS[key]["label"], symbol, e,
            )

    # 2. Run risk assessment if we have TA data
    if risk_agent and ta_analysis:
        signal = ta_analysis.get("signal", "HOLD")
        if signal != "HOLD":
            proposed_side = "buy" if signal == "BUY" else "sell"
            try:
                risk_result = risk_agent.analyze(
                    symbol,
                    proposed_side=proposed_side,
                    price=ta_analysis.get("current_price", 0),
                    atr=ta_analysis.get("atr", 0),
                )
                print(RiskManagementAgent.format_analysis(risk_result))

                if auto_trade and risk_result.get("approved"):
                    exec_result = risk_agent.execute(symbol, risk_result)
                    order = exec_result.get("order")
                    if order and order not in (None, "skipped_no_position"):
                        print(f"  Order placed: {order}")
            except Exception as e:
                logging.getLogger("main").error(
                    "Error running Risk Management on %s: %s", symbol, e,
                )
        else:
            print(f"  [{symbol}] Signal is HOLD — skipping risk assessment.\n")
    elif risk_agent and not ta_analysis:
        logging.getLogger("main").warning(
            "Risk agent enabled but no TA data for %s — skipping risk check.",
            symbol,
        )


def main():
    parser = argparse.ArgumentParser(description="Automated Trading Agents")
    parser.add_argument("symbols", nargs="*", help="Ticker symbols to analyse")
    parser.add_argument("--test", action="store_true", help="Run connection test")
    parser.add_argument(
        "--agent",
        choices=ALL_AGENT_KEYS,
        default=None,
        help="Run a specific agent (default: full pipeline)",
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

    # Determine which agents to run
    if args.agent and args.agent != "risk":
        selected_signals = {args.agent: SIGNAL_AGENTS[args.agent]}
        risk_agent = None
    elif args.agent == "risk":
        # Risk alone still needs TA to provide price/ATR
        selected_signals = {"ta": SIGNAL_AGENTS["ta"]}
        risk_agent = RiskManagementAgent(auto_trade=args.auto_trade)
    else:
        # Full pipeline: all signal agents + risk
        selected_signals = dict(SIGNAL_AGENTS)
        risk_agent = RiskManagementAgent(auto_trade=args.auto_trade)

    agent_labels = [v["label"] for v in selected_signals.values()]
    if risk_agent:
        agent_labels.append("Risk Management")

    print(f"\nTrading mode : {Settings.TRADING_MODE}")
    print(f"Agents       : {', '.join(agent_labels)}")
    print(f"Symbols      : {', '.join(symbols)}")
    print(f"Timeframe    : {args.timeframe}")
    print(f"Auto-trade   : {'ON' if args.auto_trade else 'OFF'}\n")

    # Instantiate signal agents
    agents = {
        key: info["class"](auto_trade=False)  # signal agents don't trade directly
        for key, info in selected_signals.items()
    }

    for symbol in symbols:
        run_pipeline(
            symbol,
            signal_agents=agents,
            risk_agent=risk_agent,
            auto_trade=args.auto_trade,
            timeframe=args.timeframe,
        )


if __name__ == "__main__":
    main()
