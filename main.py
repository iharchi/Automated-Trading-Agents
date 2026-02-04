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
    python main.py --dry-run               # Full pipeline with execution checks (no orders)
    python main.py --ta-weight 0.7 --sentiment-weight 0.3   # Custom weights
    python main.py --backtest AAPL         # Backtest TA strategy on AAPL
    python main.py --backtest AAPL --days 730 --capital 50000
"""

import argparse
import logging
import sys

from config.settings import Settings
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.sentiment_analysis_agent import SentimentAnalysisAgent
from agents.risk_management_agent import RiskManagementAgent
from agents.portfolio_manager_agent import PortfolioManagerAgent
from agents.execution_agent import ExecutionAgent
from utils.trade_journal import TradeJournal

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
        "--dry-run",
        action="store_true",
        help="Run full pipeline including execution checks, but skip actual order placement",
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
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtesting simulation instead of live analysis",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Lookback period in days for backtesting (default: 365)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000,
        help="Initial capital for backtesting (default: 100000)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show performance dashboard from journal data",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=10,
        help="Number of recent decisions in dashboard (default: 10)",
    )
    args = parser.parse_args()

    setup_logging()

    if args.test:
        success = run_connection_test()
        sys.exit(0 if success else 1)

    symbols = args.symbols or Settings.DEFAULT_SYMBOLS

    # ── Dashboard mode ─────────────────────────────────────────
    if args.dashboard:
        from dashboard import render_dashboard

        symbol_filter = symbols[0] if len(symbols) == 1 else None
        print(render_dashboard(symbol=symbol_filter, last=args.last))
        return

    # ── Backtest mode ────────────────────────────────────────
    if args.backtest:
        from backtest import Backtester

        print(f"\nBacktest Mode")
        print(f"Capital      : ${args.capital:,.0f}")
        print(f"Period       : {args.days} days")
        print(f"Symbols      : {', '.join(symbols)}\n")

        bt = Backtester(initial_capital=args.capital)
        for symbol in symbols:
            try:
                result = bt.run(symbol, days=args.days)
                print(Backtester.format_result(result))
            except Exception as e:
                logging.getLogger("main").error(
                    "Backtest error for %s: %s", symbol, e,
                )
        return

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

    # ── Full pipeline via Portfolio Manager + Execution Agent ─
    journal = TradeJournal()
    execute_orders = args.auto_trade or args.dry_run

    print(f"\nTrading mode : {Settings.TRADING_MODE}")
    print(f"Pipeline     : Portfolio Manager (TA + Sentiment + Risk) → Execution")
    print(f"Weights      : TA={args.ta_weight:.0%}  Sentiment={args.sentiment_weight:.0%}")
    print(f"Symbols      : {', '.join(symbols)}")
    print(f"Timeframe    : {args.timeframe}")
    print(f"Auto-trade   : {'ON' if args.auto_trade else 'OFF'}")
    print(f"Dry-run      : {'ON' if args.dry_run else 'OFF'}")
    print(f"Journal      : {journal.journal_dir}\n")

    pm = PortfolioManagerAgent(
        ta_weight=args.ta_weight,
        sentiment_weight=args.sentiment_weight,
        auto_trade=False,  # execution handled by ExecutionAgent now
    )
    exec_agent = ExecutionAgent(
        client=pm.client,
        dry_run=args.dry_run,
    )

    for symbol in symbols:
        try:
            decision = pm.analyze(symbol, timeframe=args.timeframe)
            print(PortfolioManagerAgent.format_analysis(decision))

            # Log signals and decision to journal
            for sig in decision.get("agent_signals", []):
                s = sig if isinstance(sig, dict) else sig.__dict__
                journal.log_signal(
                    symbol=symbol,
                    agent=s.get("agent", ""),
                    signal=s.get("signal", "HOLD"),
                    score=s.get("normalised_score", 0),
                    price=decision.get("current_price", 0),
                )
            journal.log_decision(decision)

            # ── Execution Agent ──────────────────────────
            if execute_orders:
                exec_analysis = exec_agent.analyze(symbol, decision=decision)
                exec_result = exec_agent.execute(symbol, exec_analysis)
                print(ExecutionAgent.format_analysis(exec_result))

                # Journal the order outcome
                side = exec_result.get("side", "")
                qty = exec_result.get("qty", 0)
                status = exec_result.get("status", "skipped")

                if status not in ("skipped",):
                    order_info = {
                        "id": exec_result.get("order_id", ""),
                        "status": status,
                        "type": exec_result.get("order_type", "market"),
                    }
                    journal.log_order(
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        order_result=order_info,
                        reason=exec_result.get("error", ""),
                    )
        except Exception as e:
            logging.getLogger("main").error(
                "Error in portfolio pipeline for %s: %s", symbol, e,
            )


if __name__ == "__main__":
    main()
