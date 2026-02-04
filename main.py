#!/usr/bin/env python3
"""Entry point for the Automated Trading Agents system.

Usage:
    python main.py                         # Full pipeline (TA + Sentiment + Risk)
    python main.py AAPL TSLA MSFT          # Analyse specific symbols
    python main.py --agent ta              # Technical Analysis only
    python main.py --agent sentiment       # Sentiment Analysis only
    python main.py --agent risk            # TA + Risk (no sentiment)
    python main.py --agent mtf             # Multi-Timeframe Analysis only
    python main.py --test                  # Run connection test only
    python main.py --auto-trade            # Enable paper-trade execution
    python main.py --dry-run               # Full pipeline with execution checks (no orders)
    python main.py --multi-timeframe       # Use multi-timeframe analysis pipeline
    python main.py --multi-timeframe --mtf-mode majority   # MTF with majority agreement
    python main.py --ta-weight 0.7 --sentiment-weight 0.3   # Custom weights
    python main.py --backtest AAPL         # Backtest TA strategy on AAPL
    python main.py --backtest AAPL --days 730 --capital 50000
    python main.py --scan                  # Scan default watchlist for opportunities
    python main.py --scan --source NASDAQ100_TOP50 --top-n 15
    python main.py --scan --source /path/to/symbols.csv --signal-filter buy
    python main.py --positions             # Show positions with trailing stop status
    python main.py --update-stops          # Update trailing stops based on current prices
    python main.py --trail-mode atr        # Set trailing mode (percentage, atr, fixed, stepped)
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
from agents.multi_timeframe_agent import MultiTimeframeAgent
from utils.trade_journal import TradeJournal
from utils.notifier import Notifier
from utils.watchlist_scanner import WatchlistScanner
from utils.trailing_stop_manager import TrailingStopManager

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
    "mtf": {
        "class": MultiTimeframeAgent,
        "label": "Multi-Timeframe Analysis",
    },
}


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, Settings.LOG_LEVEL, logging.INFO),
        format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def create_notifier() -> Notifier:
    """Build a Notifier instance from Settings."""
    return Notifier(
        # Email
        email_enabled=Settings.NOTIFY_EMAIL_ENABLED,
        smtp_host=Settings.NOTIFY_SMTP_HOST,
        smtp_port=Settings.NOTIFY_SMTP_PORT,
        smtp_user=Settings.NOTIFY_SMTP_USER,
        smtp_password=Settings.NOTIFY_SMTP_PASSWORD,
        email_from=Settings.NOTIFY_EMAIL_FROM,
        email_to=Settings.NOTIFY_EMAIL_TO,
        # Slack
        slack_enabled=Settings.NOTIFY_SLACK_ENABLED,
        slack_webhook_url=Settings.NOTIFY_SLACK_WEBHOOK,
        # Discord
        discord_enabled=Settings.NOTIFY_DISCORD_ENABLED,
        discord_webhook_url=Settings.NOTIFY_DISCORD_WEBHOOK,
        # Generic webhook
        webhook_enabled=Settings.NOTIFY_WEBHOOK_ENABLED,
        webhook_url=Settings.NOTIFY_WEBHOOK_URL,
        # Filtering
        enabled_events=Settings.NOTIFY_ENABLED_EVENTS,
        min_signal_score=Settings.NOTIFY_MIN_SIGNAL_SCORE,
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
        "--multi-timeframe",
        action="store_true",
        help="Enable multi-timeframe analysis (combines signals from multiple timeframes)",
    )
    parser.add_argument(
        "--mtf-mode",
        default="unanimous",
        choices=["unanimous", "majority", "weighted"],
        help="Multi-timeframe agreement mode (default: unanimous)",
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
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan a watchlist for trading opportunities",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Watchlist source: SP500_TOP50, NASDAQ100_TOP50, POPULAR_TECH, HIGH_DIVIDEND, ETFS, CSV path, or comma-separated symbols",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top candidates to show (default from config)",
    )
    parser.add_argument(
        "--signal-filter",
        choices=["all", "buy", "sell"],
        default=None,
        help="Filter results by signal type (default: all)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=None,
        help="Minimum absolute score to include (default from config)",
    )
    parser.add_argument(
        "--export-csv",
        default=None,
        help="Export scan results to CSV file",
    )
    parser.add_argument(
        "--list-watchlists",
        action="store_true",
        help="List available built-in watchlists",
    )
    parser.add_argument(
        "--positions",
        action="store_true",
        help="Show current positions with trailing stop status",
    )
    parser.add_argument(
        "--update-stops",
        action="store_true",
        help="Update trailing stops based on current prices",
    )
    parser.add_argument(
        "--trail-mode",
        choices=["percentage", "atr", "fixed", "stepped"],
        default=None,
        help="Trailing stop mode for new positions",
    )
    parser.add_argument(
        "--trail-value",
        type=float,
        default=None,
        help="Trail value (percent, ATR multiplier, or dollar amount depending on mode)",
    )
    parser.add_argument(
        "--sync-stops",
        action="store_true",
        help="Sync trailing stops with broker positions",
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

    # ── List watchlists ────────────────────────────────────────
    if args.list_watchlists:
        print("\nAvailable built-in watchlists:")
        for name in WatchlistScanner.list_watchlists():
            print(f"  - {name}")
        print("\nYou can also use:")
        print("  - Path to a CSV file with a 'symbol' column")
        print("  - Comma-separated symbols (e.g., 'AAPL,MSFT,GOOGL')")
        print("  - 'alpaca' to scan tradeable Alpaca assets\n")
        return

    # ── Scan mode ──────────────────────────────────────────────
    if args.scan:
        source = args.source or Settings.SCAN_DEFAULT_SOURCE
        top_n = args.top_n if args.top_n is not None else Settings.SCAN_TOP_N
        signal_filter = args.signal_filter or Settings.SCAN_SIGNAL_FILTER
        min_score = args.min_score if args.min_score is not None else Settings.SCAN_MIN_SCORE

        print(f"\nWatchlist Scanner")
        print(f"Source       : {source}")
        print(f"Timeframe    : {args.timeframe}")
        print(f"Top N        : {top_n}")
        print(f"Filter       : {signal_filter}")
        print(f"Min Score    : {min_score}")
        print(f"Scanning...\n")

        scanner = WatchlistScanner(
            timeframe=args.timeframe,
            rate_limit_delay=Settings.SCAN_RATE_LIMIT,
        )
        summary = scanner.scan(
            source=source,
            top_n=top_n,
            signal_filter=signal_filter,
            min_score=min_score,
        )
        print(WatchlistScanner.format_results(summary))

        if args.export_csv:
            WatchlistScanner.to_csv(summary, args.export_csv)
            print(f"Results exported to {args.export_csv}\n")
        return

    # ── Positions / Trailing Stops mode ───────────────────────
    if args.positions or args.update_stops or args.sync_stops:
        from utils.alpaca_client import AlpacaClient

        client = AlpacaClient()
        trail_mode = args.trail_mode or Settings.TRAIL_DEFAULT_MODE
        trail_value = args.trail_value
        if trail_value is None:
            if trail_mode == "percentage":
                trail_value = Settings.TRAIL_PERCENTAGE
            elif trail_mode == "atr":
                trail_value = Settings.TRAIL_ATR_MULTIPLIER
            elif trail_mode == "fixed":
                trail_value = Settings.TRAIL_FIXED_AMOUNT
            else:
                trail_value = Settings.TRAIL_PERCENTAGE

        manager = TrailingStopManager(
            client=client,
            auto_sync=Settings.TRAIL_AUTO_SYNC,
        )

        # Sync with broker if requested
        if args.sync_stops:
            print("\nSyncing trailing stops with broker positions...")
            result = manager.sync_with_broker()
            print(f"  Added: {', '.join(result['added']) or 'none'}")
            print(f"  Removed: {', '.join(result['removed']) or 'none'}\n")

        # Update stops based on current prices
        if args.update_stops:
            print("\nUpdating trailing stops...")
            positions = manager.list_positions()
            if not positions:
                print("  No positions to update.\n")
            else:
                # Get current prices
                prices = {}
                for pos in positions:
                    try:
                        quote = client.get_latest_quote(pos.symbol)
                        prices[pos.symbol] = (quote["ask_price"] + quote["bid_price"]) / 2
                    except Exception as e:
                        logging.getLogger("main").warning(
                            "Could not get price for %s: %s", pos.symbol, e
                        )

                updates = manager.update_stops(prices)
                if updates:
                    print(manager.format_updates(updates))
                else:
                    print("  No stops needed adjustment.\n")

        # Show positions
        if args.positions:
            positions = [manager.get_status(p.symbol) for p in manager.list_positions()]
            if positions:
                print(TrailingStopManager.format_status(positions))
            else:
                print("\n  No tracked positions.\n")

            # Also show broker positions for reference
            try:
                broker_positions = client.get_positions()
                if broker_positions:
                    print("  Broker Positions:")
                    print(f"  {'Symbol':<8} {'Qty':>8} {'Price':>10} {'P&L':>12}")
                    print(f"  {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 12}")
                    for p in broker_positions:
                        print(
                            f"  {p['symbol']:<8} {p['qty']:>8} "
                            f"${p['current_price']:>9.2f} "
                            f"${p['unrealized_pl']:>11.2f}"
                        )
                    print()
            except Exception as e:
                logging.getLogger("main").warning("Could not fetch broker positions: %s", e)

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
                elif args.agent == "mtf":
                    # Multi-timeframe agent needs special config
                    agent = MultiTimeframeAgent(
                        timeframes=Settings.MTF_TIMEFRAMES,
                        agreement_mode=args.mtf_mode,
                        min_agreement=Settings.MTF_MIN_AGREEMENT,
                        auto_trade=args.auto_trade,
                    )
                    analysis = agent.analyze(symbol)
                    print(MultiTimeframeAgent.format_analysis(analysis))
                else:
                    agent_cls = STANDALONE_AGENTS[args.agent]["class"]
                    agent = agent_cls(auto_trade=args.auto_trade)
                    run_standalone(agent, symbol, timeframe=args.timeframe)
            except Exception as e:
                logging.getLogger("main").error(
                    "Error running %s on %s: %s", label, symbol, e,
                )
        return

    # ── Multi-Timeframe pipeline mode ─────────────────────────
    use_mtf = args.multi_timeframe or Settings.MTF_ENABLED

    if use_mtf:
        journal = TradeJournal()
        notifier = create_notifier()
        execute_orders = args.auto_trade or args.dry_run

        print(f"\nTrading mode : {Settings.TRADING_MODE}")
        print(f"Pipeline     : Multi-Timeframe TA → Risk → Execution")
        print(f"Timeframes   : {', '.join(Settings.MTF_TIMEFRAMES)}")
        print(f"Agreement    : {args.mtf_mode}")
        print(f"Symbols      : {', '.join(symbols)}")
        print(f"Auto-trade   : {'ON' if args.auto_trade else 'OFF'}")
        print(f"Dry-run      : {'ON' if args.dry_run else 'OFF'}")
        print(f"Journal      : {journal.journal_dir}\n")

        mtf_agent = MultiTimeframeAgent(
            timeframes=Settings.MTF_TIMEFRAMES,
            agreement_mode=args.mtf_mode,
            min_agreement=Settings.MTF_MIN_AGREEMENT,
            auto_trade=False,
        )
        risk_agent = RiskManagementAgent(auto_trade=False)
        exec_agent = ExecutionAgent(client=mtf_agent.client, dry_run=args.dry_run)

        for symbol in symbols:
            try:
                # Run multi-timeframe analysis
                mtf_result = mtf_agent.analyze(symbol)
                print(MultiTimeframeAgent.format_analysis(mtf_result))

                signal = mtf_result.get("final_signal", "HOLD")
                price = mtf_result.get("current_price", 0)
                atr = mtf_result.get("atr", 0)

                # Log to journal
                journal.log_signal(
                    symbol=symbol,
                    agent="MultiTimeframe",
                    signal=signal,
                    score=mtf_result.get("weighted_score", 0),
                    price=price,
                )

                # Notify on actionable signals
                if signal in ("BUY", "SELL"):
                    notifier.notify_signal(
                        symbol=symbol,
                        signal=signal,
                        score=mtf_result.get("weighted_score", 0),
                        price=price,
                        confidence=mtf_result.get("confidence", 0),
                        agreement=mtf_result.get("agreement_ratio", 0),
                    )

                # Run through risk management if actionable
                if signal != "HOLD":
                    proposed_side = "buy" if signal == "BUY" else "sell"
                    risk_result = risk_agent.analyze(
                        symbol,
                        proposed_side=proposed_side,
                        price=price,
                        atr=atr,
                    )
                    print(RiskManagementAgent.format_analysis(risk_result))

                    # Build decision dict for execution agent
                    decision = {
                        "symbol": symbol,
                        "signal": signal,
                        "risk_approved": risk_result.get("approved", False),
                        "position_size": risk_result.get("position_size", 0),
                        "current_price": price,
                        "stop_loss": risk_result.get("stop_loss", 0),
                        "take_profit": risk_result.get("take_profit", 0),
                        "atr": atr,
                        "combined_score": mtf_result.get("weighted_score", 0),
                        "confidence": mtf_result.get("confidence", 0),
                    }
                    journal.log_decision(decision)

                    # Execute if enabled
                    if execute_orders and risk_result.get("approved"):
                        exec_analysis = exec_agent.analyze(symbol, decision=decision)
                        exec_result = exec_agent.execute(symbol, exec_analysis)
                        print(ExecutionAgent.format_analysis(exec_result))

                        status = exec_result.get("status", "skipped")
                        if status not in ("skipped",):
                            order_info = {
                                "id": exec_result.get("order_id", ""),
                                "status": status,
                                "type": exec_result.get("order_type", "market"),
                            }
                            journal.log_order(
                                symbol=symbol,
                                side=proposed_side,
                                qty=exec_result.get("qty", 0),
                                order_result=order_info,
                            )

                            if status == "filled":
                                notifier.notify_order_filled(
                                    symbol=symbol,
                                    side=proposed_side,
                                    qty=exec_result.get("qty", 0),
                                    avg_price=exec_result.get("filled_avg_price", 0),
                                    order_id=exec_result.get("order_id", ""),
                                )
                            elif status == "failed":
                                notifier.notify_order_failed(
                                    symbol=symbol,
                                    side=proposed_side,
                                    qty=exec_result.get("qty", 0),
                                    error=exec_result.get("error", "Unknown error"),
                                )
            except Exception as e:
                logging.getLogger("main").error(
                    "Error in MTF pipeline for %s: %s", symbol, e,
                )
        return

    # ── Full pipeline via Portfolio Manager + Execution Agent ─
    journal = TradeJournal()
    notifier = create_notifier()
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

            # ── Notify on actionable signals ─────────────
            signal = decision.get("signal", "HOLD")
            if signal in ("BUY", "SELL"):
                notifier.notify_signal(
                    symbol=symbol,
                    signal=signal,
                    score=decision.get("combined_score", 0),
                    price=decision.get("current_price", 0),
                    confidence=decision.get("confidence", 0),
                )

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

                    # ── Notify on order events ───────────
                    if status == "filled":
                        notifier.notify_order_filled(
                            symbol=symbol,
                            side=side,
                            qty=qty,
                            avg_price=exec_result.get("filled_avg_price", 0),
                            order_id=exec_result.get("order_id", ""),
                        )
                    elif status == "failed":
                        notifier.notify_order_failed(
                            symbol=symbol,
                            side=side,
                            qty=qty,
                            error=exec_result.get("error", "Unknown error"),
                        )
                    elif status not in ("dry_run",):
                        # Order was placed but not yet filled
                        notifier.notify_order_placed(
                            symbol=symbol,
                            side=side,
                            qty=qty,
                            order_type=exec_result.get("order_type", "market"),
                            order_id=exec_result.get("order_id", ""),
                        )
        except Exception as e:
            logging.getLogger("main").error(
                "Error in portfolio pipeline for %s: %s", symbol, e,
            )


if __name__ == "__main__":
    main()
