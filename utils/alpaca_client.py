import logging
from datetime import datetime, timedelta

import alpaca_trade_api as tradeapi
import pandas as pd

from config.settings import Settings

logger = logging.getLogger(__name__)


class AlpacaClient:
    """Wrapper around the Alpaca REST API for paper trading."""

    def __init__(self):
        if not Settings.validate():
            raise ValueError(
                "Alpaca API keys are not configured. "
                "Copy .env.example to .env and fill in your keys."
            )
        self.api = tradeapi.REST(
            key_id=Settings.ALPACA_API_KEY,
            secret_key=Settings.ALPACA_SECRET_KEY,
            base_url=Settings.ALPACA_BASE_URL,
            api_version="v2",
        )

    # ── Account ──────────────────────────────────────────────

    def get_account(self) -> dict:
        """Return account details as a dictionary."""
        account = self.api.get_account()
        return {
            "id": account.id,
            "status": account.status,
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "currency": account.currency,
        }

    # ── Market Data ──────────────────────────────────────────

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 100,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch historical bars and return as a DataFrame.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            timeframe: Bar timeframe – "1Min", "5Min", "15Min", "1Hour", "1Day".
            timeframe_map: Maps human-readable strings to Alpaca TimeFrame.
            limit: Maximum number of bars.
            start: ISO-8601 start date string (optional).
            end: ISO-8601 end date string (optional).
        """
        from alpaca_trade_api.rest import TimeFrame

        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, "Min"),
            "15Min": TimeFrame(15, "Min"),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, TimeFrame.Day)

        if start is None:
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        bars = self.api.get_bars(
            symbol, tf, start=start, end=end, limit=limit
        ).df

        if bars.empty:
            logger.warning("No bars returned for %s", symbol)
            return bars

        bars.index = bars.index.tz_convert("US/Eastern")
        return bars

    def get_latest_quote(self, symbol: str) -> dict:
        """Return the latest quote for a symbol."""
        quote = self.api.get_latest_quote(symbol)
        return {
            "symbol": symbol,
            "ask_price": float(quote.ap),
            "ask_size": quote.as_,
            "bid_price": float(quote.bp),
            "bid_size": quote.bs,
        }

    # ── Orders ───────────────────────────────────────────────

    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> dict:
        """Submit a paper-trade order and return order details."""
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force,
        )
        logger.info(
            "Order submitted: %s %s %s @ %s", side.upper(), qty, symbol, order_type
        )
        return {
            "id": order.id,
            "symbol": order.symbol,
            "qty": order.qty,
            "side": order.side,
            "type": order.type,
            "status": order.status,
            "submitted_at": str(order.submitted_at),
        }

    def get_positions(self) -> list[dict]:
        """Return all open positions."""
        positions = self.api.list_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": int(p.qty),
                "side": p.side,
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "current_price": float(p.current_price),
            }
            for p in positions
        ]

    # ── News ─────────────────────────────────────────────────

    def get_news(
        self,
        symbol: str,
        limit: int = 20,
        start: str | None = None,
        end: str | None = None,
        include_content: bool = False,
    ) -> list[dict]:
        """Fetch recent news articles for a symbol.

        Returns a list of dicts with keys:
            id, headline, summary, source, url, created_at, symbols
        """
        kwargs: dict = {"symbol": symbol, "limit": limit, "include_content": include_content}
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end

        articles = self.api.get_news(**kwargs)
        return [
            {
                "id": a.id,
                "headline": a.headline,
                "summary": getattr(a, "summary", "") or "",
                "source": a.source,
                "url": getattr(a, "url", ""),
                "created_at": str(a.created_at),
                "symbols": list(a.symbols) if a.symbols else [],
            }
            for a in articles
        ]

    # ── Market Clock ─────────────────────────────────────────

    def is_market_open(self) -> bool:
        """Check whether the market is currently open."""
        clock = self.api.get_clock()
        return clock.is_open
