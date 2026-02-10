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
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        daily_pnl = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity > 0 else 0
        return {
            "id": account.id,
            "status": account.status,
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "equity": equity,
            "last_equity": last_equity,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
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
            symbol, tf, start=start, end=end, limit=limit, feed="iex"
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
                "avg_entry_price": float(p.avg_entry_price),
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

    # ── Assets ──────────────────────────────────────────────────

    def get_tradeable_assets(
        self,
        *,
        min_price: float = 5.0,
        max_price: float = 1000.0,
        asset_class: str = "us_equity",
    ) -> list[str]:
        """Get all tradeable stock symbols.

        Args:
            min_price: Minimum price filter (default $5)
            max_price: Maximum price filter (default $1000)
            asset_class: Asset class to filter (default us_equity)

        Returns:
            List of ticker symbols
        """
        assets = self.api.list_assets(status="active", asset_class=asset_class)

        symbols = []
        for asset in assets:
            if asset.tradable and asset.fractionable:
                # Filter out OTC and weird tickers
                if "." not in asset.symbol and len(asset.symbol) <= 5:
                    symbols.append(asset.symbol)

        logger.info("Found %d tradeable assets", len(symbols))
        return symbols

    def get_top_volume_stocks(self, limit: int = 100) -> list[str]:
        """Get top stocks by trading volume (most active).

        Note: This fetches snapshots which requires market data subscription.
        Falls back to a curated list if not available.
        """
        try:
            # Try to get most active stocks
            snapshots = self.api.get_snapshots(
                ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN"]
            )
            # Sort by volume and return top
            sorted_by_vol = sorted(
                snapshots.items(),
                key=lambda x: x[1].daily_bar.v if x[1].daily_bar else 0,
                reverse=True,
            )
            return [sym for sym, _ in sorted_by_vol[:limit]]
        except Exception as e:
            logger.warning("Could not fetch volume data: %s", e)
            # Fallback to curated high-volume list
            return [
                "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN",
                "META", "GOOGL", "NFLX", "COIN", "MARA", "RIOT", "SOFI",
                "PLTR", "INTC", "BAC", "F", "T", "PFE", "AAL", "NIO",
                "SNAP", "UBER", "HOOD", "RBLX", "DKNG", "LCID", "RIVN",
            ][:limit]

    # ── Symbol Validation ──────────────────────────────────────

    _tradeable_cache: dict[str, bool] = {}
    _asset_cache: dict[str, object] = {}

    def is_tradeable(self, symbol: str) -> bool:
        """Check if a symbol is tradeable on Alpaca.

        Returns True if:
        - Asset exists and is active
        - Asset is tradable (not halted)
        - Not an OTC or weird ticker format

        Results are cached to avoid repeated API calls.
        """
        # Check cache first
        if symbol in self._tradeable_cache:
            return self._tradeable_cache[symbol]

        try:
            asset = self.api.get_asset(symbol)
            is_valid = (
                asset.status == "active"
                and asset.tradable
                and "." not in symbol  # Filter OTC-style tickers
                and len(symbol) <= 5  # Filter weird long tickers
            )
            self._tradeable_cache[symbol] = is_valid
            self._asset_cache[symbol] = asset
            return is_valid
        except Exception as e:
            logger.debug("Asset check failed for %s: %s", symbol, e)
            self._tradeable_cache[symbol] = False
            return False

    def validate_symbols(
        self,
        symbols: list[str],
        *,
        min_price: float = 1.0,
        max_price: float = 10000.0,
        min_volume: int = 10000,
        check_price_data: bool = True,
    ) -> list[str]:
        """Validate and filter a list of symbols for trading.

        Args:
            symbols: List of ticker symbols to validate
            min_price: Minimum stock price (default $1)
            max_price: Maximum stock price (default $10000)
            min_volume: Minimum daily volume (default 10k shares)
            check_price_data: Whether to verify price data exists

        Returns:
            List of valid, tradeable symbols
        """
        valid_symbols = []
        rejected = {"not_tradeable": [], "bad_price": [], "low_volume": [], "no_data": []}

        for symbol in symbols:
            # Skip obviously bad symbols
            if not symbol or len(symbol) > 5 or "." in symbol:
                rejected["not_tradeable"].append(symbol)
                continue

            # Check if tradeable on Alpaca
            if not self.is_tradeable(symbol):
                rejected["not_tradeable"].append(symbol)
                continue

            # Check price data if requested
            if check_price_data:
                try:
                    bars = self.get_bars(symbol, timeframe="1Day", limit=5)
                    if bars.empty:
                        rejected["no_data"].append(symbol)
                        continue

                    price = float(bars["close"].iloc[-1])
                    volume = float(bars["volume"].iloc[-1])

                    # Price filter
                    if price < min_price or price > max_price:
                        rejected["bad_price"].append(f"{symbol}(${price:.2f})")
                        continue

                    # Volume filter
                    if volume < min_volume:
                        rejected["low_volume"].append(f"{symbol}({int(volume)})")
                        continue

                except Exception as e:
                    logger.debug("Price check failed for %s: %s", symbol, e)
                    rejected["no_data"].append(symbol)
                    continue

            valid_symbols.append(symbol)

        # Log summary
        if rejected["not_tradeable"]:
            logger.info("Rejected (not tradeable): %s", ", ".join(rejected["not_tradeable"][:10]))
        if rejected["bad_price"]:
            logger.info("Rejected (price): %s", ", ".join(rejected["bad_price"][:10]))
        if rejected["low_volume"]:
            logger.info("Rejected (low volume): %s", ", ".join(rejected["low_volume"][:10]))
        if rejected["no_data"]:
            logger.info("Rejected (no data): %s", ", ".join(rejected["no_data"][:10]))

        logger.info("Symbol validation: %d/%d passed", len(valid_symbols), len(symbols))
        return valid_symbols

    def get_price_quick(self, symbol: str) -> float | None:
        """Get current price quickly, returns None if unavailable."""
        try:
            quote = self.api.get_latest_quote(symbol)
            # Use midpoint of bid/ask for most accurate price
            bid = float(quote.bp) if quote.bp else 0
            ask = float(quote.ap) if quote.ap else 0
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            elif ask > 0:
                return ask
            elif bid > 0:
                return bid

            # Fallback to last trade
            trade = self.api.get_latest_trade(symbol)
            if trade:
                return float(trade.p)
            return None
        except Exception as e:
            logger.debug("Quick price failed for %s: %s", symbol, e)
            return None
