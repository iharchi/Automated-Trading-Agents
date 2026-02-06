"""Portfolio Guardian Agent

Actively monitors and protects the portfolio by:
    - Enforcing take-profit targets (10% default)
    - Managing trailing stops (5% from peak)
    - Enforcing stop-losses
    - Monitoring daily P&L limits
    - Enforcing maximum drawdown limits
    - Auto-closing positions that hit targets
    - Sector diversification and rebalancing

This agent runs continuously to protect profits and limit losses.

Pipeline position:
    Runs independently, monitoring all open positions in real-time.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from agents.base_agent import BaseAgent
from utils.alpaca_client import AlpacaClient
from utils.profit_monitor import ProfitMonitor, PortfolioMetrics

logger = logging.getLogger(__name__)


# Sector mappings for diversification
SECTOR_MAP = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "GOOG": "Technology", "META": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "INTC": "Technology", "CRM": "Technology",
    "ORCL": "Technology", "CSCO": "Technology", "ADBE": "Technology",
    "AVGO": "Technology", "TXN": "Technology", "QCOM": "Technology",
    "IBM": "Technology", "NOW": "Technology", "INTU": "Technology",
    "AMAT": "Technology", "MU": "Technology", "LRCX": "Technology",
    "KLAC": "Technology", "SNPS": "Technology", "CDNS": "Technology",
    "PLTR": "Technology", "CRWD": "Technology", "NET": "Technology",
    "SNOW": "Technology", "DDOG": "Technology", "ZS": "Technology",

    # Consumer Discretionary
    "AMZN": "Consumer", "TSLA": "Consumer", "HD": "Consumer",
    "NKE": "Consumer", "MCD": "Consumer", "SBUX": "Consumer",
    "TGT": "Consumer", "LOW": "Consumer", "BKNG": "Consumer",
    "CMG": "Consumer", "ABNB": "Consumer", "MAR": "Consumer",
    "RIVN": "Consumer", "LCID": "Consumer", "F": "Consumer",
    "GM": "Consumer", "ROKU": "Consumer", "RBLX": "Consumer",

    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "BLK": "Financials", "SCHW": "Financials", "AXP": "Financials",
    "V": "Financials", "MA": "Financials", "PYPL": "Financials",
    "SQ": "Financials", "COIN": "Financials", "HOOD": "Financials",
    "SOFI": "Financials", "AFRM": "Financials", "UPST": "Financials",

    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "BMY": "Healthcare",
    "AMGN": "Healthcare", "GILD": "Healthcare", "MRNA": "Healthcare",
    "BIIB": "Healthcare", "REGN": "Healthcare", "VRTX": "Healthcare",

    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy", "PXD": "Energy",
    "MPC": "Energy", "VLO": "Energy", "PSX": "Energy",
    "OXY": "Energy", "HAL": "Energy", "BKR": "Energy",

    # Crypto-related
    "MARA": "Crypto", "RIOT": "Crypto", "MSTR": "Crypto",
    "CLSK": "Crypto", "HUT": "Crypto", "BITF": "Crypto",

    # ETFs
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF",
    "DIA": "ETF", "VTI": "ETF", "VOO": "ETF",
    "ARKK": "ETF", "XLF": "ETF", "XLK": "ETF",
    "XLE": "ETF", "XLV": "ETF", "XLI": "ETF",

    # Communication
    "DIS": "Communication", "NFLX": "Communication", "T": "Communication",
    "VZ": "Communication", "TMUS": "Communication", "CMCSA": "Communication",

    # Industrials
    "CAT": "Industrials", "DE": "Industrials", "BA": "Industrials",
    "HON": "Industrials", "UPS": "Industrials", "RTX": "Industrials",
    "LMT": "Industrials", "GE": "Industrials", "MMM": "Industrials",

    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "ECL": "Materials", "FCX": "Materials", "NEM": "Materials",
    "NUE": "Materials", "DOW": "Materials", "DD": "Materials",
}

DEFAULT_SECTOR = "Other"


@dataclass
class SectorAllocation:
    """Sector allocation metrics."""
    sector: str
    symbols: list[str]
    market_value: float
    allocation_pct: float
    target_pct: float
    deviation: float  # How far from target


@dataclass
class RebalanceAction:
    """Suggested rebalancing action."""
    symbol: str
    sector: str
    action: str  # "reduce" or "increase"
    current_pct: float
    target_pct: float
    suggested_change_pct: float


@dataclass
class GuardianStatus:
    """Current status of the Portfolio Guardian."""
    is_active: bool = True
    trading_halted: bool = False
    halt_reason: str = ""

    # Limits
    daily_loss_limit_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    take_profit_pct: float = 0.10
    trailing_stop_pct: float = 0.05
    max_sector_allocation_pct: float = 0.35

    # Current state
    current_daily_pnl_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    positions_monitored: int = 0
    positions_at_profit_target: int = 0
    positions_at_stop: int = 0

    # Sector balance
    sector_allocations: list[SectorAllocation] = field(default_factory=list)
    sector_balanced: bool = True
    overweight_sectors: list[str] = field(default_factory=list)

    # Actions taken
    positions_closed_today: list[dict] = field(default_factory=list)
    last_check_time: datetime = field(default_factory=datetime.now)


class PortfolioGuardianAgent(BaseAgent):
    """Agent that actively protects and balances the portfolio."""

    # Default limits
    DEFAULT_TAKE_PROFIT_PCT = 0.10      # 10% profit target
    DEFAULT_TRAILING_STOP_PCT = 0.05    # 5% trailing stop
    DEFAULT_STOP_LOSS_PCT = 0.05        # 5% stop loss
    DEFAULT_DAILY_LOSS_LIMIT = 0.03     # 3% daily loss limit
    DEFAULT_MAX_DRAWDOWN = 0.10         # 10% max drawdown
    DEFAULT_MAX_SECTOR_PCT = 0.35       # 35% max per sector
    DEFAULT_MIN_SECTOR_PCT = 0.05       # 5% min per sector (for diversification)

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
        trailing_stop_pct: float = DEFAULT_TRAILING_STOP_PCT,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
        daily_loss_limit_pct: float = DEFAULT_DAILY_LOSS_LIMIT,
        max_drawdown_pct: float = DEFAULT_MAX_DRAWDOWN,
        max_sector_pct: float = DEFAULT_MAX_SECTOR_PCT,
        auto_trade: bool = False,
        dry_run: bool = False,
    ):
        super().__init__(name="PortfolioGuardian", client=client)
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.stop_loss_pct = stop_loss_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_sector_pct = max_sector_pct
        self.auto_trade = auto_trade
        self.dry_run = dry_run

        # Initialize profit monitor
        self.profit_monitor = ProfitMonitor(
            client=self.client,
            take_profit_pct=take_profit_pct,
            trailing_stop_pct=trailing_stop_pct,
            stop_loss_pct=stop_loss_pct,
            daily_loss_limit_pct=daily_loss_limit_pct,
            max_drawdown_pct=max_drawdown_pct,
        )

        # State
        self._trading_halted = False
        self._halt_reason = ""
        self._positions_closed_today: list[dict] = []

    @staticmethod
    def get_sector(symbol: str) -> str:
        """Get the sector for a symbol."""
        return SECTOR_MAP.get(symbol, DEFAULT_SECTOR)

    def _get_sector_allocations(self, positions: list[dict], equity: float) -> list[SectorAllocation]:
        """Calculate sector allocations from current positions."""
        sector_data: dict[str, dict] = {}

        for pos in positions:
            symbol = pos["symbol"]
            market_value = abs(pos["market_value"])
            sector = self.get_sector(symbol)

            if sector not in sector_data:
                sector_data[sector] = {"symbols": [], "market_value": 0.0}

            sector_data[sector]["symbols"].append(symbol)
            sector_data[sector]["market_value"] += market_value

        # Calculate allocations
        allocations = []
        num_sectors = len(sector_data) if sector_data else 1
        target_pct = 1.0 / num_sectors  # Equal weight target

        for sector, data in sector_data.items():
            allocation_pct = data["market_value"] / equity if equity > 0 else 0
            deviation = allocation_pct - target_pct

            allocations.append(SectorAllocation(
                sector=sector,
                symbols=data["symbols"],
                market_value=data["market_value"],
                allocation_pct=round(allocation_pct, 4),
                target_pct=round(target_pct, 4),
                deviation=round(deviation, 4),
            ))

        # Sort by allocation (highest first)
        allocations.sort(key=lambda x: x.allocation_pct, reverse=True)
        return allocations

    def _check_sector_balance(self, allocations: list[SectorAllocation]) -> tuple[bool, list[str]]:
        """Check if sectors are balanced within limits."""
        overweight = []
        for alloc in allocations:
            if alloc.allocation_pct > self.max_sector_pct:
                overweight.append(alloc.sector)

        balanced = len(overweight) == 0
        return balanced, overweight

    def _get_rebalance_suggestions(
        self, allocations: list[SectorAllocation], positions: list[dict]
    ) -> list[RebalanceAction]:
        """Get suggestions for rebalancing the portfolio."""
        suggestions = []

        for alloc in allocations:
            if alloc.allocation_pct > self.max_sector_pct:
                # Sector is overweight - suggest reducing
                excess_pct = alloc.allocation_pct - self.max_sector_pct
                for symbol in alloc.symbols:
                    pos = next((p for p in positions if p["symbol"] == symbol), None)
                    if pos:
                        pos_pct = abs(pos["market_value"]) / sum(abs(p["market_value"]) for p in positions)
                        suggestions.append(RebalanceAction(
                            symbol=symbol,
                            sector=alloc.sector,
                            action="reduce",
                            current_pct=alloc.allocation_pct,
                            target_pct=self.max_sector_pct,
                            suggested_change_pct=excess_pct,
                        ))

        return suggestions

    def _close_position(self, symbol: str, qty: int, reason: str) -> dict:
        """Close a position (sell for long, buy to cover for short)."""
        result = {
            "symbol": symbol,
            "qty": qty,
            "reason": reason,
            "status": "pending",
            "order_id": None,
            "timestamp": datetime.now().isoformat(),
        }

        if self.dry_run:
            result["status"] = "dry_run"
            logger.info("[DRY RUN] Would close %d shares of %s (reason: %s)", qty, symbol, reason)
            return result

        try:
            # Use Alpaca's close_position API
            order = self.client.api.close_position(symbol)
            result["status"] = "submitted"
            result["order_id"] = order.id
            logger.info("Closed position: %s (%s) - Order ID: %s", symbol, reason, order.id)

            # Reset profit monitor tracking for this symbol
            self.profit_monitor.reset_position_tracking(symbol)

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error("Failed to close %s: %s", symbol, e)

        return result

    def analyze(self, symbol: str = "", **kwargs) -> dict:
        """Analyze the entire portfolio for protection actions.

        This method checks:
            1. Positions hitting take-profit
            2. Positions hitting trailing stops
            3. Positions hitting stop-loss
            4. Daily loss limits
            5. Maximum drawdown
            6. Sector balance
        """
        # Get portfolio metrics
        metrics = self.profit_monitor.get_portfolio_metrics()

        # Check positions to close
        positions_to_close = self.profit_monitor.get_positions_to_close()

        # Check trading halts
        should_halt, halt_reason = self.profit_monitor.should_halt_trading()
        if should_halt:
            self._trading_halted = True
            self._halt_reason = halt_reason

        # Get positions for sector analysis
        positions = self.client.get_positions()
        account = self.client.get_account()
        equity = account["equity"]

        # Sector analysis
        sector_allocations = self._get_sector_allocations(positions, equity)
        sector_balanced, overweight_sectors = self._check_sector_balance(sector_allocations)
        rebalance_suggestions = self._get_rebalance_suggestions(sector_allocations, positions)

        # Build status
        status = GuardianStatus(
            is_active=True,
            trading_halted=self._trading_halted,
            halt_reason=self._halt_reason,
            daily_loss_limit_pct=self.daily_loss_limit_pct,
            max_drawdown_pct=self.max_drawdown_pct,
            take_profit_pct=self.take_profit_pct,
            trailing_stop_pct=self.trailing_stop_pct,
            max_sector_allocation_pct=self.max_sector_pct,
            current_daily_pnl_pct=metrics.daily_pnl_pct,
            current_drawdown_pct=metrics.current_drawdown_pct,
            positions_monitored=metrics.total_positions,
            positions_at_profit_target=sum(1 for p in positions_to_close if p["reason"] == "take_profit"),
            positions_at_stop=sum(1 for p in positions_to_close if p["reason"] in ("stop_loss", "trailing_stop")),
            sector_allocations=sector_allocations,
            sector_balanced=sector_balanced,
            overweight_sectors=overweight_sectors,
            positions_closed_today=self._positions_closed_today,
        )

        return {
            "status": status.__dict__,
            "metrics": metrics.__dict__,
            "positions_to_close": positions_to_close,
            "rebalance_suggestions": [r.__dict__ for r in rebalance_suggestions],
            "sector_allocations": [a.__dict__ for a in sector_allocations],
        }

    def execute(self, symbol: str = "", analysis: dict = None, **kwargs) -> dict:
        """Execute protection actions: close positions hitting targets/stops.

        If auto_trade is False, this just returns what would be done.
        """
        if analysis is None:
            analysis = self.analyze()

        positions_to_close = analysis.get("positions_to_close", [])
        results = []

        if not self.auto_trade:
            return {
                "mode": "monitoring_only",
                "positions_to_close": positions_to_close,
                "actions_taken": [],
            }

        # Close positions hitting targets
        for pos in positions_to_close:
            result = self._close_position(
                pos["symbol"],
                pos["qty"],
                pos["reason"],
            )
            results.append(result)
            self._positions_closed_today.append(result)

        return {
            "mode": "auto_trade" if not self.dry_run else "dry_run",
            "positions_closed": len(results),
            "actions_taken": results,
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
        }

    def run_continuous(
        self,
        interval: int = 30,
        *,
        callback: callable = None,
    ) -> None:
        """Run the guardian in continuous monitoring mode.

        Args:
            interval: Seconds between checks
            callback: Optional function called with (analysis, results)
        """
        logger.info(
            "Starting Portfolio Guardian: interval=%ds, auto_trade=%s, dry_run=%s",
            interval, self.auto_trade, self.dry_run,
        )

        try:
            while True:
                analysis = self.analyze()
                results = self.execute(analysis=analysis)

                if callback:
                    callback(analysis, results)
                else:
                    # Default: print status
                    print(self.format_analysis(analysis))

                if self._trading_halted:
                    logger.warning("TRADING HALTED: %s", self._halt_reason)
                    print(f"\n{'!' * 60}")
                    print(f"  TRADING HALTED: {self._halt_reason}")
                    print(f"{'!' * 60}\n")

                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Portfolio Guardian stopped by user")

    def should_allow_new_trade(self, symbol: str, sector: str = None) -> tuple[bool, str]:
        """Check if a new trade should be allowed based on guardian rules.

        Returns:
            (allowed: bool, reason: str)
        """
        # Check if trading is halted
        if self._trading_halted:
            return False, f"Trading halted: {self._halt_reason}"

        # Check daily loss limit
        breached, pnl_pct = self.profit_monitor.check_daily_loss_limit()
        if breached:
            return False, f"Daily loss limit breached: {pnl_pct:.2%}"

        # Check max drawdown
        dd_breached, dd_pct = self.profit_monitor.check_max_drawdown()
        if dd_breached:
            return False, f"Max drawdown breached: {dd_pct:.2%}"

        # Check sector allocation
        if sector is None:
            sector = self.get_sector(symbol)

        positions = self.client.get_positions()
        account = self.client.get_account()
        equity = account["equity"]

        sector_allocations = self._get_sector_allocations(positions, equity)
        for alloc in sector_allocations:
            if alloc.sector == sector and alloc.allocation_pct >= self.max_sector_pct:
                return False, f"Sector {sector} at max allocation: {alloc.allocation_pct:.1%}"

        return True, "OK"

    @staticmethod
    def format_analysis(analysis: dict) -> str:
        """Human-readable guardian status."""
        status = analysis.get("status", {})
        metrics = analysis.get("metrics", {})

        lines = [
            f"\n{'#' * 70}",
            f"  PORTFOLIO GUARDIAN STATUS",
            f"{'#' * 70}",
        ]

        # Trading status
        if status.get("trading_halted"):
            lines.append(f"  STATUS: HALTED - {status.get('halt_reason', 'Unknown')}")
        else:
            lines.append(f"  STATUS: ACTIVE")

        lines.append(f"{'─' * 70}")

        # Limits
        lines.append(
            f"  Limits: Take Profit={status.get('take_profit_pct', 0):.0%}  "
            f"Trailing Stop={status.get('trailing_stop_pct', 0):.0%}  "
            f"Daily Loss={status.get('daily_loss_limit_pct', 0):.0%}  "
            f"Max DD={status.get('max_drawdown_pct', 0):.0%}"
        )

        # Current state
        lines.append(
            f"  Current: Daily P&L={status.get('current_daily_pnl_pct', 0):+.2%}  "
            f"Drawdown={status.get('current_drawdown_pct', 0):.2%}  "
            f"Positions={status.get('positions_monitored', 0)}"
        )

        # Positions at targets
        at_profit = status.get("positions_at_profit_target", 0)
        at_stop = status.get("positions_at_stop", 0)
        if at_profit > 0 or at_stop > 0:
            lines.append(
                f"  Alerts: {at_profit} at profit target, {at_stop} at stop loss"
            )

        # Sector balance
        lines.append(f"{'─' * 70}")
        if status.get("sector_balanced"):
            lines.append("  Sectors: BALANCED")
        else:
            overweight = status.get("overweight_sectors", [])
            lines.append(f"  Sectors: IMBALANCED - Overweight: {', '.join(overweight)}")

        # Sector allocations
        allocations = analysis.get("sector_allocations", [])
        if allocations:
            lines.append(f"  {'Sector':15s} {'Allocation':>12s} {'Target':>10s} {'Deviation':>10s}")
            for alloc in allocations[:5]:  # Top 5 sectors
                if isinstance(alloc, dict):
                    lines.append(
                        f"  {alloc['sector']:15s} {alloc['allocation_pct']:>11.1%} "
                        f"{alloc['target_pct']:>9.1%} {alloc['deviation']:>+9.1%}"
                    )

        # Positions to close
        to_close = analysis.get("positions_to_close", [])
        if to_close:
            lines.append(f"{'─' * 70}")
            lines.append("  POSITIONS TO CLOSE:")
            for pos in to_close:
                lines.append(
                    f"    {pos['symbol']:8s} {pos['reason']:15s} "
                    f"P&L: {pos['pnl_pct']:+.2%} (${pos['pnl']:+,.2f})"
                )

        lines.append(f"{'#' * 70}\n")
        return "\n".join(lines)
