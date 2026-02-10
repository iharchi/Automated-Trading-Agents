"""Risk Management Agent

Evaluates proposed trades against portfolio-level constraints and
calculates safe position sizes.  This agent does **not** generate
directional signals — it gates and sizes orders proposed by the other
agents (Technical Analysis, Sentiment, etc.).

Checks performed:
    1. Single-position concentration limit (max % of equity in one stock)
    2. Total portfolio exposure limit (sum of all positions / equity)
    3. Per-trade risk budget via ATR-based stop-loss sizing
    4. Available buying power validation
    5. Stop-loss and take-profit price calculation
    6. Daily loss limit check
    7. Maximum drawdown check
    8. Sector concentration check

Typical flow:
    ta_analysis  = ta_agent.analyze("AAPL")
    risk_review  = risk_agent.analyze("AAPL", proposed_side="buy",
                                      price=ta_analysis["current_price"],
                                      atr=ta_analysis["atr"])
    # risk_review["approved"] tells you whether the trade is safe
    # risk_review["position_size"] tells you how many shares to buy
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from agents.base_agent import BaseAgent
from utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

# ── Default limits ───────────────────────────────────────────────
# Note: Increased for paper trading flexibility. Adjust for live trading.

MAX_POSITION_PCT = 0.30        # Max 30% of equity in a single stock (optimized)
MAX_PORTFOLIO_EXPOSURE = 1.75  # Allow up to 175% exposure (margin, optimized)
RISK_PER_TRADE_PCT = 0.03      # Risk at most 3% of equity per trade (optimized)
ATR_STOP_MULTIPLIER = 1.75     # Stop-loss = entry - (ATR * 1.75) (wider stops)
TAKE_PROFIT_RATIO = 2.5        # Take-profit at 2.5:1 reward-to-risk (optimized)

# Limits for smarter risk management (relaxed for paper trading)
DAILY_LOSS_LIMIT_PCT = 0.10    # Stop trading after 10% daily loss (relaxed for paper)
MAX_DRAWDOWN_PCT = 0.25        # Stop trading after 25% drawdown from peak (relaxed)
MAX_SECTOR_CONCENTRATION = 0.60  # Max 60% in any single sector (more flexibility)
MAX_CORRELATED_POSITIONS = 5   # Max positions in highly correlated assets

# Sector mappings (simplified - full version in portfolio_guardian_agent.py)
SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "NVDA": "Technology", "AMD": "Technology", "META": "Technology",
    "AMZN": "Consumer", "TSLA": "Consumer", "RIVN": "Consumer",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "COIN": "Crypto", "MARA": "Crypto", "RIOT": "Crypto", "MSTR": "Crypto",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
}


@dataclass
class RiskCheck:
    """Result of a single risk rule evaluation."""

    rule: str
    passed: bool
    detail: str


@dataclass
class RiskResult:
    """Aggregated risk assessment for a proposed trade."""

    symbol: str
    proposed_side: str = ""
    checks: list[RiskCheck] = field(default_factory=list)
    approved: bool = False
    position_size: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_amount: float = 0.0
    current_exposure_pct: float = 0.0
    current_position_pct: float = 0.0
    daily_pnl_pct: float = 0.0
    drawdown_pct: float = 0.0
    sector: str = ""
    sector_exposure_pct: float = 0.0


class RiskManagementAgent(BaseAgent):
    """Agent that validates and sizes trades against portfolio risk rules."""

    def __init__(
        self,
        client: AlpacaClient | None = None,
        *,
        max_position_pct: float = MAX_POSITION_PCT,
        max_portfolio_exposure: float = MAX_PORTFOLIO_EXPOSURE,
        risk_per_trade_pct: float = RISK_PER_TRADE_PCT,
        atr_stop_multiplier: float = ATR_STOP_MULTIPLIER,
        take_profit_ratio: float = TAKE_PROFIT_RATIO,
        daily_loss_limit_pct: float = DAILY_LOSS_LIMIT_PCT,
        max_drawdown_pct: float = MAX_DRAWDOWN_PCT,
        max_sector_concentration: float = MAX_SECTOR_CONCENTRATION,
        auto_trade: bool = False,
    ):
        super().__init__(name="RiskManagement", client=client)
        self.max_position_pct = max_position_pct
        self.max_portfolio_exposure = max_portfolio_exposure
        self.risk_per_trade_pct = risk_per_trade_pct
        self.atr_stop_multiplier = atr_stop_multiplier
        self.take_profit_ratio = take_profit_ratio
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_sector_concentration = max_sector_concentration
        self.auto_trade = auto_trade

        # Daily tracking state
        self._day_start_equity: float = 0.0
        self._peak_equity: float = 0.0
        self._current_date: date = date.today()

    # ── Daily/Drawdown Tracking ───────────────────────────────

    def _update_tracking(self, equity: float) -> tuple[float, float]:
        """Update daily and drawdown tracking. Returns (daily_pnl_pct, drawdown_pct)."""
        today = date.today()

        # Reset on new day
        if today != self._current_date:
            self._current_date = today
            self._day_start_equity = equity
            logger.info("New trading day - reset daily tracking. Start equity: $%.2f", equity)

        # Initialize on first call
        if self._day_start_equity == 0:
            self._day_start_equity = equity

        if self._peak_equity == 0:
            self._peak_equity = equity

        # Update peak
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Calculate metrics
        daily_pnl_pct = (equity - self._day_start_equity) / self._day_start_equity if self._day_start_equity > 0 else 0
        drawdown_pct = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0

        return daily_pnl_pct, drawdown_pct

    @staticmethod
    def get_sector(symbol: str) -> str:
        """Get sector for a symbol."""
        return SECTOR_MAP.get(symbol.upper(), "Other")

    # ── Portfolio helpers ────────────────────────────────────

    def _get_portfolio_snapshot(self) -> dict:
        """Fetch account + positions and compute key ratios."""
        account = self.client.get_account()
        positions = self.client.get_positions()

        equity = account["equity"]
        total_market_value = sum(abs(p["market_value"]) for p in positions)
        exposure_pct = total_market_value / equity if equity else 0.0

        positions_by_symbol = {p["symbol"]: p for p in positions}

        return {
            "equity": equity,
            "cash": account["cash"],
            "buying_power": account["buying_power"],
            "total_market_value": total_market_value,
            "exposure_pct": round(exposure_pct, 4),
            "positions": positions_by_symbol,
        }

    # ── Position sizing ──────────────────────────────────────

    def _calculate_position_size(
        self, equity: float, price: float, atr: float
    ) -> tuple[int, float, float]:
        """ATR-based position sizing.

        Returns:
            (shares, stop_loss_price, risk_amount)
        """
        if price <= 0 or atr <= 0:
            return 0, 0.0, 0.0

        risk_budget = equity * self.risk_per_trade_pct
        stop_distance = atr * self.atr_stop_multiplier

        if stop_distance <= 0:
            return 0, 0.0, 0.0

        shares = int(risk_budget / stop_distance)
        shares = max(shares, 0)

        stop_loss = round(price - stop_distance, 2)
        actual_risk = shares * stop_distance

        return shares, stop_loss, round(actual_risk, 2)

    # ── Risk checks ──────────────────────────────────────────

    def _check_concentration(
        self, symbol: str, price: float, shares: int, equity: float,
        existing_value: float,
    ) -> RiskCheck:
        """Ensure the position won't exceed max single-stock concentration."""
        new_value = existing_value + (shares * price)
        pct = new_value / equity if equity else 1.0
        limit = self.max_position_pct

        if pct > limit:
            return RiskCheck(
                rule="concentration",
                passed=False,
                detail=(
                    f"Position would be {pct:.1%} of equity "
                    f"(limit {limit:.0%}). "
                    f"Existing ${existing_value:,.0f} + "
                    f"new ${shares * price:,.0f}"
                ),
            )
        return RiskCheck(
            rule="concentration",
            passed=True,
            detail=f"Position {pct:.1%} of equity (limit {limit:.0%})",
        )

    def _check_exposure(
        self, total_market_value: float, new_trade_value: float, equity: float
    ) -> RiskCheck:
        """Ensure total portfolio exposure stays within limit."""
        new_exposure = (total_market_value + new_trade_value) / equity if equity else 1.0
        limit = self.max_portfolio_exposure

        if new_exposure > limit:
            return RiskCheck(
                rule="exposure",
                passed=False,
                detail=(
                    f"Total exposure would be {new_exposure:.1%} "
                    f"(limit {limit:.0%})"
                ),
            )
        return RiskCheck(
            rule="exposure",
            passed=True,
            detail=f"Total exposure {new_exposure:.1%} (limit {limit:.0%})",
        )

    def _check_buying_power(
        self, buying_power: float, price: float, shares: int
    ) -> RiskCheck:
        """Ensure there's enough buying power for the order."""
        cost = shares * price
        if cost > buying_power:
            return RiskCheck(
                rule="buying_power",
                passed=False,
                detail=(
                    f"Order cost ${cost:,.0f} exceeds "
                    f"buying power ${buying_power:,.0f}"
                ),
            )
        return RiskCheck(
            rule="buying_power",
            passed=True,
            detail=f"Order cost ${cost:,.0f} within buying power ${buying_power:,.0f}",
        )

    def _check_position_for_sell(
        self, symbol: str, positions: dict
    ) -> RiskCheck:
        """Ensure we actually hold the stock before selling."""
        if symbol not in positions:
            return RiskCheck(
                rule="sell_position",
                passed=False,
                detail=f"No existing position in {symbol} to sell",
            )
        qty = positions[symbol]["qty"]
        return RiskCheck(
            rule="sell_position",
            passed=True,
            detail=f"Holding {qty} shares of {symbol}",
        )

    def _check_daily_loss_limit(self, daily_pnl_pct: float) -> RiskCheck:
        """Check if daily loss limit has been breached."""
        limit = self.daily_loss_limit_pct
        breached = daily_pnl_pct <= -limit

        if breached:
            return RiskCheck(
                rule="daily_loss_limit",
                passed=False,
                detail=f"Daily loss {daily_pnl_pct:.2%} exceeds limit {-limit:.2%}",
            )
        return RiskCheck(
            rule="daily_loss_limit",
            passed=True,
            detail=f"Daily P&L {daily_pnl_pct:+.2%} within limit {-limit:.2%}",
        )

    def _check_max_drawdown(self, drawdown_pct: float) -> RiskCheck:
        """Check if maximum drawdown has been breached."""
        limit = self.max_drawdown_pct
        breached = drawdown_pct >= limit

        if breached:
            return RiskCheck(
                rule="max_drawdown",
                passed=False,
                detail=f"Drawdown {drawdown_pct:.2%} exceeds limit {limit:.2%}",
            )
        return RiskCheck(
            rule="max_drawdown",
            passed=True,
            detail=f"Drawdown {drawdown_pct:.2%} within limit {limit:.2%}",
        )

    def _check_sector_concentration(
        self, symbol: str, positions: dict, equity: float, new_trade_value: float
    ) -> RiskCheck:
        """Check if adding this position would over-concentrate a sector."""
        sector = self.get_sector(symbol)
        limit = self.max_sector_concentration

        # Calculate current sector exposure
        sector_value = 0.0
        for sym, pos in positions.items():
            if self.get_sector(sym) == sector:
                sector_value += abs(pos.get("market_value", 0))

        # Add new trade value
        new_sector_value = sector_value + new_trade_value
        new_sector_pct = new_sector_value / equity if equity > 0 else 0

        if new_sector_pct > limit:
            return RiskCheck(
                rule="sector_concentration",
                passed=False,
                detail=f"Sector {sector} would be {new_sector_pct:.1%} (limit {limit:.0%})",
            )
        return RiskCheck(
            rule="sector_concentration",
            passed=True,
            detail=f"Sector {sector} at {new_sector_pct:.1%} (limit {limit:.0%})",
        )

    # ── Core lifecycle ───────────────────────────────────────

    def analyze(self, symbol: str, **kwargs) -> dict:
        """Assess risk for a proposed trade.

        Required kwargs:
            proposed_side: "buy" or "sell"
            price: current share price
            atr: Average True Range (from TA agent)

        Optional kwargs:
            skip_daily_checks: bool - Skip daily loss/drawdown checks (default False)
        """
        proposed_side = kwargs.get("proposed_side", "buy")
        price = float(kwargs.get("price", 0))
        atr = float(kwargs.get("atr", 0))
        skip_daily_checks = kwargs.get("skip_daily_checks", False)

        snapshot = self._get_portfolio_snapshot()
        equity = snapshot["equity"]

        # Update daily and drawdown tracking
        daily_pnl_pct, drawdown_pct = self._update_tracking(equity)

        existing_pos = snapshot["positions"].get(symbol, {})
        existing_value = abs(existing_pos.get("market_value", 0))

        checks: list[RiskCheck] = []

        # Always check daily loss and drawdown limits first (unless skipped)
        if not skip_daily_checks:
            checks.append(self._check_daily_loss_limit(daily_pnl_pct))
            checks.append(self._check_max_drawdown(drawdown_pct))

        if proposed_side == "buy":
            shares, stop_loss, risk_amount = self._calculate_position_size(
                equity, price, atr
            )
            take_profit = round(
                price + (price - stop_loss) * self.take_profit_ratio, 2
            ) if stop_loss > 0 else 0.0

            trade_value = shares * price

            checks.append(
                self._check_concentration(
                    symbol, price, shares, equity, existing_value
                )
            )
            checks.append(
                self._check_exposure(
                    snapshot["total_market_value"], trade_value, equity
                )
            )
            checks.append(
                self._check_buying_power(snapshot["buying_power"], price, shares)
            )
            # Check sector concentration
            checks.append(
                self._check_sector_concentration(
                    symbol, snapshot["positions"], equity, trade_value
                )
            )
        else:
            # Sell: just verify we hold the position
            shares = existing_pos.get("qty", 0)
            stop_loss = 0.0
            take_profit = 0.0
            risk_amount = 0.0
            checks.append(
                self._check_position_for_sell(symbol, snapshot["positions"])
            )

        # Approve if all checks pass (share count from PositionSizer is used instead)
        approved = all(c.passed for c in checks)

        # Get sector info
        sector = self.get_sector(symbol)
        sector_value = sum(
            abs(pos.get("market_value", 0))
            for sym, pos in snapshot["positions"].items()
            if self.get_sector(sym) == sector
        )
        sector_exposure_pct = sector_value / equity if equity > 0 else 0

        result = RiskResult(
            symbol=symbol,
            proposed_side=proposed_side,
            checks=checks,
            approved=approved,
            position_size=shares,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            current_exposure_pct=snapshot["exposure_pct"],
            current_position_pct=round(
                existing_value / equity if equity else 0, 4
            ),
            daily_pnl_pct=round(daily_pnl_pct, 4),
            drawdown_pct=round(drawdown_pct, 4),
            sector=sector,
            sector_exposure_pct=round(sector_exposure_pct, 4),
        )
        return result.__dict__

    def execute(self, symbol: str, analysis: dict, **kwargs) -> dict:
        """Place the risk-approved order (if auto_trade is enabled)."""
        result = {
            "symbol": symbol,
            "approved": analysis.get("approved", False),
            "position_size": analysis.get("position_size", 0),
            "stop_loss": analysis.get("stop_loss"),
            "take_profit": analysis.get("take_profit"),
            "order": None,
        }

        if not self.auto_trade or not analysis.get("approved"):
            return result

        side = analysis.get("proposed_side", "buy")
        qty = analysis["position_size"]

        order = self.client.submit_order(symbol=symbol, qty=qty, side=side)
        result["order"] = order
        return result

    # ── Pretty print ─────────────────────────────────────────

    @staticmethod
    def format_analysis(analysis: dict) -> str:
        """Return a human-readable summary of a risk assessment."""
        lines = [
            f"\n{'=' * 60}",
            f"  Risk Assessment — {analysis['symbol']}  "
            f"({analysis['proposed_side'].upper()})",
            f"  Portfolio exposure: {analysis['current_exposure_pct']:.1%}  |  "
            f"Position in {analysis['symbol']}: "
            f"{analysis['current_position_pct']:.1%}",
        ]

        # Add new metrics if available
        if "daily_pnl_pct" in analysis:
            lines.append(
                f"  Daily P&L: {analysis['daily_pnl_pct']:+.2%}  |  "
                f"Drawdown: {analysis['drawdown_pct']:.2%}"
            )
        if "sector" in analysis:
            lines.append(
                f"  Sector: {analysis['sector']} ({analysis['sector_exposure_pct']:.1%} exposure)"
            )

        lines.append(f"{'=' * 60}")

        for chk in analysis.get("checks", []):
            if isinstance(chk, dict):
                icon = "PASS" if chk["passed"] else "FAIL"
                lines.append(f"  [{icon}] {chk['rule']:20s}  {chk['detail']}")
            else:
                icon = "PASS" if chk.passed else "FAIL"
                lines.append(f"  [{icon}] {chk.rule:20s}  {chk.detail}")

        lines.append(f"{'─' * 60}")

        approved = analysis.get("approved", False)
        verdict = "APPROVED" if approved else "REJECTED"
        lines.append(f"  Verdict: {verdict}")

        if approved:
            lines.append(
                f"  Size: {analysis['position_size']} shares  |  "
                f"Stop: ${analysis['stop_loss']}  |  "
                f"Target: ${analysis['take_profit']}  |  "
                f"Risk: ${analysis['risk_amount']}"
            )

        lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)
