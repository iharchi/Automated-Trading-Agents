"""Position Sizer with Kelly Criterion

Advanced position sizing that combines:
    1. Kelly Criterion — optimal fraction based on win rate + payoff ratio
    2. Half-Kelly / fractional Kelly — for conservative risk management
    3. Volatility scaling — adjust size using ATR or historical volatility
    4. Regime-aware adjustments — from Market Regime Detector
    5. Portfolio heat — total risk across all open positions

Kelly formula:
    f* = (p * b - q) / b
    where  p = win probability
           q = loss probability = 1 - p
           b = avg_win / avg_loss  (payoff ratio)

Usage:
    sizer = PositionSizer()
    result = sizer.calculate(
        symbol="AAPL",
        price=150.0,
        atr=2.5,
        equity=100_000,
        signal="BUY",
        win_rate=0.55,
        avg_win=3.50,
        avg_loss=2.00,
    )
    print(PositionSizer.format_result(result))
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Defaults ────────────────────────────────────────────────────

DEFAULT_KELLY_FRACTION = 0.5    # Half-Kelly (recommended)
DEFAULT_MAX_POSITION_PCT = 0.10  # Never exceed 10% of equity
DEFAULT_MAX_PORTFOLIO_HEAT = 0.06  # Max 6% of equity at risk
DEFAULT_ATR_RISK_MULT = 1.5    # Stop = entry - (ATR * multiplier)
DEFAULT_TAKE_PROFIT_RATIO = 2.0  # 2:1 reward-to-risk
DEFAULT_WIN_RATE = 0.50         # Assumed win rate if unknown
DEFAULT_PAYOFF_RATIO = 1.5     # Assumed avg_win/avg_loss if unknown
DEFAULT_VOL_TARGET = 0.15      # Annual vol target for vol-scaling


@dataclass
class SizingResult:
    """Full breakdown of a position-sizing calculation."""

    symbol: str
    signal: str = "HOLD"
    price: float = 0.0

    # Kelly inputs
    win_rate: float = 0.0
    payoff_ratio: float = 0.0
    kelly_fraction: float = 0.0     # Full Kelly f*
    applied_fraction: float = 0.0   # After kelly_factor scaling

    # Sizing outputs
    shares: int = 0
    dollar_amount: float = 0.0
    position_pct: float = 0.0       # Position as % of equity

    # Risk
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_per_share: float = 0.0
    total_risk: float = 0.0         # shares * risk_per_share
    risk_pct: float = 0.0           # total_risk as % of equity

    # Limits applied
    regime_factor: float = 1.0
    volatility_factor: float = 1.0
    heat_limited: bool = False
    concentration_limited: bool = False
    original_shares: int = 0        # Shares before limits

    # Context
    equity: float = 0.0
    portfolio_heat: float = 0.0     # Total risk % across positions
    regime: str = ""
    method: str = ""                # "kelly", "atr", "volatility"
    error: str = ""


class PositionSizer:
    """Advanced position sizing with Kelly Criterion and risk controls."""

    def __init__(
        self,
        *,
        kelly_factor: float = DEFAULT_KELLY_FRACTION,
        max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
        max_portfolio_heat: float = DEFAULT_MAX_PORTFOLIO_HEAT,
        atr_risk_mult: float = DEFAULT_ATR_RISK_MULT,
        take_profit_ratio: float = DEFAULT_TAKE_PROFIT_RATIO,
        default_win_rate: float = DEFAULT_WIN_RATE,
        default_payoff_ratio: float = DEFAULT_PAYOFF_RATIO,
        vol_target: float = DEFAULT_VOL_TARGET,
    ):
        self.kelly_factor = kelly_factor
        self.max_position_pct = max_position_pct
        self.max_portfolio_heat = max_portfolio_heat
        self.atr_risk_mult = atr_risk_mult
        self.take_profit_ratio = take_profit_ratio
        self.default_win_rate = default_win_rate
        self.default_payoff_ratio = default_payoff_ratio
        self.vol_target = vol_target

    # ── Kelly Criterion ───────────────────────────────────────────

    @staticmethod
    def kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
        """Calculate the full Kelly fraction.

        f* = (p * b - q) / b

        Returns:
            Fraction of bankroll to wager.  Negative means don't bet.
        """
        if payoff_ratio <= 0:
            return 0.0
        p = max(0.0, min(1.0, win_rate))
        q = 1.0 - p
        f_star = (p * payoff_ratio - q) / payoff_ratio
        return f_star

    # ── Volatility scaling ────────────────────────────────────────

    @staticmethod
    def volatility_scale(
        annual_vol: float, vol_target: float = DEFAULT_VOL_TARGET,
    ) -> float:
        """Scale position by target-vol / actual-vol.

        High vol → smaller position.  Low vol → larger position.
        Capped at [0.25, 2.0] to avoid extreme sizing.
        """
        if annual_vol <= 0:
            return 1.0
        factor = vol_target / annual_vol
        return max(0.25, min(2.0, factor))

    # ── Main calculation ──────────────────────────────────────────

    def calculate(
        self,
        symbol: str,
        *,
        price: float,
        equity: float,
        signal: str = "BUY",
        atr: float = 0.0,
        win_rate: float | None = None,
        avg_win: float | None = None,
        avg_loss: float | None = None,
        annual_vol: float | None = None,
        regime_adjustments: dict | None = None,
        portfolio_heat: float = 0.0,
    ) -> SizingResult:
        """Calculate position size using Kelly + risk controls.

        Args:
            symbol: Ticker symbol.
            price: Current share price.
            equity: Total account equity.
            signal: "BUY" or "SELL".
            atr: Average True Range (for stop-loss calc).
            win_rate: Historical win rate (0-1). Uses default if None.
            avg_win: Average winning trade in dollars per share.
            avg_loss: Average losing trade in dollars per share.
            annual_vol: Annualised volatility for vol-scaling.
            regime_adjustments: Dict from MarketRegimeDetector with
                position_size_factor, stop_multiplier, etc.
            portfolio_heat: Current total risk % across all positions.

        Returns:
            SizingResult with full breakdown.
        """
        result = SizingResult(symbol=symbol, signal=signal, price=price, equity=equity)

        if price <= 0 or equity <= 0:
            result.error = "Invalid price or equity"
            return result

        if signal not in ("BUY", "SELL"):
            result.error = f"Non-actionable signal: {signal}"
            return result

        # ── Resolve inputs ───────────────────────────────────────
        wr = win_rate if win_rate is not None else self.default_win_rate
        wr = max(0.0, min(1.0, wr))

        # Payoff ratio
        if avg_win is not None and avg_loss is not None and avg_loss > 0:
            payoff = avg_win / avg_loss
        else:
            payoff = self.default_payoff_ratio

        result.win_rate = round(wr, 4)
        result.payoff_ratio = round(payoff, 4)

        # ── Kelly fraction ───────────────────────────────────────
        full_kelly = self.kelly_fraction(wr, payoff)
        result.kelly_fraction = round(full_kelly, 4)

        if full_kelly <= 0:
            result.method = "kelly"
            result.error = "Negative Kelly: edge insufficient"
            return result

        applied = full_kelly * self.kelly_factor
        result.applied_fraction = round(applied, 4)
        result.method = "kelly"

        # ── Stop-loss and take-profit ────────────────────────────
        stop_mult = self.atr_risk_mult
        tp_ratio = self.take_profit_ratio

        regime_factor = 1.0
        regime_name = ""
        if regime_adjustments:
            regime_factor = regime_adjustments.get("position_size_factor", 1.0)
            stop_mult *= regime_adjustments.get("stop_multiplier", 1.0)
            tp_ratio *= regime_adjustments.get("take_profit_multiplier", 1.0)
            regime_name = regime_adjustments.get("description", "")

        result.regime_factor = round(regime_factor, 4)
        result.regime = regime_name

        if atr > 0:
            risk_per_share = atr * stop_mult
        else:
            # Fallback: 2% of price as risk
            risk_per_share = price * 0.02

        if signal == "BUY":
            result.stop_loss = round(price - risk_per_share, 2)
            tp_distance = risk_per_share * tp_ratio
            result.take_profit = round(price + tp_distance, 2)
        else:
            result.stop_loss = round(price + risk_per_share, 2)
            tp_distance = risk_per_share * tp_ratio
            result.take_profit = round(price - tp_distance, 2)

        result.risk_per_share = round(risk_per_share, 2)

        # ── Dollar amount from Kelly ─────────────────────────────
        kelly_dollars = equity * applied

        # ── Volatility scaling ───────────────────────────────────
        vol_factor = 1.0
        if annual_vol is not None and annual_vol > 0:
            vol_factor = self.volatility_scale(annual_vol, self.vol_target)
            result.method = "kelly+vol"
        result.volatility_factor = round(vol_factor, 4)

        # ── Apply factors ────────────────────────────────────────
        adjusted_dollars = kelly_dollars * regime_factor * vol_factor
        shares = int(adjusted_dollars / price) if price > 0 else 0
        shares = max(shares, 0)
        result.original_shares = shares

        # ── Concentration cap ────────────────────────────────────
        max_shares = int(equity * self.max_position_pct / price) if price > 0 else 0
        if shares > max_shares:
            shares = max_shares
            result.concentration_limited = True

        # ── Portfolio heat cap ───────────────────────────────────
        result.portfolio_heat = round(portfolio_heat, 4)
        remaining_heat = max(0, self.max_portfolio_heat - portfolio_heat)

        if risk_per_share > 0 and remaining_heat > 0:
            max_risk_dollars = equity * remaining_heat
            heat_shares = int(max_risk_dollars / risk_per_share)
            if shares > heat_shares:
                shares = heat_shares
                result.heat_limited = True
        elif remaining_heat <= 0:
            shares = 0
            result.heat_limited = True

        # ── Final result ─────────────────────────────────────────
        shares = max(shares, 0)
        result.shares = shares
        result.dollar_amount = round(shares * price, 2)
        result.position_pct = round(
            (shares * price) / equity if equity > 0 else 0, 4,
        )
        result.total_risk = round(shares * risk_per_share, 2)
        result.risk_pct = round(
            result.total_risk / equity if equity > 0 else 0, 4,
        )

        return result

    # ── Formatting ────────────────────────────────────────────────

    @staticmethod
    def format_result(result: "SizingResult | dict") -> str:
        """Human-readable position sizing summary."""
        if isinstance(result, dict):
            r = type("_R", (), result)()
        else:
            r = result

        lines = [
            f"\n{'=' * 62}",
            f"  POSITION SIZER -- {r.symbol}  ({r.signal})",
            f"  Price: ${r.price:,.2f}  |  Equity: ${r.equity:,.0f}",
            f"{'=' * 62}",
        ]

        if r.error:
            lines.append(f"  Error: {r.error}")
            lines.append(f"{'=' * 62}\n")
            return "\n".join(lines)

        lines.append(f"  Method         : {r.method}")
        lines.append(
            f"  Win Rate       : {r.win_rate:.0%}  |  "
            f"Payoff Ratio: {r.payoff_ratio:.2f}"
        )
        lines.append(
            f"  Kelly (full)   : {r.kelly_fraction:.2%}  |  "
            f"Applied ({r.applied_fraction / r.kelly_fraction * 100:.0f}%): "
            f"{r.applied_fraction:.2%}"
            if r.kelly_fraction > 0 else
            f"  Kelly (full)   : {r.kelly_fraction:.2%}"
        )
        lines.append(f"{'~' * 62}")
        lines.append(
            f"  Shares         : {r.shares:,}  "
            f"(${r.dollar_amount:,.0f} / {r.position_pct:.1%} of equity)"
        )
        lines.append(
            f"  Stop Loss      : ${r.stop_loss:,.2f}  |  "
            f"Take Profit: ${r.take_profit:,.2f}"
        )
        lines.append(
            f"  Risk per Share  : ${r.risk_per_share:,.2f}  |  "
            f"Total Risk: ${r.total_risk:,.0f} ({r.risk_pct:.2%})"
        )

        limits = []
        if r.concentration_limited:
            limits.append("concentration")
        if r.heat_limited:
            limits.append("portfolio-heat")
        if limits:
            lines.append(f"  Limits Applied  : {', '.join(limits)}")
            if r.original_shares != r.shares:
                lines.append(
                    f"  Original Shares : {r.original_shares:,} → {r.shares:,}"
                )

        if r.regime:
            lines.append(
                f"  Regime Factor   : {r.regime_factor:.2f}x  ({r.regime})"
            )
        if r.volatility_factor != 1.0:
            lines.append(f"  Vol Factor      : {r.volatility_factor:.2f}x")
        if r.portfolio_heat > 0:
            lines.append(f"  Portfolio Heat   : {r.portfolio_heat:.2%}")

        lines.append(f"{'=' * 62}\n")
        return "\n".join(lines)
