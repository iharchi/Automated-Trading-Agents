import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# ── Load YAML config (if it exists) ─────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
_yaml_cfg: dict = {}

if _CONFIG_PATH.exists():
    with open(_CONFIG_PATH) as f:
        _yaml_cfg = yaml.safe_load(f) or {}


def _y(section: str, key: str, default=None):
    """Read a value from the YAML config: _y('risk_management', 'max_position_pct')."""
    return _yaml_cfg.get(section, {}).get(key, default)


class Settings:
    """Central configuration loaded from config.yaml + environment variables.

    Precedence: env vars > config.yaml > hardcoded defaults.
    """

    # ── Alpaca API (always from env for security) ────────────
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv(
        "ALPACA_BASE_URL",
        _y("alpaca", "base_url", "https://paper-api.alpaca.markets"),
    )

    # ── General trading ──────────────────────────────────────
    DEFAULT_SYMBOLS: list[str] = os.getenv(
        "DEFAULT_SYMBOLS",
        ",".join(_y("trading", "default_symbols", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])),
    ).split(",")

    TRADING_MODE: str = os.getenv("TRADING_MODE", _y("trading", "mode", "paper"))
    TIMEFRAME: str = _y("trading", "timeframe", "1Day")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", _y("logging", "level", "INFO"))

    # ── Technical Analysis ───────────────────────────────────
    TA_RSI_PERIOD: int = int(_y("technical_analysis", "rsi_period", 14))
    TA_RSI_OVERSOLD: int = int(_y("technical_analysis", "rsi_oversold", 30))
    TA_RSI_OVERBOUGHT: int = int(_y("technical_analysis", "rsi_overbought", 70))
    TA_EMA_SHORT: int = int(_y("technical_analysis", "ema_short", 9))
    TA_EMA_LONG: int = int(_y("technical_analysis", "ema_long", 21))
    TA_BB_PERIOD: int = int(_y("technical_analysis", "bb_period", 20))
    TA_BB_STD: int = int(_y("technical_analysis", "bb_std", 2))
    TA_STOCHASTIC_PERIOD: int = int(_y("technical_analysis", "stochastic_period", 14))
    TA_STOCHASTIC_SMOOTH: int = int(_y("technical_analysis", "stochastic_smooth", 3))
    TA_STOCHASTIC_OVERSOLD: int = int(_y("technical_analysis", "stochastic_oversold", 20))
    TA_STOCHASTIC_OVERBOUGHT: int = int(_y("technical_analysis", "stochastic_overbought", 80))
    TA_ADX_PERIOD: int = int(_y("technical_analysis", "adx_period", 14))
    TA_ADX_TREND_THRESHOLD: int = int(_y("technical_analysis", "adx_trend_threshold", 25))
    TA_BUY_THRESHOLD: int = int(_y("technical_analysis", "buy_threshold", 2))
    TA_SELL_THRESHOLD: int = int(_y("technical_analysis", "sell_threshold", -2))

    # ── Multi-Timeframe Analysis ─────────────────────────────
    MTF_ENABLED: bool = _y("multi_timeframe", "enabled", False)
    MTF_TIMEFRAMES: list[str] = _y("multi_timeframe", "timeframes", ["1Day", "1Hour"])
    MTF_AGREEMENT_MODE: str = _y("multi_timeframe", "agreement_mode", "unanimous")
    MTF_MIN_AGREEMENT: float = float(_y("multi_timeframe", "min_agreement", 0.5))

    # ── Sentiment Analysis ───────────────────────────────────
    SENT_NEWS_LIMIT: int = int(_y("sentiment_analysis", "news_limit", 20))
    SENT_LOOKBACK_DAYS: int = int(_y("sentiment_analysis", "lookback_days", 7))
    SENT_BULLISH_THRESHOLD: float = float(_y("sentiment_analysis", "bullish_threshold", 0.15))
    SENT_BEARISH_THRESHOLD: float = float(_y("sentiment_analysis", "bearish_threshold", -0.15))

    # ── Risk Management ──────────────────────────────────────
    RISK_MAX_POSITION_PCT: float = float(_y("risk_management", "max_position_pct", 0.10))
    RISK_MAX_EXPOSURE: float = float(_y("risk_management", "max_portfolio_exposure", 0.90))
    RISK_PER_TRADE_PCT: float = float(_y("risk_management", "risk_per_trade_pct", 0.02))
    RISK_ATR_STOP_MULT: float = float(_y("risk_management", "atr_stop_multiplier", 1.5))
    RISK_TP_RATIO: float = float(_y("risk_management", "take_profit_ratio", 2.0))

    # ── Portfolio Manager ────────────────────────────────────
    PM_TA_WEIGHT: float = float(_y("portfolio_manager", "ta_weight", 0.65))
    PM_SENTIMENT_WEIGHT: float = float(_y("portfolio_manager", "sentiment_weight", 0.35))
    PM_BUY_THRESHOLD: float = float(_y("portfolio_manager", "buy_threshold", 0.25))
    PM_SELL_THRESHOLD: float = float(_y("portfolio_manager", "sell_threshold", -0.25))

    # ── Scheduler ────────────────────────────────────────────
    SCHED_INTERVAL: int = int(_y("scheduler", "interval_minutes", 15))

    # ── Backtesting ──────────────────────────────────────────
    BT_INITIAL_CAPITAL: float = float(_y("backtest", "initial_capital", 100_000))
    BT_DAYS: int = int(_y("backtest", "days", 365))

    # ── Watchlist Scanner ───────────────────────────────────
    SCAN_DEFAULT_SOURCE: str = _y("scanner", "default_source", "SP500_TOP50")
    SCAN_TOP_N: int = int(_y("scanner", "top_n", 10))
    SCAN_SIGNAL_FILTER: str = _y("scanner", "signal_filter", "all")
    SCAN_MIN_SCORE: int = int(_y("scanner", "min_score", 2))
    SCAN_RATE_LIMIT: float = float(_y("scanner", "rate_limit_delay", 0.1))

    # ── Trailing Stops ──────────────────────────────────────
    TRAIL_ENABLED: bool = _y("trailing_stops", "enabled", True)
    TRAIL_DEFAULT_MODE: str = _y("trailing_stops", "default_mode", "percentage")
    TRAIL_PERCENTAGE: float = float(_y("trailing_stops", "trail_percentage", 5.0))
    TRAIL_ATR_MULTIPLIER: float = float(_y("trailing_stops", "trail_atr_multiplier", 2.0))
    TRAIL_FIXED_AMOUNT: float = float(_y("trailing_stops", "trail_fixed_amount", 5.0))
    TRAIL_STEP_THRESHOLD: float = float(_y("trailing_stops", "step_threshold", 2.0))
    TRAIL_AUTO_SYNC: bool = _y("trailing_stops", "auto_sync", True)
    TRAIL_PERSIST_STATE: bool = _y("trailing_stops", "persist_state", True)

    # ── Correlation Filter ──────────────────────────────────
    CORR_ENABLED: bool = _y("correlation", "enabled", True)
    CORR_THRESHOLD: float = float(_y("correlation", "threshold", 0.70))
    CORR_LOOKBACK_DAYS: int = int(_y("correlation", "lookback_days", 90))
    CORR_USE_SECTOR_GROUPS: bool = _y("correlation", "use_sector_groups", True)
    CORR_CACHE_TTL: int = int(_y("correlation", "cache_ttl_minutes", 60))

    # ── Position Sizer (Kelly) ────────────────────────────
    SIZER_KELLY_FACTOR: float = float(_y("position_sizer", "kelly_factor", 0.5))
    SIZER_MAX_POSITION_PCT: float = float(_y("position_sizer", "max_position_pct", 0.10))
    SIZER_MAX_PORTFOLIO_HEAT: float = float(_y("position_sizer", "max_portfolio_heat", 0.06))
    SIZER_ATR_RISK_MULT: float = float(_y("position_sizer", "atr_risk_multiplier", 1.5))
    SIZER_TP_RATIO: float = float(_y("position_sizer", "take_profit_ratio", 2.0))
    SIZER_DEFAULT_WIN_RATE: float = float(_y("position_sizer", "default_win_rate", 0.50))
    SIZER_DEFAULT_PAYOFF: float = float(_y("position_sizer", "default_payoff_ratio", 1.5))
    SIZER_VOL_TARGET: float = float(_y("position_sizer", "vol_target", 0.15))

    # ── Signal Aggregator ─────────────────────────────────
    AGG_TA_WEIGHT: float = float(_y("aggregator", "ta_weight", 0.40))
    AGG_SENTIMENT_WEIGHT: float = float(_y("aggregator", "sentiment_weight", 0.20))
    AGG_MTF_WEIGHT: float = float(_y("aggregator", "mtf_weight", 0.30))
    AGG_REGIME_WEIGHT: float = float(_y("aggregator", "regime_weight", 0.10))
    AGG_BUY_THRESHOLD: float = float(_y("aggregator", "buy_threshold", 0.25))
    AGG_SELL_THRESHOLD: float = float(_y("aggregator", "sell_threshold", -0.25))
    AGG_MIN_SOURCES: int = int(_y("aggregator", "min_sources", 1))
    AGG_REGIME_ADAPTIVE: bool = _y("aggregator", "regime_adaptive", True)
    AGG_AGREEMENT_BONUS: float = float(_y("aggregator", "agreement_bonus", 0.10))

    # ── Market Regime Detector ──────────────────────────────
    REGIME_ADX_THRESHOLD: float = float(_y("regime", "adx_trend_threshold", 25.0))
    REGIME_VOL_HIGH: float = float(_y("regime", "vol_high_threshold", 0.30))
    REGIME_BB_SQUEEZE_PCT: float = float(_y("regime", "bb_squeeze_percentile", 20.0))
    REGIME_EMA_SHORT: int = int(_y("regime", "ema_short_period", 9))
    REGIME_EMA_LONG: int = int(_y("regime", "ema_long_period", 50))
    REGIME_VOL_LOOKBACK: int = int(_y("regime", "vol_lookback", 20))
    REGIME_SLOPE_LOOKBACK: int = int(_y("regime", "slope_lookback", 5))

    # ── Event Bus ──────────────────────────────────────────
    EVENTBUS_ENABLED: bool = _y("event_bus", "enabled", True)
    EVENTBUS_STRICT: bool = _y("event_bus", "strict_mode", False)
    EVENTBUS_MAX_HISTORY: int = int(_y("event_bus", "max_history", 100))
    EVENTBUS_LOG_EVENTS: bool = _y("event_bus", "log_events", False)

    # ── Notifications ────────────────────────────────────────
    NOTIFY_ENABLED_EVENTS: list[str] = _y(
        "notifications", "enabled_events",
        ["signal", "order_placed", "order_filled", "order_failed"],
    )
    NOTIFY_MIN_SIGNAL_SCORE: float = float(
        _y("notifications", "min_signal_score", 0.25)
    )

    # Email
    NOTIFY_EMAIL_ENABLED: bool = (
        _yaml_cfg.get("notifications", {}).get("email", {}).get("enabled", False)
    )
    NOTIFY_SMTP_HOST: str = _yaml_cfg.get("notifications", {}).get("email", {}).get(
        "smtp_host", "smtp.gmail.com"
    )
    NOTIFY_SMTP_PORT: int = int(
        _yaml_cfg.get("notifications", {}).get("email", {}).get("smtp_port", 587)
    )
    NOTIFY_SMTP_USER: str = os.getenv("SMTP_USER", "")
    NOTIFY_SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    NOTIFY_EMAIL_FROM: str = _yaml_cfg.get("notifications", {}).get("email", {}).get(
        "from_address", ""
    )
    NOTIFY_EMAIL_TO: list[str] = _yaml_cfg.get("notifications", {}).get("email", {}).get(
        "to_addresses", []
    )

    # Slack
    NOTIFY_SLACK_ENABLED: bool = (
        _yaml_cfg.get("notifications", {}).get("slack", {}).get("enabled", False)
    )
    NOTIFY_SLACK_WEBHOOK: str = os.getenv("SLACK_WEBHOOK_URL", "")

    # Discord
    NOTIFY_DISCORD_ENABLED: bool = (
        _yaml_cfg.get("notifications", {}).get("discord", {}).get("enabled", False)
    )
    NOTIFY_DISCORD_WEBHOOK: str = os.getenv("DISCORD_WEBHOOK_URL", "")

    # Generic webhook
    NOTIFY_WEBHOOK_ENABLED: bool = (
        _yaml_cfg.get("notifications", {}).get("webhook", {}).get("enabled", False)
    )
    NOTIFY_WEBHOOK_URL: str = os.getenv("NOTIFY_WEBHOOK_URL", "")

    @classmethod
    def validate(cls) -> bool:
        """Check that required keys are set."""
        if not cls.ALPACA_API_KEY or cls.ALPACA_API_KEY == "your_api_key_here":
            return False
        if not cls.ALPACA_SECRET_KEY or cls.ALPACA_SECRET_KEY == "your_secret_key_here":
            return False
        return True
