import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration loaded from environment variables."""

    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
    )

    DEFAULT_SYMBOLS: list[str] = os.getenv(
        "DEFAULT_SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,TSLA"
    ).split(",")

    TRADING_MODE: str = os.getenv("TRADING_MODE", "paper")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls) -> bool:
        """Check that required keys are set."""
        if not cls.ALPACA_API_KEY or cls.ALPACA_API_KEY == "your_api_key_here":
            return False
        if not cls.ALPACA_SECRET_KEY or cls.ALPACA_SECRET_KEY == "your_secret_key_here":
            return False
        return True
