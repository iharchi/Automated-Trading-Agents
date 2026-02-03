import logging
from abc import ABC, abstractmethod

from utils.alpaca_client import AlpacaClient


class BaseAgent(ABC):
    """Abstract base for every trading agent in the system.

    Subclasses must implement ``analyze`` and ``execute``.  The public
    ``run`` method calls them in sequence, giving a consistent lifecycle
    across all agents.
    """

    def __init__(self, name: str, client: AlpacaClient | None = None):
        self.name = name
        self.client = client or AlpacaClient()
        self.logger = logging.getLogger(f"agent.{name}")

    # ── Lifecycle ────────────────────────────────────────────

    def run(self, symbol: str, **kwargs) -> dict:
        """Analyse a symbol and, if a signal is produced, execute on it.

        Returns a result dict with at least ``{"symbol", "signal"}``.
        """
        self.logger.info("[%s] Running analysis on %s", self.name, symbol)
        analysis = self.analyze(symbol, **kwargs)
        result = self.execute(symbol, analysis, **kwargs)
        self.logger.info("[%s] Result for %s: %s", self.name, symbol, result)
        return result

    # ── Must override ────────────────────────────────────────

    @abstractmethod
    def analyze(self, symbol: str, **kwargs) -> dict:
        """Return an analysis dict (indicators, scores, etc.)."""

    @abstractmethod
    def execute(self, symbol: str, analysis: dict, **kwargs) -> dict:
        """Decide on and optionally place an order. Return result dict."""
