"""Tests for TradeJournal CSV read/write operations."""

import os
import tempfile
import unittest

from utils.trade_journal import TradeJournal


class TestTradeJournal(unittest.TestCase):
    """Test journal CSV creation, writing, and reading."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.journal = TradeJournal(journal_dir=self.tmpdir)

    def test_csv_files_created(self):
        """Journal init should create three CSV files with headers."""
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "signals.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "decisions.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "orders.csv")))

    def test_log_signal_and_read(self):
        """Writing a signal should be readable back."""
        self.journal.log_signal(
            symbol="AAPL",
            agent="TechnicalAnalysis",
            signal="BUY",
            score=0.75,
            price=150.0,
            detail="test signal",
        )
        rows = self.journal.read_signals(tail=5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "AAPL")
        self.assertEqual(rows[0]["agent"], "TechnicalAnalysis")
        self.assertEqual(rows[0]["signal"], "BUY")

    def test_log_decision_and_read(self):
        """Writing a decision should be readable back."""
        decision = {
            "symbol": "TSLA",
            "signal": "SELL",
            "combined_score": -0.35,
            "confidence": 0.35,
            "risk_approved": True,
            "position_size": 10,
            "stop_loss": 190.0,
            "take_profit": 210.0,
            "agent_signals": [
                {
                    "agent": "TechnicalAnalysis",
                    "normalised_score": -0.5,
                    "weight": 0.65,
                    "signal": "SELL",
                },
                {
                    "agent": "SentimentAnalysis",
                    "normalised_score": -0.1,
                    "weight": 0.35,
                    "signal": "HOLD",
                },
            ],
        }
        self.journal.log_decision(decision)
        rows = self.journal.read_decisions(tail=5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "TSLA")
        self.assertEqual(rows[0]["signal"], "SELL")

    def test_log_order_dict_and_read(self):
        """Order from a dict (real order response) should be logged."""
        order = {
            "id": "abc-123",
            "status": "filled",
            "type": "market",
        }
        self.journal.log_order(
            symbol="AAPL", side="buy", qty=10,
            order_result=order, reason="signal=BUY",
        )
        rows = self.journal.read_orders(tail=5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "AAPL")
        self.assertEqual(rows[0]["order_id"], "abc-123")
        self.assertEqual(rows[0]["status"], "filled")

    def test_log_order_skipped(self):
        """A skipped order (string result) should be logged."""
        self.journal.log_order(
            symbol="MSFT", side="sell", qty=5,
            order_result="skipped_no_position",
        )
        rows = self.journal.read_orders(tail=5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "skipped_no_position")

    def test_multiple_entries_append(self):
        """Multiple log calls should append, not overwrite."""
        for i in range(5):
            self.journal.log_signal(
                symbol=f"SYM{i}", agent="TA", signal="HOLD", score=0.0,
            )
        rows = self.journal.read_signals(tail=10)
        self.assertEqual(len(rows), 5)

    def test_read_tail_limit(self):
        """read_signals(tail=2) should return only the last 2 entries."""
        for i in range(5):
            self.journal.log_signal(
                symbol=f"SYM{i}", agent="TA", signal="HOLD", score=0.0,
            )
        rows = self.journal.read_signals(tail=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["symbol"], "SYM3")
        self.assertEqual(rows[1]["symbol"], "SYM4")


if __name__ == "__main__":
    unittest.main()
