#!/usr/bin/env python3
"""Quick smoke test to verify Alpaca API connectivity."""

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings
from utils.alpaca_client import AlpacaClient


def test_connection():
    print("=" * 50)
    print("  Alpaca Paper Trading – Connection Test")
    print("=" * 50)

    # 1. Validate env vars
    print("\n[1/4] Checking environment variables...")
    if not Settings.validate():
        print("  FAIL: API keys not configured.")
        print("  Copy .env.example to .env and fill in your keys.")
        return False
    print("  OK: API keys found.")

    # 2. Authenticate
    print("\n[2/4] Authenticating with Alpaca...")
    try:
        client = AlpacaClient()
        account = client.get_account()
    except Exception as e:
        print(f"  FAIL: {e}")
        return False
    print(f"  OK: Account {account['id']} ({account['status']})")
    print(f"      Cash: ${account['cash']:,.2f}  |  Equity: ${account['equity']:,.2f}")

    # 3. Fetch market data
    print("\n[3/4] Fetching market data (AAPL, last 5 bars)...")
    try:
        bars = client.get_bars("AAPL", timeframe="1Day", limit=5)
        if bars.empty:
            print("  WARN: No bars returned (market may be closed).")
        else:
            print(f"  OK: Received {len(bars)} bars.")
            print(bars[["open", "high", "low", "close", "volume"]].tail().to_string())
    except Exception as e:
        print(f"  WARN: Could not fetch bars – {e}")

    # 4. Market clock
    print("\n[4/4] Checking market clock...")
    try:
        is_open = client.is_market_open()
        status = "OPEN" if is_open else "CLOSED"
        print(f"  Market is currently: {status}")
    except Exception as e:
        print(f"  WARN: {e}")

    print("\n" + "=" * 50)
    print("  Connection test PASSED")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
