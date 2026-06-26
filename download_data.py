"""
Download TSMX (Direxion Daily TSM Bull 2X Shares) daily history from the EODHD
end-of-day API into ./json/. EODHD returns records already in the format read by
load_local() in crossover_backtest.py:
  {date, open, high, low, close, adjusted_close, volume}
so this project is self-contained.

NOTE: TSMX began trading 2024-10-03, so history is short (~1.7y as of 2026-06).

Requires EODHD_API_TOKEN in the environment (it's in ~/.bash_profile).

Run: python3 download_data.py                 # 2024-01-01 -> today
     python3 download_data.py 2024-01-01 2026-06-25
"""

import json
import os
import sys

import requests

TICKERS = ["TSMX"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json")
START = sys.argv[1] if len(sys.argv) > 1 else "2024-01-01"
END = sys.argv[2] if len(sys.argv) > 2 else None   # None = latest
EXCHANGE = "US"

TOKEN = os.environ.get("EODHD_API_TOKEN")
EXPECTED = {"date", "open", "high", "low", "close", "adjusted_close", "volume"}


def fetch(ticker):
    params = {"api_token": TOKEN, "fmt": "json", "from": START}
    if END:
        params["to"] = END
    r = requests.get(f"https://eodhd.com/api/eod/{ticker}.{EXCHANGE}", params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"unexpected response: {str(data)[:120]}")
    missing = EXPECTED - set(data[0])
    if missing:
        raise RuntimeError(f"missing fields {missing}")
    return data


def main():
    if not TOKEN:
        sys.exit("ERROR: EODHD_API_TOKEN not set in environment "
                 "(it's in ~/.bash_profile; run from a login shell or `source ~/.bash_profile`).")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading {len(TICKERS)} tickers from EODHD -> {OUT_DIR}  "
          f"(from {START} to {END or 'latest'})\n")
    failures = []
    for t in TICKERS:
        try:
            rec = fetch(t)
        except Exception as e:
            print(f"  {t:<6} FAILED: {e}")
            failures.append(t)
            continue
        with open(os.path.join(OUT_DIR, f"{t}.json"), "w") as f:
            json.dump(rec, f)
        print(f"  {t:<6} {len(rec):>5} rows  {rec[0]['date']} -> {rec[-1]['date']}")
    print("\nDone." if not failures else f"\nDone with failures: {failures}")


if __name__ == "__main__":
    main()
