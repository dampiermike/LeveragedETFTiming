"""
Download daily end-of-day history for the whole ETF universe listed in etfs.txt
from the EODHD end-of-day API into ./json/. EODHD returns records already in the
format read by load_local() in crossover_backtest.py:
  {date, open, high, low, close, adjusted_close, volume}

Each ticker is written to ./json/<TICKER>.json, overwriting any existing file.

Requires EODHD_API_TOKEN in the environment (it's in ~/.bash_profile).

Run: python3 download_universe.py                 # 2024-01-01 -> latest
     python3 download_universe.py 2024-01-01 2026-06-25
"""

import json
import os
import sys
import time

import config_loader  # noqa: F401  -- loads config.env into os.environ (must be first)
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_FILE = os.path.join(HERE, "etfs.txt")
OUT_DIR = os.path.join(HERE, "json")
START = sys.argv[1] if len(sys.argv) > 1 else "2024-01-01"
END = sys.argv[2] if len(sys.argv) > 2 else None   # None = latest
EXCHANGE = "US"

TOKEN = os.environ.get("EODHD_API_TOKEN")
EXPECTED = {"date", "open", "high", "low", "close", "adjusted_close", "volume"}


def load_universe():
    with open(UNIVERSE_FILE) as f:
        # one ticker per line; skip blanks and comments, dedupe preserving order
        seen, tickers = set(), []
        for line in f:
            t = line.strip().upper()
            if not t or t.startswith("#") or t in seen:
                continue
            seen.add(t)
            tickers.append(t)
    return tickers


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
    tickers = load_universe()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading {len(tickers)} tickers from EODHD -> {OUT_DIR}  "
          f"(from {START} to {END or 'latest'})\n")
    failures = []
    for i, t in enumerate(tickers, 1):
        try:
            rec = fetch(t)
        except Exception as e:
            print(f"  [{i:>3}/{len(tickers)}] {t:<6} FAILED: {e}")
            failures.append(t)
            continue
        with open(os.path.join(OUT_DIR, f"{t}.json"), "w") as f:
            json.dump(rec, f)
        print(f"  [{i:>3}/{len(tickers)}] {t:<6} {len(rec):>5} rows  "
              f"{rec[0]['date']} -> {rec[-1]['date']}")
        time.sleep(0.1)  # be gentle on the API
    print("\nDone." if not failures
          else f"\nDone with {len(failures)} failures: {failures}")


if __name__ == "__main__":
    main()
