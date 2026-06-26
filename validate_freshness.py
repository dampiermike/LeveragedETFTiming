#!/usr/bin/env python3
"""
Freshness assertion for the TSMX daily JSON.

Checks that json/TSMX.json ends on the expected trading day. Exits 0 if current,
1 (with details) if stale. Called from run_daily.sh after download_data.py so a
stale file fires the failure-email trap before any signal is sent.

Expected trading day is resolved in this order:
  1. --expected-date YYYY-MM-DD CLI override
  2. yfinance SPY's last close (handles NYSE holidays) if yfinance is installed
  3. Last weekday on or before today (fallback, NO holiday awareness)

Usage:
    python3 validate_freshness.py
    python3 validate_freshness.py --expected-date 2026-06-25
    python3 validate_freshness.py --tolerance-days 1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
JSON_PATH = SCRIPT_DIR / "json" / "TSMX.json"


def _spy_last_close() -> pd.Timestamp | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        spy = yf.download("SPY", period="7d", auto_adjust=False, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = [c[0] for c in spy.columns]
        spy = spy.reset_index()
        if spy.empty:
            return None
        return pd.to_datetime(spy["Date"].max())
    except Exception:
        return None


def _last_weekday(ref: date | None = None) -> pd.Timestamp:
    d = ref or date.today()
    while d.weekday() >= 5:   # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return pd.Timestamp(d)


def expected_trading_day(cli_override: str | None) -> tuple[pd.Timestamp, str]:
    if cli_override:
        return pd.Timestamp(cli_override), "CLI override"
    spy = _spy_last_close()
    if spy is not None:
        return spy, "SPY (yfinance)"
    return _last_weekday(), "last weekday (fallback, no holiday awareness)"


def _last_bar_date(path: Path) -> pd.Timestamp | None:
    try:
        data = json.load(open(path))
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    try:
        return pd.to_datetime(max(r["date"] for r in data))
    except Exception:
        return None


def _weekdays_behind(last: pd.Timestamp, expected: pd.Timestamp) -> int:
    """Weekday count between last (exclusive) and expected (inclusive).
    Overcounts holidays — conservative."""
    if last >= expected:
        return 0
    cur, n = last + pd.Timedelta(days=1), 0
    while cur <= expected:
        if cur.weekday() < 5:
            n += 1
        cur += pd.Timedelta(days=1)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--expected-date", help="Override expected trading day (YYYY-MM-DD).")
    ap.add_argument("--tolerance-days", type=int, default=0,
                    help="Tolerance in (week)days behind. Default: 0")
    args = ap.parse_args()

    expected, source = expected_trading_day(args.expected_date)
    print("validate_freshness")
    print(f"  file:        {JSON_PATH}")
    print(f"  expected TD: {expected.date()}  (source: {source})")

    if not JSON_PATH.exists():
        print(f"  ❌ missing file: {JSON_PATH}", file=sys.stderr)
        return 1

    last = _last_bar_date(JSON_PATH)
    if last is None:
        print(f"  ❌ could not parse last bar date from {JSON_PATH.name}", file=sys.stderr)
        return 1

    behind = _weekdays_behind(last, expected)
    ok = behind <= args.tolerance_days
    print(f"  last bar:    {last.date()}   ({behind} weekday(s) behind, tol {args.tolerance_days})")
    if not ok:
        print(f"\n  ❌ FRESHNESS FAIL — TSMX.json is {behind} day(s) stale", file=sys.stderr)
        return 1
    print(f"\n  ✅ TSMX.json current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
