"""
Daily crossover screener — lists ETFs that just crossed WMA(5) ABOVE SMA(40)
on the latest close AND look like the target (default TECL) trending shape.

For every ETF in ./json it computes, on DIVIDEND-ADJUSTED closes (same basis as
the backtest; enable "Adjust data for dividends" on TradingView to match):
  - a fresh bullish crossover: WMA5 crossed >= SMA40 within the last --within
    completed trading days (default 1 = crossed on the most recent close),
  - the entry gates: target-shape distance <= --max-dist AND annualized slope
    >= --min-slope (so we skip inverse/bear junk and smooth-but-flat funds).
It also reports the market regime (SPY vs its 200-day SMA); when risk-off, the
strategy would NOT act on these, and the report says so.

Sends an email + iMessage/SMS (same plumbing as tsmx_daily_signal.py). Run after
the close, after the universe json has been refreshed.

Run: python3 daily_crossover_signal.py                 # screen + send
     python3 daily_crossover_signal.py --dry-run        # print only
     python3 daily_crossover_signal.py --target TQQQ --within 2
"""

import argparse
import glob
import os

import pandas as pd

from crossover_backtest import load_local, moving_average
from slope_search import shape_profile
from portfolio_backtest import (shape_reference, shape_dist_from, RESERVED,
                                FAST, SLOW, FAST_TYPE, SLOW_TYPE)
from tsmx_daily_signal import send_email, send_imessage, TO_EMAIL, SMS_NUMBERS

REGIME_TICKER, REGIME_MA = "SPY", 200
SHAPE_MIN_BARS = 60
JSON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json")


def market_regime(asof):
    c = load_local(REGIME_TICKER)["Adj Close"].dropna()
    c = c[c.index <= asof]
    sma = c.rolling(REGIME_MA).mean()
    on = bool(c.iloc[-1] >= sma.iloc[-1])
    return on, float(c.iloc[-1]), float(sma.iloc[-1])


def screen(target, max_dist, min_slope, within, asof):
    """Return a DataFrame of ETFs whose WMA5 crossed >= SMA40 within `within`
    completed closes and which pass the target-shape + slope gates."""
    ref = shape_reference(target)
    rows = []
    for p in sorted(glob.glob(os.path.join(JSON_DIR, "*.json"))):
        t = os.path.basename(p)[:-5]
        if t in RESERVED:
            continue
        try:
            c = load_local(t)["Adj Close"].dropna()
        except Exception:
            continue
        c = c[c.index <= asof]
        if len(c) < max(SLOW + 2, SHAPE_MIN_BARS):
            continue
        fast = moving_average(c, FAST, FAST_TYPE)
        slow = moving_average(c, SLOW, SLOW_TYPE)
        ok = fast.notna() & slow.notna()
        if not ok.any() or fast.iloc[-1] < slow.iloc[-1]:
            continue                                   # must be bullish now
        state = (fast[ok] >= slow[ok]).to_numpy()
        idx = c.index[ok]
        cu = len(state) - 1
        for i in range(len(state) - 1, 0, -1):
            if state[i] and not state[i - 1]:
                cu = i
                break
        days = len(state) - 1 - cu
        if days >= within:                             # crossed too long ago
            continue
        prof = shape_profile(c.iloc[-252:])
        if prof["annual_slope_pct"] < min_slope:
            continue
        dist = shape_dist_from(prof, *ref)
        if dist > max_dist:
            continue
        rows.append({"ticker": t, "dist": dist, "slope": prof["annual_slope_pct"],
                     "r2": prof["r2"], "cross_date": idx[cu].date(), "days": days,
                     "gap_pct": (fast.iloc[-1] / slow.iloc[-1] - 1) * 100,
                     "close": float(c.iloc[-1])})
    return pd.DataFrame(rows).sort_values("dist").reset_index(drop=True)


def build_report(df, target, regime, asof_str, max_dist, min_slope):
    on, spy, spy_sma = regime
    lines = [
        f"Daily WMA{FAST}/SMA{SLOW} Crossover Screen — {asof_str}",
        "=" * 56,
        f"Target shape : {target}   (dist <= {max_dist}, slope >= {min_slope:.0f}%/yr)",
        f"Market regime: SPY {spy:.0f} vs {REGIME_MA}d-SMA {spy_sma:.0f}  ->  "
        f"{'RISK-ON (act on signals)' if on else 'RISK-OFF (hold — do not enter)'}",
        "",
    ]
    if df.empty:
        lines += ["No ETFs crossed WMA above SMA on the latest close.", ""]
    else:
        lines.append(f"{len(df)} new bullish crossover(s):")
        lines.append(f"  {'TICKER':<7}{'DIST':>6}{'SLOPE%':>8}{'R2':>6}{'GAP%':>7}{'CLOSE':>10}")
        lines.append("  " + "-" * 44)
        for _, r in df.iterrows():
            lines.append(f"  {r['ticker']:<7}{r['dist']:>6.2f}{r['slope']:>8.0f}"
                         f"{r['r2']:>6.2f}{r['gap_pct']:>7.1f}{r['close']:>10.2f}")
        lines.append("")
    lines += ["=" * 56,
              "Signals on dividend-ADJUSTED closes (TradingView: enable 'Adjust data for dividends')."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="TECL", help="shape-similarity target (default TECL)")
    ap.add_argument("--max-dist", type=float, default=1.5)
    ap.add_argument("--min-slope", type=float, default=20.0)
    ap.add_argument("--within", type=int, default=1,
                    help="crossed within this many completed closes (1 = on the latest close)")
    ap.add_argument("--dry-run", action="store_true", help="print only; no email/SMS")
    ap.add_argument("--email-only", action="store_true",
                    help="send email via SMTP but skip iMessage/SMS (for non-Mac/cloud runs)")
    args = ap.parse_args()

    asof = load_local(REGIME_TICKER)["Adj Close"].dropna().index[-1]
    asof_str = asof.strftime("%Y-%m-%d")
    regime = market_regime(asof)
    df = screen(args.target, args.max_dist, args.min_slope, args.within, asof)

    body = build_report(df, args.target, regime, asof_str, args.max_dist, args.min_slope)
    tickers = ", ".join(df["ticker"]) if not df.empty else "none"
    subject = f"Crossover Screen {asof_str}: {len(df)} new ({args.target})"
    sms = f"{asof_str[5:]} WMA>SMA ({args.target}): {tickers}"[:160]

    print(body)
    print(f"\nSUBJECT: {subject}\nSMS    : {sms}")

    if args.dry_run:
        print("\n[dry-run] no email/SMS sent.")
        return
    send_email(subject, body)
    if not args.email_only:
        send_imessage(SMS_NUMBERS, sms)


if __name__ == "__main__":
    main()
