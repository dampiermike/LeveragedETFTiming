"""
Slope search across the ETF universe over a trailing 1-year window.

For each ticker we fit an OLS line to log(Adj Close) vs. a day index over the
window ending on --end (default: the most recent date in the data). From that
fit we report:

  - annualized slope: exp(daily_slope)**252 - 1, the smoothed annual growth the
    line implies (a clean momentum measure, a la Clenow).
  - R^2: how well a straight line explains the log-price path (trend quality).

Results are sorted by R^2 descending so the cleanest, most persistent trends
float to the top. --end is a variable so this can be re-run as of any date.

Run: python3 slope_search.py                 # most recent date, top 10
     python3 slope_search.py --end 2026-03-31 --top 20 --window 252
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from crossover_backtest import load_local

JSON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json")
TRADING_DAYS = 252


def all_tickers():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(JSON_DIR, "*.json")))


def fit_slope(close: pd.Series):
    """OLS of log(price) on a day index. Returns (annualized_slope, r2, n)."""
    y = np.log(close.to_numpy(dtype=float))
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fit = slope * x + intercept
    ss_res = np.sum((y - fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    annualized = np.exp(slope) ** TRADING_DAYS - 1
    return annualized, r2, len(y)


# Ranking keys -> the DataFrame column they sort on (all descending).
RANK_KEYS = {"r2": "r2", "slope": "annual_slope_pct", "slope_r2": "slope_r2"}

# Shape features used by the --like similarity search. These capture what a price
# chart "looks like" beyond raw return: trend persistence (r2, above50), growth
# pace (log_ann, on a non-explosive log scale), and drawdown character (maxdd,
# ulcer, near_high). Similarity is Euclidean distance in z-scored feature space.
SHAPE_FEATURES = ["log_ann", "r2", "maxdd", "ulcer", "near_high", "above50"]

# Match direction per feature: this is what makes a candidate "meet the criteria"
# rather than just be numerically near the target. For quality criteria we use a
# ONE-SIDED distance -- a candidate is penalized only when it's WORSE than the
# target; being better (smoother, shallower drawdown, nearer its highs) is free.
#   +1 = higher is better (penalize only when candidate < target)
#   -1 = lower is better  (penalize only when candidate > target)
#    0 = symmetric        (octane/growth: we want a similar pace, either way)
# This is why GGLL drops out: it's worse than TSMX on every quality axis, and a
# symmetric metric was masking that with its higher growth.
FEATURE_DIR = {"log_ann": 0, "r2": +1, "maxdd": +1, "ulcer": -1,
               "near_high": +1, "above50": +1}


def shape_profile(close: pd.Series) -> dict:
    """Compute the chart-shape feature set for one price series."""
    arr = close.to_numpy(dtype=float)
    y = np.log(arr)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fit = slope * x + intercept
    ss_res = np.sum((y - fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dd = arr / np.maximum.accumulate(arr) - 1          # drawdown path (<= 0)
    sma50 = close.rolling(50).mean()
    above50 = float((close > sma50).iloc[-60:].mean())  # uptrend-intact: frac of last 60d over 50d MA
    return {
        "log_ann": slope * TRADING_DAYS,                # log annual growth (stable, non-explosive)
        "annual_slope_pct": (np.exp(slope) ** TRADING_DAYS - 1) * 100,
        "r2": r2,
        "maxdd": float(dd.min()) * 100,                 # deepest drawdown %, closer to 0 is better
        "ulcer": float(np.sqrt(np.mean(dd ** 2))) * 100,  # drawdown pain (depth+duration)
        "near_high": float(arr[-1] / arr.max()),        # 1.0 = sitting at the window high
        "above50": above50,
    }


def profiles(end=None, window=TRADING_DAYS, min_bars=None) -> pd.DataFrame:
    """Shape profile for every ticker over the window. Indexed by ticker."""
    if min_bars is None:
        min_bars = int(window * 0.8)
    rows = []
    for ticker in all_tickers():
        try:
            px = load_local(ticker)
        except Exception:
            continue
        close = px["Adj Close"].dropna()
        if end is not None:
            close = close[close.index <= pd.Timestamp(end)]
        close = close.iloc[-window:]
        if len(close) < min_bars:
            continue
        p = shape_profile(close)
        p.update({"ticker": ticker, "bars": len(close),
                  "last_close": float(close.iloc[-1])})
        rows.append(p)
    return pd.DataFrame(rows).set_index("ticker")


def similar_to(target, end=None, window=TRADING_DAYS, min_bars=None) -> pd.DataFrame:
    """Rank every ticker by how closely its chart shape matches `target`'s, using a
    ONE-SIDED Euclidean distance in z-scored SHAPE_FEATURES space: quality criteria
    only count against a candidate when it's WORSE than the target (see FEATURE_DIR),
    so names that fail the criteria can't be rescued by extra growth. Returns a
    DataFrame with a `dist` column ascending (target itself is dist 0)."""
    df = profiles(end=end, window=window, min_bars=min_bars)
    if target not in df.index:
        raise SystemExit(f"--like target {target!r} not in qualifying universe "
                         f"(missing json or too few bars)")
    z = (df[SHAPE_FEATURES] - df[SHAPE_FEATURES].mean()) / df[SHAPE_FEATURES].std()
    diff = z.to_numpy() - z.loc[target].to_numpy()
    pen = np.empty_like(diff)
    for j, f in enumerate(SHAPE_FEATURES):
        d = FEATURE_DIR[f]
        if d == 0:                       # symmetric: any deviation counts
            pen[:, j] = diff[:, j]
        elif d > 0:                      # higher is better: penalize only when worse (lower)
            pen[:, j] = np.minimum(diff[:, j], 0.0)
        else:                            # lower is better: penalize only when worse (higher)
            pen[:, j] = np.maximum(diff[:, j], 0.0)
    df["dist"] = np.sqrt((pen ** 2).sum(axis=1))
    return df.sort_values("dist").reset_index()


def search(end=None, window=TRADING_DAYS, min_bars=None, rank_by="r2",
           min_slope=None, max_slope=None):
    """Compute slope/R^2 for every ticker over the `window` trading days ending
    on or before `end`. Returns a DataFrame sorted by `rank_by` descending.

    rank_by: 'r2' (trend smoothness), 'slope' (annualized growth), or 'slope_r2'
    (Clenow momentum = annualized slope x R^2 -> strong AND clean trends).

    min_slope/max_slope (percent, annualized) bound the slope band: the floor
    drops flat cash funds, the cap drops explosive/parabolic names. This is the
    'steady growth' lever -- keep believable growers, then rank by steadiness."""
    if min_bars is None:
        min_bars = int(window * 0.8)
    rows = []
    for ticker in all_tickers():
        try:
            px = load_local(ticker)
        except Exception as e:
            continue
        close = px["Adj Close"].dropna()
        if end is not None:
            close = close[close.index <= pd.Timestamp(end)]
        close = close.iloc[-window:]
        if len(close) < min_bars:
            continue
        ann, r2, n = fit_slope(close)
        rows.append({"ticker": ticker, "r2": r2, "annual_slope_pct": ann * 100,
                     "slope_r2": ann * 100 * r2,
                     "bars": n, "start": close.index[0].date(),
                     "end": close.index[-1].date(),
                     "last_close": float(close.iloc[-1])})
    df = pd.DataFrame(rows)
    if min_slope is not None:
        df = df[df["annual_slope_pct"] >= min_slope]
    if max_slope is not None:
        df = df[df["annual_slope_pct"] <= max_slope]
    col = RANK_KEYS[rank_by]
    df = df.sort_values(col, ascending=False).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--end", default=None,
                    help="anchor date (YYYY-MM-DD); default = most recent date available")
    ap.add_argument("--window", type=int, default=TRADING_DAYS,
                    help="trailing window length in trading days (default 252 = ~1y)")
    ap.add_argument("--top", type=int, default=10, help="how many rows to show")
    ap.add_argument("--rank-by", choices=list(RANK_KEYS), default="r2",
                    help="sort key: r2 (smoothness), slope (growth), "
                         "slope_r2 (Clenow momentum = slope x R^2)")
    ap.add_argument("--min-slope", type=float, default=None,
                    help="floor on annualized slope %% (drops flat cash funds)")
    ap.add_argument("--max-slope", type=float, default=None,
                    help="cap on annualized slope %% (drops explosive/parabolic names)")
    ap.add_argument("--min-bars", type=int, default=None,
                    help="skip tickers with fewer than this many bars (default 80%% of window)")
    ap.add_argument("--like", default=None, metavar="TICKER",
                    help="similarity mode: rank ETFs whose chart SHAPE most resembles "
                         "TICKER (trend persistence + drawdown character), not raw return")
    ap.add_argument("--out", default=None, help="optional CSV path for the full ranking")
    args = ap.parse_args()

    if args.like:
        run_like(args)
        return

    df = search(end=args.end, window=args.window, min_bars=args.min_bars,
                rank_by=args.rank_by, min_slope=args.min_slope, max_slope=args.max_slope)
    if args.out:
        df.to_csv(args.out, index=False)

    anchor = args.end or "latest"
    print("=" * 80)
    print(f"Slope search  |  window {args.window} trading days  |  anchor {anchor}")
    print(f"Ranked by {args.rank_by} descending  |  {len(df)} tickers qualified  |  top {args.top}")
    print("=" * 80)
    print(f"{'#':>2}  {'TICKER':<8}{'R^2':>8}{'ANN.SLOPE%':>12}{'SLOPExR2':>10}{'BARS':>6}  "
          f"{'WINDOW':<24}{'CLOSE':>9}")
    print("-" * 80)
    for i, r in df.head(args.top).iterrows():
        win = f"{r['start']}->{r['end']}"
        print(f"{i + 1:>2}  {r['ticker']:<8}{r['r2']:>8.3f}{r['annual_slope_pct']:>12.1f}"
              f"{r['slope_r2']:>10.1f}{int(r['bars']):>6}  {win:<24}{r['last_close']:>9.2f}")
    print("=" * 80)


def run_like(args):
    target = args.like.upper()
    df = similar_to(target, end=args.end, window=args.window, min_bars=args.min_bars)
    if args.out:
        df.to_csv(args.out, index=False)

    anchor = args.end or "latest"
    print("=" * 88)
    print(f"Shape match to {target}  |  window {args.window} trading days  |  anchor {anchor}")
    print(f"Ranked by profile distance (smaller = more alike)  |  {len(df)} tickers qualified")
    print("=" * 88)
    print(f"{'#':>2}  {'TICKER':<8}{'DIST':>6}{'ANN%':>7}{'R^2':>7}{'MAXDD%':>8}"
          f"{'ULCER':>7}{'NEARHI':>8}{'ABV50':>7}{'CLOSE':>9}")
    print("-" * 88)
    for i, r in df.head(args.top + 1).iterrows():   # +1: row 0 is the target itself
        mark = " *" if r["ticker"] == target else ""
        print(f"{i:>2}  {r['ticker']:<8}{r['dist']:>6.2f}{r['annual_slope_pct']:>7.0f}"
              f"{r['r2']:>7.3f}{r['maxdd']:>8.1f}{r['ulcer']:>7.1f}{r['near_high']:>8.3f}"
              f"{r['above50']:>7.2f}{r['last_close']:>9.2f}{mark}")
    print("=" * 88)
    print("ANN%=annualized trend  R^2=trend smoothness  MAXDD%=deepest drawdown  "
          "ULCER=drawdown pain")
    print("NEARHI=close/window-high (1=at high)  ABV50=frac of last 60d above 50d MA  "
          "(* = target)")


if __name__ == "__main__":
    main()
