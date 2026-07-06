"""
Multi-ETF WMA/SMA crossover portfolio backtest, close-only execution.

Strategy
--------
Universe: every ETF in ./json (re-downloaded from inception).
Per ETF signal:  WMA(5) of close vs SMA(40) of close.
  - BULLISH state: WMA5 >= SMA40.
  - ENTRY:  on the day WMA5 crosses ABOVE SMA40 (fresh cross up).
  - EXIT:   on the day WMA5 crosses BELOW SMA40 (fresh cross down).
Execution: at the CLOSE of the signal day (close-only; no next-open fill).

Portfolio:
  - Hold at most 5 ETFs at once, each sized 20% of total equity at entry
    (funded from cash, capped by available cash so we never lever).
  - If more than (free slots) ETFs cross up on the same day, pick the ones whose
    1-year chart SHAPE most resembles TSMX -- the one-sided similarity from
    slope_search (closest dist wins). This ties selection to the --like tool.
  - Fewer than 5 signals -> fewer positions; the rest stays in cash.

Re-entry is only on a FRESH cross up: a slot that frees up is not back-filled
with an ETF that crossed earlier; it waits for the next crossover.

Run: python3 portfolio_backtest.py
     python3 portfolio_backtest.py --start 2011-01-01 --slots 5 --benchmark VOO
"""

import argparse

import numpy as np
import pandas as pd

from crossover_backtest import load_local, moving_average
from slope_search import (all_tickers, shape_profile, profiles,
                          SHAPE_FEATURES, FEATURE_DIR)

FAST, SLOW = 5, 40
FAST_TYPE, SLOW_TYPE = "wma", "sma"
SHAPE_WINDOW = 252        # trailing days for the TSMX-shape ranking
SHAPE_MIN_BARS = 60       # need at least this much history to rank a candidate's shape

# On-the-fly per-ETF optimization (--optimize): search WMA(fast) over SMA(slow),
# pick the pair with the best long/flat Sharpe on the ETF's trailing OPT_WINDOW.
OPT_WINDOW = 252                          # 1-year lookback for the optimization
OPT_FAST_GRID = [3, 5, 8, 10, 15, 20]
OPT_SLOW_GRID = [20, 30, 40, 50, 75, 100]
OPT_REOPT_EVERY = 21                      # re-optimize an idle ETF ~monthly
WCACHE = {f: (np.arange(1, f + 1, dtype=float) / (f * (f + 1) / 2)) for f in OPT_FAST_GRID}


def optimize_fast_slow(c):
    """Best (fast, slow) WMA/SMA pair by long/flat Sharpe over the window `c`
    (numpy adj-close array). Returns None if no pair has enough warmed-up data."""
    n = len(c)
    rets = c[1:] / c[:-1] - 1
    csum = np.concatenate([[0.0], np.cumsum(c)])
    best_sharpe, best = -1e18, None
    for slow in OPT_SLOW_GRID:
        if n < slow + 5:
            continue
        sma = (csum[slow:] - csum[:-slow]) / slow          # days slow-1 .. n-1
        for fast in OPT_FAST_GRID:
            if fast >= slow:
                continue
            wma_valid = np.convolve(c, WCACHE[fast][::-1], "valid")  # days fast-1 .. n-1
            wma = wma_valid[slow - fast:]                  # align to days slow-1 .. n-1
            state = (wma >= sma).astype(float)
            sret = state[:-1] * rets[slow - 1:]            # pos(t)=state(t-1) * ret(t)
            sd = sret.std()
            if sd > 0:
                sharpe = sret.mean() / sd * np.sqrt(252)
                if sharpe > best_sharpe:
                    best_sharpe, best = sharpe, (fast, slow)
    return best


def cross_states(c, i, fast, slow):
    """(bull_today, bull_prev) for adj-close array c at index i using WMA(fast)/SMA(slow)."""
    wf = WCACHE[fast]
    sma_i = c[i - slow + 1:i + 1].mean()
    sma_p = c[i - slow:i].mean()
    wma_i = c[i - fast + 1:i + 1] @ wf
    wma_p = c[i - fast:i] @ wf
    return (wma_i >= sma_i), (wma_p >= sma_p)


def build_signals():
    """For every ticker load adj close and compute WMA5/SMA40 state + crossover
    events. Returns (closes dict, up_events, dn_events) where the event maps are
    date -> [tickers crossing up/down that day]."""
    closes, up_events, dn_events = {}, {}, {}
    for t in all_tickers():
        try:
            px = load_local(t)
        except Exception:
            continue
        c = px["Adj Close"].dropna()
        if len(c) < SLOW + 2:
            continue
        fast = moving_average(c, FAST, FAST_TYPE)
        slow = moving_average(c, SLOW, SLOW_TYPE)
        ok = fast.notna() & slow.notna()
        bull = (fast >= slow)[ok]
        prev = bull.shift(1)
        cross_up = bull & (prev == False)
        cross_dn = (~bull) & (prev == True)
        closes[t] = c
        for d in c.index[ok][cross_up.to_numpy()]:
            up_events.setdefault(d, []).append(t)
        for d in c.index[ok][cross_dn.to_numpy()]:
            dn_events.setdefault(d, []).append(t)
    return closes, up_events, dn_events


def shape_reference(target="TSMX"):
    """Fixed scaler + target vector, taken from the current cross-section -- the
    same basis the live --like tool uses. Returns (mean, std, target_z)."""
    df = profiles(window=SHAPE_WINDOW)
    if target not in df.index:
        raise SystemExit(f"{target} profile unavailable -- cannot rank by {target} shape")
    mean = df[SHAPE_FEATURES].mean()
    std = df[SHAPE_FEATURES].std()
    target_z = ((df.loc[target, SHAPE_FEATURES] - mean) / std).to_numpy()
    return mean, std, target_z


def shape_dist_from(prof: dict, mean, std, target_z) -> float:
    """One-sided distance (slope_search semantics) from a precomputed profile to TSMX."""
    v = pd.Series(prof)[SHAPE_FEATURES]
    z = ((v - mean) / std).to_numpy()
    diff = z - target_z
    pen = np.empty_like(diff)
    for j, f in enumerate(SHAPE_FEATURES):
        d = FEATURE_DIR[f]
        pen[j] = diff[j] if d == 0 else (min(diff[j], 0.0) if d > 0 else max(diff[j], 0.0))
    return float(np.sqrt((pen ** 2).sum()))


def rank_candidates(cands, date, closes, ref, max_dist, min_slope):
    """Score entry candidates and GATE them on two criteria, so we only buy ETFs
    that both LOOK like TSMX and are actually CLIMBING:
      - TSMX-shape distance <= max_dist  (excludes inverse/bear/decaying funds)
      - annualized slope    >= min_slope (excludes smooth-but-flat bond/cash funds)
    Returns survivors closest-first. Too-short histories are excluded."""
    mean, std, target_z = ref
    scored = []
    for t in cands:
        win = closes[t].loc[:date].iloc[-SHAPE_WINDOW:]
        if len(win) < SHAPE_MIN_BARS:
            continue
        prof = shape_profile(win)
        if prof["annual_slope_pct"] < min_slope:
            continue
        dist = shape_dist_from(prof, mean, std, target_z)
        if dist <= max_dist:
            scored.append((dist, t))
    scored.sort()
    return [t for _, t in scored]


# Infrastructure tickers: used for cash yield / regime signal, never traded.
RESERVED = {"SPY", "BIL"}


def regime_series(ticker, ma, dates):
    """Boolean 'risk-on' series: ticker's close >= its `ma`-day SMA. NaN warmup -> True."""
    try:
        c = load_local(ticker)["Adj Close"].dropna()
    except Exception:
        return pd.Series(True, index=dates)
    on = (c >= c.rolling(ma).mean()).astype(float)   # float avoids object-dtype ffill warning
    return on.reindex(dates).ffill().fillna(1.0).astype(bool)


def cash_return_series(ticker, dates):
    """Daily return of a T-bill ETF, to accrue yield on idle cash. Missing -> 0."""
    try:
        c = load_local(ticker)["Adj Close"].dropna()
    except Exception:
        return pd.Series(0.0, index=dates)
    return c.pct_change().reindex(dates).fillna(0.0)


def run(start, end, slots, weight, cost_rate, initial_equity, benchmark, max_dist, min_slope,
        use_regime=False, use_cash_yield=False, regime_ticker="SPY", regime_ma=200,
        cash_ticker="BIL", target="TSMX"):
    closes, up_events, dn_events = build_signals()
    ref = shape_reference(target)

    # Master daily calendar + forward-filled price matrix for valuation.
    all_dates = sorted(set().union(*[c.index for c in closes.values()]))
    all_dates = [d for d in all_dates
                 if d >= pd.Timestamp(start) and (end is None or d <= pd.Timestamp(end))]
    price = pd.DataFrame({t: c for t, c in closes.items()}).reindex(all_dates).ffill()

    regime = regime_series(regime_ticker, regime_ma, all_dates) if use_regime \
        else pd.Series(True, index=all_dates)
    cash_ret = cash_return_series(cash_ticker, all_dates) if use_cash_yield \
        else pd.Series(0.0, index=all_dates)

    cash = initial_equity
    holds = {}                 # ticker -> {shares, entry_px, entry_dt}
    trades, curve = [], []

    for d in all_dates:
        cash *= 1 + cash_ret.at[d]          # accrue T-bill yield on idle cash

        # 1) EXITS at today's close (fresh cross down on a held name).
        for t in list(holds):
            if t in dn_events.get(d, []):
                px = price.at[d, t]
                if pd.isna(px):
                    continue
                fill = px * (1 - cost_rate)
                pos = holds.pop(t)
                cash += pos["shares"] * fill
                trades.append({"ticker": t, "entry_dt": pos["entry_dt"],
                               "entry_px": pos["entry_px"], "exit_dt": d, "exit_px": fill,
                               "return_pct": (fill / pos["entry_px"] - 1) * 100})

        # 2) ENTRIES at today's close (fresh cross up, not already held).
        #    Only when the broad market is risk-on (regime filter).
        cands = [t for t in up_events.get(d, [])
                 if t not in holds and t not in RESERVED and not pd.isna(price.at[d, t])]
        free = slots - len(holds)
        if cands and free > 0 and bool(regime.at[d]):
            for t in rank_candidates(cands, d, closes, ref, max_dist, min_slope)[:free]:
                equity = cash + sum(h["shares"] * price.at[d, k] for k, h in holds.items())
                spend = min(weight * equity, cash)
                if spend <= 0:
                    break
                fill = price.at[d, t] * (1 + cost_rate)
                holds[t] = {"shares": spend / fill, "entry_px": fill, "entry_dt": d}
                cash -= spend

        invested = sum(h["shares"] * price.at[d, k] for k, h in holds.items())
        curve.append({"date": d, "equity": cash + invested, "n_pos": len(holds),
                      "invested_frac": invested / (cash + invested) if (cash + invested) else 0})

    eq = pd.DataFrame(curve).set_index("date")
    tr = pd.DataFrame(trades)
    bh = benchmark_curve(benchmark, closes, eq.index, initial_equity)
    return eq, tr, bh


def run_optimized(start, end, slots, weight, cost_rate, initial_equity, benchmark,
                  max_dist, min_slope, reopt_every, target="TSMX"):
    """Like run(), but each ETF's WMA/SMA windows are OPTIMIZED on its trailing
    1-year history (best long/flat Sharpe) instead of a fixed 5/40. Params are
    cached and refreshed ~monthly while an ETF is idle; on entry the winning
    (fast, slow) is LOCKED and that same crossover governs the exit."""
    closes = {}
    arrs, posmap = {}, {}
    for t in all_tickers():
        try:
            c = load_local(t)["Adj Close"].dropna()
        except Exception:
            continue
        if len(c) < min(OPT_SLOW_GRID) + SHAPE_MIN_BARS:
            continue
        closes[t] = c
        arrs[t] = c.to_numpy(dtype=float)
        posmap[t] = {d: i for i, d in enumerate(c.index)}
    ref = shape_reference(target)

    all_dates = sorted(set().union(*[c.index for c in closes.values()]))
    all_dates = [d for d in all_dates
                 if d >= pd.Timestamp(start) and (end is None or d <= pd.Timestamp(end))]
    price = pd.DataFrame(closes).reindex(all_dates).ffill()

    cash = initial_equity
    holds = {}                 # ticker -> {shares, entry_px, entry_dt, fast, slow}
    opt_cache = {}             # ticker -> (opt_index, fast, slow)
    trades, curve = [], []

    for d in all_dates:
        # 1) EXITS: held name whose LOCKED crossover crosses down at today's close.
        for t in list(holds):
            i = posmap[t].get(d)
            if i is None:
                continue
            h = holds[t]
            bull, prev = cross_states(arrs[t], i, h["fast"], h["slow"])
            if (not bull) and prev:
                fill = price.at[d, t] * (1 - cost_rate)
                cash += h["shares"] * fill
                holds.pop(t)
                trades.append({"ticker": t, "entry_dt": h["entry_dt"], "entry_px": h["entry_px"],
                               "exit_dt": d, "exit_px": fill, "fast": h["fast"], "slow": h["slow"],
                               "return_pct": (fill / h["entry_px"] - 1) * 100})

        # 2) ENTRIES: scan idle ETFs; (re)optimize ~monthly, take fresh cross ups.
        free = slots - len(holds)
        cands = []
        if free > 0:
            for t in closes:
                if t in holds:
                    continue
                i = posmap[t].get(d)
                if i is None or i < max(OPT_SLOW_GRID):
                    continue
                cached = opt_cache.get(t)
                if cached is None or i - cached[0] >= reopt_every:
                    best = optimize_fast_slow(arrs[t][max(0, i - OPT_WINDOW + 1):i + 1])
                    if best is None:
                        continue
                    cached = (i, best[0], best[1])
                    opt_cache[t] = cached
                _, fast, slow = cached
                if i < slow:
                    continue
                bull, prev = cross_states(arrs[t], i, fast, slow)
                if bull and not prev:
                    cands.append(t)

        if cands and free > 0:
            for t in rank_candidates(cands, d, closes, ref, max_dist, min_slope)[:free]:
                equity = cash + sum(h["shares"] * price.at[d, k] for k, h in holds.items())
                spend = min(weight * equity, cash)
                if spend <= 0:
                    break
                fast, slow = opt_cache[t][1], opt_cache[t][2]
                fill = price.at[d, t] * (1 + cost_rate)
                holds[t] = {"shares": spend / fill, "entry_px": fill, "entry_dt": d,
                            "fast": fast, "slow": slow}
                cash -= spend

        invested = sum(h["shares"] * price.at[d, k] for k, h in holds.items())
        curve.append({"date": d, "equity": cash + invested, "n_pos": len(holds),
                      "invested_frac": invested / (cash + invested) if (cash + invested) else 0})

    eq = pd.DataFrame(curve).set_index("date")
    tr = pd.DataFrame(trades)
    bh = benchmark_curve(benchmark, closes, eq.index, initial_equity)
    return eq, tr, bh


def benchmark_curve(ticker, closes, dates, initial_equity):
    if ticker not in closes:
        return None
    p = closes[ticker].reindex(dates).ffill().dropna()
    if p.empty:
        return None
    return initial_equity * p / p.iloc[0]


def metrics(eq_series):
    e = eq_series.dropna()
    years = (e.index[-1] - e.index[0]).days / 365.25
    total = e.iloc[-1] / e.iloc[0] - 1
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")
    max_dd = (e / e.cummax() - 1).min()
    daily = e.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else float("nan")
    return dict(final=e.iloc[-1], total=total * 100, cagr=cagr * 100,
                maxdd=max_dd * 100, sharpe=sharpe, years=years)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--slots", type=int, default=5, help="max concurrent positions")
    ap.add_argument("--weight", type=float, default=0.20, help="fraction of equity per position")
    ap.add_argument("--commission-bps", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--initial-equity", type=float, default=100_000.0)
    ap.add_argument("--benchmark", default="VOO")
    ap.add_argument("--target", default="TSMX",
                    help="shape-similarity target ticker for the gate/ranking (default TSMX)")
    ap.add_argument("--max-dist", type=float, default=1.5,
                    help="only trade crossovers whose TSMX-shape distance <= this "
                         "(gates out inverse/bear/junk; use a large value to disable)")
    ap.add_argument("--min-slope", type=float, default=20.0,
                    help="annualized slope %% floor on entries (excludes smooth-but-flat "
                         "bond/cash funds; 0 to disable)")
    ap.add_argument("--regime-filter", action="store_true",
                    help="only enter when SPY is above its 200-day SMA (risk-on filter)")
    ap.add_argument("--cash-yield", action="store_true",
                    help="accrue T-bill (BIL) yield on idle cash instead of 0%%")
    ap.add_argument("--optimize", action="store_true",
                    help="optimize each ETF's WMA/SMA windows on its trailing 1y by Sharpe "
                         "instead of using a fixed 5/40 crossover")
    ap.add_argument("--reopt-every", type=int, default=OPT_REOPT_EVERY,
                    help="trading days between re-optimizations of an idle ETF")
    ap.add_argument("--out-prefix", default="portfolio_backtest")
    args = ap.parse_args()

    cost_rate = (args.commission_bps + args.slippage_bps) / 1e4
    if args.optimize:
        eq, tr, bh = run_optimized(args.start, args.end, args.slots, args.weight, cost_rate,
                                   args.initial_equity, args.benchmark, args.max_dist,
                                   args.min_slope, args.reopt_every, target=args.target)
    else:
        eq, tr, bh = run(args.start, args.end, args.slots, args.weight, cost_rate,
                         args.initial_equity, args.benchmark, args.max_dist, args.min_slope,
                         use_regime=args.regime_filter, use_cash_yield=args.cash_yield,
                         target=args.target)

    eq.to_csv(f"{args.out_prefix}_equity.csv")
    tr.to_csv(f"{args.out_prefix}_trades.csv", index=False)

    m = metrics(eq["equity"])
    sig = ("per-ETF optimized WMA/SMA (1y Sharpe)" if args.optimize
           else f"fixed WMA{FAST}/SMA{SLOW}")
    print("=" * 64)
    print(f"Multi-ETF crossover portfolio  (close-only)  |  signal: {sig}")
    print(f"Period {eq.index[0].date()} -> {eq.index[-1].date()}  ({m['years']:.1f}y)  "
          f"| {args.slots} slots x {args.weight:.0%} | costs {args.commission_bps+args.slippage_bps:.0f}bps")
    print(f"Entry gate: {args.target}-shape dist <= {args.max_dist} AND slope >= {args.min_slope:.0f}%/yr"
          f"  |  ranked by shape dist")
    extras = []
    if args.regime_filter: extras.append("regime: SPY>200d-SMA")
    if args.cash_yield:    extras.append("cash earns BIL yield")
    if extras: print("Add-ons: " + "  |  ".join(extras))
    print("=" * 64)
    hdr = f"{'':<20}{'STRATEGY':>14}"
    if bh is not None:
        hdr += f"{'B&H '+args.benchmark:>14}"
    print(hdr)
    bm = metrics(bh) if bh is not None else None
    for label, key, fmt in [("Final equity ($)", "final", "{:,.0f}"),
                            ("Total return (%)", "total", "{:,.1f}"),
                            ("CAGR (%)", "cagr", "{:.1f}"),
                            ("Max drawdown (%)", "maxdd", "{:.1f}"),
                            ("Sharpe", "sharpe", "{:.2f}")]:
        row = f"{label:<20}{fmt.format(m[key]):>14}"
        if bm:
            row += f"{fmt.format(bm[key]):>14}"
        print(row)
    print("-" * 64)
    wins = (tr["return_pct"] > 0).sum() if len(tr) else 0
    print(f"{'# trades':<20}{len(tr):>14}")
    print(f"{'Win rate (%)':<20}{(wins/len(tr)*100 if len(tr) else float('nan')):>14.1f}")
    print(f"{'Avg trade (%)':<20}{(tr['return_pct'].mean() if len(tr) else float('nan')):>14.2f}")
    print(f"{'Avg # positions':<20}{eq['n_pos'].mean():>14.2f}")
    print(f"{'Avg invested (%)':<20}{eq['invested_frac'].mean()*100:>14.1f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
