"""
Sweep SMA/WMA crossovers on TSMX: every (fast_type, slow_type) pair in
{SMA, WMA} x fast window x slow window (fast < slow), long while fast MA > slow MA.

TSMX has only ~1.7y of history (inception 2024-10-03), so windows are kept short
and all combos are aligned to a common start (past the longest slow-MA warmup) so
the grid is comparable.

Run: python3 sweep_crossover.py
"""

import itertools

import pandas as pd

from crossover_backtest import build_frame, run_crossover, metrics, buy_hold

FAST_TYPES = ["sma", "wma"]
SLOW_TYPES = ["sma", "wma"]
FAST_WINDOWS = [5, 10, 15, 20, 30]
SLOW_WINDOWS = [20, 30, 40, 50, 75, 100]
COST_RATE = (1.0 + 5.0) / 1e4    # commission + slippage bps
INITIAL_EQUITY = 100_000.0

df = build_frame()
# Align every combo to a common start so different slow windows are comparable:
# drop the warmup for the longest slow window used in the grid.
warmup = max(SLOW_WINDOWS)
common_start = df.index[warmup]
df = df[df.index >= common_start]

bh = buy_hold(df["close"], INITIAL_EQUITY)
period = f"{df.index[0].date()} -> {df.index[-1].date()}"

rows = []
for ftype, stype, fast, slow in itertools.product(FAST_TYPES, SLOW_TYPES, FAST_WINDOWS, SLOW_WINDOWS):
    if fast >= slow:
        continue
    eq, tr = run_crossover(df, fast, slow, ftype, stype, COST_RATE, INITIAL_EQUITY)
    if len(eq) == 0:
        continue
    m = metrics(eq, tr)
    rows.append({"pair": f"{ftype.upper()}/{stype.upper()}", "fast": fast, "slow": slow,
                 "CAGR%": round(m["cagr_pct"], 1), "maxDD%": round(m["max_drawdown_pct"], 1),
                 "Sharpe": round(m["sharpe"], 2), "expo%": round(m["exposure_pct"], 0),
                 "trades": m["num_trades"], "win%": round(m["win_rate_pct"], 0),
                 "finalEq": round(m["final_equity"])})

res = pd.DataFrame(rows)
res.to_csv("sweep_crossover_results.csv", index=False)

pd.set_option("display.width", 200, "display.max_rows", 400)
print(f"\nPeriod: {period}   (all combos aligned to a {warmup}-day warmup)")
print(f"Benchmark B&H TSMX: CAGR {bh['cagr_pct']:.1f}%  maxDD {bh['max_drawdown_pct']:.1f}%  "
      f"Sharpe {bh['sharpe']:.2f}\n")

print("=== TOP 15 by Sharpe ===")
print(res.sort_values("Sharpe", ascending=False).head(15).to_string(index=False))

print("\n=== TOP 15 by CAGR ===")
print(res.sort_values("CAGR%", ascending=False).head(15).to_string(index=False))

print("\n=== TOP 15 by lowest drawdown ===")
print(res.sort_values("maxDD%", ascending=False).head(15).to_string(index=False))

print("\n=== Mean Sharpe by MA-type pair ===")
print(res.pivot_table(index="pair", values="Sharpe", aggfunc="mean").round(2).to_string())

print(f"\nSaved full grid ({len(res)} rows) -> sweep_crossover_results.csv")
