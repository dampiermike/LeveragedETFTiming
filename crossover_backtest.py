"""
SMA/WMA crossover backtest on TSMX (Direxion Daily TSM Bull 2X Shares).

  - LONG  TSMX when fast MA closes above slow MA.
  - FLAT  (cash) when fast MA closes below slow MA.

The two MAs can each be an SMA or a WMA (--fast-type / --slow-type), so this
covers SMA/SMA, WMA/WMA, and the mixed WMA-SMA / SMA-WMA crossovers. Long-only,
next-day OPEN execution (signal at close of t fills at open of t+1), costs on.

TSMX began trading 2024-10-03, so the usable backtest is short -- keep windows
modest (the slow MA must warm up before any signal exists).

Run: python3 crossover_backtest.py --fast 10 --slow 30 --fast-type wma --slow-type sma
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

JSON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json")


def load_local(ticker: str) -> pd.DataFrame:
    """Load json/<TICKER>.json (EODHD daily) -> Date-indexed OHLC + Adj Close, with
    Adj Open/High/Low on the adjusted scale."""
    df = pd.DataFrame(json.load(open(os.path.join(JSON_DIR, f"{ticker}.json"))))
    df["Date"] = pd.to_datetime(df["date"])
    df = df.set_index("Date").sort_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "adjusted_close": "Adj Close"})
    df = df[["Open", "High", "Low", "Close", "Adj Close"]].astype(float)
    df = df.dropna(subset=["Close", "Adj Close"])
    adj = df["Adj Close"] / df["Close"]
    for c in ("Open", "High", "Low"):
        df[f"Adj {c}"] = df[c] * adj
    return df


def moving_average(s: pd.Series, window: int, ma_type: str) -> pd.Series:
    if ma_type == "sma":
        return s.rolling(window).mean()
    if ma_type == "ema":
        return s.ewm(span=window, adjust=False, min_periods=window).mean()
    if ma_type == "wma":
        w = np.arange(1, window + 1, dtype=float)
        return s.rolling(window).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
    raise ValueError(f"unknown ma_type: {ma_type}")


def metrics(eq: pd.DataFrame, tr: pd.DataFrame) -> dict:
    e = eq["equity"]
    years = (e.index[-1] - e.index[0]).days / 365.25
    total = e.iloc[-1] / e.iloc[0] - 1
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")
    max_dd = (e / e.cummax() - 1).min()
    daily = e.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else float("nan")
    wins = (tr["return_pct"] > 0).sum() if len(tr) else 0
    return {"final_equity": e.iloc[-1], "total_return_pct": total * 100,
            "cagr_pct": cagr * 100, "max_drawdown_pct": max_dd * 100, "sharpe": sharpe,
            "exposure_pct": eq["pos"].mean() * 100, "num_trades": len(tr),
            "win_rate_pct": (wins / len(tr) * 100) if len(tr) else float("nan"),
            "avg_trade_pct": tr["return_pct"].mean() if len(tr) else float("nan"),
            "years": years}


def buy_hold(px: pd.Series, initial_equity: float) -> dict:
    years = (px.index[-1] - px.index[0]).days / 365.25
    total = px.iloc[-1] / px.iloc[0] - 1
    daily = px.pct_change().dropna()
    return {"final_equity": initial_equity * (1 + total), "total_return_pct": total * 100,
            "cagr_pct": ((px.iloc[-1] / px.iloc[0]) ** (1 / years) - 1) * 100,
            "max_drawdown_pct": (px / px.cummax() - 1).min() * 100,
            "sharpe": daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else float("nan")}


def run_crossover(df: pd.DataFrame, fast: int, slow: int, fast_type: str, slow_type: str,
                  cost_rate: float, initial_equity: float):
    """df needs columns: close (adj close, for MAs + B&H) and open (adj open, fills).
    Returns (equity_df, trades_df)."""
    fast_ma = moving_average(df["close"], fast, fast_type)
    slow_ma = moving_average(df["close"], slow, slow_type)
    work = df.copy()
    work["fast"] = fast_ma
    work["slow"] = slow_ma
    work = work[work["slow"].notna() & work["fast"].notna()]   # require both MAs warm

    # State at close of t (executed next open): long while fast >= slow, else flat.
    state = (work["fast"] >= work["slow"]).astype(int).to_numpy()
    work["state"] = state
    # Position held during day t is decided by the state at close of t-1.
    work["pos"] = pd.Series(state, index=work.index).shift(1, fill_value=0).astype(int)

    cash, shares = initial_equity, 0.0
    entry_px = entry_dt = None
    records, trades = [], []
    pos_arr = work["pos"].to_numpy()
    open_arr = work["open"].to_numpy()
    close_arr = work["close"].to_numpy()
    dates = work.index

    for i in range(len(work)):
        target, o = pos_arr[i], open_arr[i]
        if target == 1 and shares == 0.0:
            fill = o * (1 + cost_rate)
            shares, cash = cash / fill, 0.0
            entry_px, entry_dt = fill, dates[i]
        elif target == 0 and shares > 0.0:
            fill = o * (1 - cost_rate)
            cash = shares * fill
            trades.append({"entry_date": entry_dt, "entry_px": entry_px,
                           "exit_date": dates[i], "exit_px": fill,
                           "return_pct": (fill / entry_px - 1) * 100, "reason": "cross"})
            shares, entry_px, entry_dt = 0.0, None, None
        records.append({"date": dates[i], "equity": cash + shares * close_arr[i],
                        "pos": target, "tsmx": close_arr[i]})

    if shares > 0.0:
        trades.append({"entry_date": entry_dt, "entry_px": entry_px,
                       "exit_date": dates[-1], "exit_px": close_arr[-1],
                       "return_pct": (close_arr[-1] / entry_px - 1) * 100, "reason": "open_at_end"})

    eq = pd.DataFrame(records).set_index("date")
    tr = pd.DataFrame(trades)
    return eq, tr


def build_frame(start=None, end=None) -> pd.DataFrame:
    px = load_local("TSMX")
    df = pd.DataFrame(index=px.index)
    df["close"] = px["Adj Close"]
    df["open"] = px["Adj Open"]
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", type=int, default=10)
    ap.add_argument("--slow", type=int, default=30)
    ap.add_argument("--fast-type", choices=["sma", "ema", "wma"], default="wma")
    ap.add_argument("--slow-type", choices=["sma", "ema", "wma"], default="sma")
    ap.add_argument("--commission-bps", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--initial-equity", type=float, default=100_000.0)
    ap.add_argument("--out-prefix", default="crossover_tsmx")
    args = ap.parse_args()

    if args.fast >= args.slow:
        ap.error(f"--fast ({args.fast}) must be < --slow ({args.slow})")

    df = build_frame(args.start, args.end)
    cost_rate = (args.commission_bps + args.slippage_bps) / 1e4
    eq, tr = run_crossover(df, args.fast, args.slow, args.fast_type, args.slow_type,
                           cost_rate, args.initial_equity)
    m = metrics(eq, tr)
    bh = buy_hold(df["close"].reindex(eq.index), args.initial_equity)

    eq.to_csv(f"{args.out_prefix}_equity.csv")
    tr.to_csv(f"{args.out_prefix}_trades.csv", index=False)

    print("=" * 56)
    print(f"TSMX crossover: long when {args.fast_type.upper()}{args.fast} > "
          f"{args.slow_type.upper()}{args.slow}")
    print(f"Period: {eq.index[0].date()} -> {eq.index[-1].date()}  ({m['years']:.1f}y)")
    print("=" * 56)
    print(f"{'':<22}{'STRATEGY':>15}{'B&H TSMX':>15}")
    for label, key, fmt in [("Final equity ($)", "final_equity", "{:,.0f}"),
                            ("Total return (%)", "total_return_pct", "{:,.1f}"),
                            ("CAGR (%)", "cagr_pct", "{:.1f}"),
                            ("Max drawdown (%)", "max_drawdown_pct", "{:.1f}"),
                            ("Sharpe", "sharpe", "{:.2f}")]:
        print(f"{label:<22}{fmt.format(m[key]):>15}{fmt.format(bh[key]):>15}")
    print("-" * 56)
    print(f"{'Exposure (%)':<22}{m['exposure_pct']:>15.1f}")
    print(f"{'# trades':<22}{m['num_trades']:>15}")
    print(f"{'Win rate (%)':<22}{m['win_rate_pct']:>15.1f}")
    print(f"{'Avg trade (%)':<22}{m['avg_trade_pct']:>15.2f}")
    print("=" * 56)


if __name__ == "__main__":
    main()
