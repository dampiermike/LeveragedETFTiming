# TSMX SMA/WMA Crossover

Backtest a moving-average crossover on **TSMX** (Direxion Daily TSM Bull 2X Shares,
2x Taiwan Semiconductor). Long while the fast MA is above the slow MA, flat (cash)
otherwise. Long-only, next-day OPEN execution, costs on (1 bps commission + 5 bps
slippage). Self-contained on local EODHD json.

**Caveat:** TSMX began trading **2024-10-03**, so usable history is ~1.7y and the
slow-MA warmup eats into it further. Treat results as suggestive, not robust.

## Files
- `download_data.py` — pull TSMX daily history from EODHD into `json/` (needs `EODHD_API_TOKEN`).
- `crossover_backtest.py` — single run. `--fast / --slow / --fast-type / --slow-type {sma,ema,wma}`.
- `sweep_crossover.py` — grid over every {SMA,WMA} fast/slow type pair × fast × slow windows.

## Daily signal (live)
Locked config: **LONG TSMX while WMA5 ≥ WMA40, else FLAT**; signal at the close
fills at the next open. Mirrors the Nitro daily-signal process.
- `tsmx_daily_signal.py` — replays the locked engine on the latest json, reports
  tomorrow's pending action (ENTER/EXIT/HOLD/FLAT), emails it (Gmail SMTP) and
  texts it (iMessage/SMS via `osascript`). `--dry-run` prints without sending.
  Recipients + SMS numbers are constants at the top of the file.
- `validate_freshness.py` — asserts `json/TSMX.json` ends on the expected trading day.
- `run_daily.sh` — orchestrator: update json → freshness check → signal. Logs to
  `logs/`, emails a heartbeat on success / failure trace on any error. Schedule via
  cron/launchd after the close. Needs `GOOGLE_EMAIL`, `GOOGLE_APP_PASSWORD`,
  `EODHD_API_TOKEN` in `~/.bash_profile`.

## Run
```
python3 download_data.py
python3 sweep_crossover.py
python3 crossover_backtest.py --fast 5 --slow 40 --fast-type wma --slow-type wma
```

## Findings (sweep period 2025-02-28 → 2026-06-25, all combos aligned to a 100-day warmup)
B&H TSMX over this window is brutal to beat — it returned ~196% CAGR at Sharpe 1.76
(maxDD −43%). **No crossover beats buy-and-hold on raw return**; the only edge any
combo offers is cutting drawdown.

Best by Sharpe: `WMA/WMA 5/40` — CAGR 218%, maxDD −22% (vs −43% B&H), Sharpe 2.07,
6 trades. `SMA/WMA 15/30` is close (Sharpe 2.05). Mean Sharpe was nearly flat across
the four type pairs (1.49–1.60), so MA-type choice matters far less than window here.

Given the tiny sample (one bull leg), these are not tradeable conclusions — extend
the history (e.g. synthetic TSMX from 2x TSM returns) before trusting any combo.
