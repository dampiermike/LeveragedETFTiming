#!/bin/bash
# Scheduled ETF crossover screener runner (cloud/sandbox friendly).
# Unlike run_daily.sh, this does NOT source ~/.bash_profile — secrets come from
# config.env (loaded by config_loader.py inside the Python scripts). It also uses
# --email-only (SMTP) because the scheduled sandbox is Linux with no iMessage.
#
# Steps: refresh the full ETF universe from EODHD -> run the WMA5/SMA40 screen
# and email the report. Exits non-zero on any failure so the caller can react.

set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -x ".venv/bin/python3" ]; then PY=".venv/bin/python3"; else PY="python3"; fi

# Belt-and-suspenders: also export config.env into the shell env (harmless if the
# Python loader already handles it). Real env vars still win.
if [ -f config.env ]; then set -a; . ./config.env; set +a; fi

# Ensure required deps in a fresh sandbox (no-op if already present).
$PY -c "import pandas, numpy, requests" 2>/dev/null || \
  $PY -m pip install pandas numpy requests --break-system-packages -q

echo "── Step 1/2: Refresh ETF universe from EODHD (since 2005-01-01) ──"
$PY download_universe.py 2005-01-01

echo
echo "── Step 2/2: WMA5/SMA40 crossover screen + email ──"
$PY daily_crossover_signal.py --email-only

echo
echo "Done."
