#!/bin/bash
# TSMX SMA/WMA Crossover — Daily runner
# Updates the TSMX json, checks freshness, then generates and sends the daily
# signal (email + iMessage/SMS). Logs to logs/daily-*.log and sends a heartbeat
# on success / failure email on any error, exactly like the Nitro runner.

set -eo pipefail

# Load credentials (GOOGLE_EMAIL/GOOGLE_APP_PASSWORD, EODHD_API_TOKEN, etc.)
source "$HOME/.bash_profile"

# Self-locate: this script lives in the project root.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PROJECT_NAME="TSMX"

# Prefer a project venv if present, else system python3.
if [ -x ".venv/bin/python3" ]; then PY=".venv/bin/python3"; else PY="python3"; fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TS="$(date '+%Y-%m-%d_%H%M%S')"
LOG_FILE="$LOG_DIR/daily-${TS}.log"

# ── Email alert helper ────────────────────────────────────────────────────────
# Usage: send_alert "subject" "/path/to/body_file"
send_alert() {
    local subject="$1"
    local body_file="$2"
    "$PY" -c '
import os, sys, smtplib
from email.mime.text import MIMEText
user = os.environ.get("GOOGLE_EMAIL", "")
pw   = os.environ.get("GOOGLE_APP_PASSWORD", "")
if not (user and pw):
    print("send_alert: GOOGLE_EMAIL/GOOGLE_APP_PASSWORD not set — skipping", file=sys.stderr)
    sys.exit(0)
subject = sys.argv[1]
with open(sys.argv[2], "r") as f:
    body = f.read()
msg = MIMEText(body)
msg["Subject"] = subject
msg["From"]    = user
msg["To"]      = user
try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(user, pw)
        s.sendmail(user, [user], msg.as_string())
    print(f"alert sent: {subject}")
except Exception as e:
    print(f"alert email failed: {e}", file=sys.stderr)
' "$subject" "$body_file"
}

# ── Failure trap ──────────────────────────────────────────────────────────────
on_failure() {
    local rc=$?
    local alert_body="/tmp/${PROJECT_NAME}_fail_$$.txt"
    {
        echo "$PROJECT_NAME daily run FAILED"
        echo "Exit code: $rc"
        echo "Timestamp: $(date)"
        echo "Log file:  $ROOT_DIR/$LOG_FILE"
        echo ""
        echo "─── Last 50 log lines ───"
        tail -n 50 "$LOG_FILE" 2>/dev/null || echo "(no log available)"
    } > "$alert_body"
    send_alert "❌ $PROJECT_NAME daily FAILED (rc=$rc)" "$alert_body"
    rm -f "$alert_body"
    exit $rc
}
trap on_failure ERR

# ── Pipeline (all output tee'd to log file) ───────────────────────────────────
{
    echo "=========================================="
    echo "  $PROJECT_NAME Daily Run — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    echo
    echo "── Step 1/3: Update TSMX json from EODHD ──"
    $PY download_data.py

    echo
    echo "── Step 2/3: Freshness check (aborts run if stale) ──"
    $PY validate_freshness.py

    echo
    echo "── Step 3/3: Daily signal (email + iMessage/SMS) ──"
    $PY tsmx_daily_signal.py

    echo
    echo "=========================================="
    echo "  Done — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
} 2>&1 | tee "$LOG_FILE"

# ── Heartbeat on success ──────────────────────────────────────────────────────
alert_body="/tmp/${PROJECT_NAME}_ok_$$.txt"
{
    echo "$PROJECT_NAME daily run completed successfully."
    echo "Timestamp: $(date)"
    echo "Log file:  $ROOT_DIR/$LOG_FILE"
    echo ""
    echo "─── Last 30 log lines ───"
    tail -n 30 "$LOG_FILE"
} > "$alert_body"
send_alert "✅ $PROJECT_NAME daily OK" "$alert_body"
rm -f "$alert_body"
