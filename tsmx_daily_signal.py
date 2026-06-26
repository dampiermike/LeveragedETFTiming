"""
TSMX SMA/WMA Crossover - Daily Signal Generator

Replays the locked crossover engine on the latest TSMX data and reports tomorrow's
pending action based on today's close.

Locked strategy (from sweep_crossover.py, top result by Sharpe over 2025-02→2026-06):
  LONG  TSMX when WMA(5) >= WMA(40)
  FLAT  (cash)   when WMA(5) <  WMA(40)
Execution: signal at close of t fills at the OPEN of t+1.

Each evening after the close we know state(t) = WMA5(t) >= WMA40(t). The position
we currently hold reflects state(t-1); tomorrow at the open we move to match
state(t). So the pending action compares state(t-1) -> state(t).

Sends an email (Gmail SMTP) and an iMessage/SMS (osascript, via paired iPhone),
exactly like the Nitro daily signal. Run after the close, after download_data.py.

Run: python3 tsmx_daily_signal.py            # fetch+send
     python3 tsmx_daily_signal.py --dry-run  # print only, no email/SMS
"""

import argparse
import os
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd

from crossover_backtest import load_local, moving_average

# ── Locked crossover config ─────────────────────────────────────────────────────
FAST, FAST_TYPE = 5, "wma"
SLOW, SLOW_TYPE = 40, "wma"
TICKER = "TSMX"

# ── Email / SMS config ──────────────────────────────────────────────────────────
GMAIL_USER  = os.environ.get("GOOGLE_EMAIL", "dampiermike@gmail.com")
GMAIL_PASS  = os.environ.get("GOOGLE_APP_PASSWORD", "")
TO_EMAIL    = ["dampiermike@gmail.com"]
SMS_NUMBERS = ["+12256144680"]            # <-- change to your number(s)
# Numbers that must go via SMS (Continuity relay through paired iPhone) rather
# than iMessage — e.g. Android/Verizon recipients where iMessage bounces.
SMS_FORCE   = set()


def fmt_label(window, ma_type):
    return f"{ma_type.upper()}{window}"


def compute_signal():
    """Load TSMX, compute the two MAs, and derive current state + pending action.
    Returns a dict with everything the report/SMS need."""
    px = load_local(TICKER)
    close = px["Adj Close"]
    open_ = px["Adj Open"]

    fast_ma = moving_average(close, FAST, FAST_TYPE)
    slow_ma = moving_average(close, SLOW, SLOW_TYPE)

    df = pd.DataFrame({"close": close, "open": open_, "fast": fast_ma, "slow": slow_ma})
    df = df[df["fast"].notna() & df["slow"].notna()]
    if len(df) < 2:
        raise RuntimeError(f"not enough warmed-up bars (have {len(df)}, need >= 2) — "
                           f"TSMX history too short for WMA{SLOW}")

    state = (df["fast"] >= df["slow"]).astype(int)
    state_today = int(state.iloc[-1])      # state(t)  -> governs tomorrow's open
    state_prev = int(state.iloc[-2])       # state(t-1) -> position held into tomorrow

    # Pending action at tomorrow's open.
    if state_prev == 0 and state_today == 1:
        action = "ENTER"
    elif state_prev == 1 and state_today == 0:
        action = "EXIT"
    elif state_today == 1:
        action = "HOLD"
    else:
        action = "FLAT"

    # Current-position context: find the last 0->1 transition if we're long.
    holding = state_prev == 1
    entry_date = entry_open = days_held = unreal_pct = None
    if holding:
        s = state.to_numpy()
        entry_i = len(s) - 1               # default: held since start of window
        for i in range(len(s) - 1, 0, -1):
            if s[i] == 1 and s[i - 1] == 0:
                entry_i = i
                break
        # Position entered at the OPEN of the bar AFTER the 0->1 close signal.
        fill_i = min(entry_i + 1, len(df) - 1)
        entry_date = df.index[fill_i]
        entry_open = float(df["open"].iloc[fill_i])
        days_held = len(df) - 1 - fill_i
        unreal_pct = (float(df["close"].iloc[-1]) / entry_open - 1) * 100

    last = df.iloc[-1]
    today = df.index[-1]
    gap_pct = (last["fast"] / last["slow"] - 1) * 100
    return {
        "today": today,
        "today_str": today.strftime("%Y-%m-%d"),
        "close": float(last["close"]),
        "fast": float(last["fast"]),
        "slow": float(last["slow"]),
        "gap_pct": gap_pct,
        "state_today": state_today,
        "state_prev": state_prev,
        "holding": holding,
        "action": action,
        "entry_date": entry_date,
        "entry_open": entry_open,
        "days_held": days_held,
        "unreal_pct": unreal_pct,
        "n_bars": len(df),
    }


def build_subject(sig):
    fl, sl = fmt_label(FAST, FAST_TYPE), fmt_label(SLOW, SLOW_TYPE)
    a = sig["action"]
    head = {"ENTER": f"BUY {TICKER} at open",
            "EXIT": f"SELL {TICKER} at open",
            "HOLD": f"HOLD {TICKER}",
            "FLAT": "FLAT (cash)"}[a]
    return f"TSMX {fl}/{sl} Signal {sig['today_str']}: {head}"


def build_sms(sig):
    short = sig["today_str"][2:]   # YY-MM-DD
    a = sig["action"]
    if a == "ENTER":
        msg = f"TSMX {short}: BUY {TICKER} at open"
    elif a == "EXIT":
        ph = f"  {sig['unreal_pct']:+.1f}%" if sig["unreal_pct"] is not None else ""
        msg = f"TSMX {short}: SELL {TICKER} at open{ph}"
    elif a == "HOLD":
        ph = f"  {sig['unreal_pct']:+.1f}%" if sig["unreal_pct"] is not None else ""
        msg = f"TSMX {short}: HOLD {TICKER}{ph}"
    else:
        msg = f"TSMX {short}: FLAT (cash)"
    return msg[:160]


def build_body(sig):
    fl, sl = fmt_label(FAST, FAST_TYPE), fmt_label(SLOW, SLOW_TYPE)
    relation = ">=" if sig["state_today"] == 1 else "<"
    lines = [
        "TSMX SMA/WMA Crossover — Daily Signal",
        f"As of close {sig['today_str']}",
        "=" * 48,
        "",
        f"Strategy : LONG {TICKER} while {fl} >= {sl}, else FLAT (cash)",
        f"Execution: at TOMORROW's OPEN",
        "",
        "── Today's close ──",
        f"  {TICKER} close : {sig['close']:.2f}",
        f"  {fl:<7}     : {sig['fast']:.2f}",
        f"  {sl:<7}     : {sig['slow']:.2f}",
        f"  {fl} {relation} {sl}   (gap {sig['gap_pct']:+.2f}%)",
        f"  Signal state : {'LONG' if sig['state_today'] else 'FLAT'}",
        "",
    ]
    if sig["holding"]:
        lines += [
            "── Current position ──",
            f"  LONG {TICKER} since {sig['entry_date'].strftime('%Y-%m-%d')} "
            f"@ ~{sig['entry_open']:.2f} open",
            f"  Days held    : {sig['days_held']}",
            f"  Unrealized   : {sig['unreal_pct']:+.2f}%",
            "",
        ]
    else:
        lines += ["── Current position ──", "  FLAT (in cash)", ""]

    action_text = {
        "ENTER": f"  >>> BUY {TICKER} at tomorrow's open  ({fl} crossed above {sl})",
        "EXIT":  f"  >>> SELL {TICKER} at tomorrow's open  ({fl} crossed below {sl})",
        "HOLD":  f"  >>> HOLD {TICKER} (no change)",
        "FLAT":  f"  >>> STAY FLAT in cash (no change)",
    }[sig["action"]]
    lines += ["── Pending action (tomorrow's open) ──", action_text, "",
              "=" * 48,
              f"(replayed {sig['n_bars']} warmed-up bars; signal at close fills next open)"]
    return "\n".join(lines)


def send_email(subject, body_text):
    if not GMAIL_PASS:
        print("send_email: GOOGLE_APP_PASSWORD not set — skipping")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = ", ".join(TO_EMAIL)
    msg.attach(MIMEText(body_text, "plain"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())
    print(f"Email sent to {TO_EMAIL}")


def send_imessage(numbers, body):
    for num in numbers:
        service_type = "SMS" if num in SMS_FORCE else "iMessage"
        # Body passed as argv (item 1) to avoid AppleScript string-literal breakage
        # on quotes/backslashes/newlines. The Messages terms (service/participant/
        # send) require the tell-application block to compile.
        script = (
            "on run argv\n"
            '  tell application "Messages"\n'
            f"    set svc to first service whose service type = {service_type}\n"
            f'    send (item 1 of argv) to participant "{num}" of svc\n'
            "  end tell\n"
            "end run"
        )
        try:
            subprocess.run(["osascript", "-e", script, body],
                           check=True, capture_output=True, timeout=30)
            print(f"  iMessage sent to {num} ({service_type})")
        except subprocess.TimeoutExpired:
            print(f"  WARNING: iMessage to {num} timed out after 30s — not delivered")
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors="replace").strip() if e.stderr else "(no detail)"
            print(f"  WARNING: iMessage to {num} FAILED — not delivered: {err}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the report; do not send email/SMS")
    args = ap.parse_args()

    sig = compute_signal()
    subject = build_subject(sig)
    body = build_body(sig)
    sms = build_sms(sig)

    print(body)
    print()
    print(f"SUBJECT: {subject}")
    print(f"SMS    : {sms}")

    if args.dry_run:
        print("\n[dry-run] no email/SMS sent.")
        return

    print(f"\nSending email to {TO_EMAIL} ...")
    send_email(subject, body)
    print(f"Sending iMessage/SMS to {SMS_NUMBERS}: {sms}")
    send_imessage(SMS_NUMBERS, sms)


if __name__ == "__main__":
    main()
