"""Load secrets from config.env into os.environ.

Import this FIRST (before any module that reads os.environ at import time, e.g.
tsmx_daily_signal). Real environment variables always take precedence, so local
runs that already have ~/.bash_profile exported are unaffected; the scheduled
sandbox — which loads no bash_profile — gets its secrets from config.env instead.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG = os.path.join(_HERE, "config.env")


def load(path=_CFG):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:   # don't override real env
                os.environ[key] = val


load()
