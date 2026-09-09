"""1-minute account/node health probe: prints env facts, uploads nothing."""
import datetime
import platform
import sys

print("[PROBE] alive", datetime.datetime.now().isoformat(), flush=True)
print("[PROBE] python", sys.version.split()[0], "on", platform.platform(), flush=True)
print("[PROBE] OK: account can run tasks", flush=True)
