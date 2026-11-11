# workshop\Day6\trace_viewer.py — Day 6 Session 2, Block C (observability)
#
# Reads the JSON-lines agent.log that main.py writes, filters it, prints a
# compact trace table + a 3-number summary. This is the "simple trace viewer"
# from the curriculum — no dashboard, no extra install, zero LLM quota.
#
# Usage (from inside workshop\Day6, service running in another terminal):
#   python trace_viewer.py                 # last 20 events
#   python trace_viewer.py --session alice # one chat session only
#   python trace_viewer.py --errors        # only 4xx/5xx requests
#   python trace_viewer.py -n 50           # widen the window
#
# Industry pointer (no lab time): LangSmith / OpenTelemetry give the same
# data as distributed traces across services — the FIELDS are what matter:
# ts, session, route/status, latency, budget, http_status.

import json
import sys
import argparse
from pathlib import Path

LOG_PATH = Path(__file__).with_name("agent.log")   # same folder as main.py


def main() -> None:
    ap = argparse.ArgumentParser(description="View agent.log traces")
    ap.add_argument("--session", default=None, help="Filter by session_id")
    ap.add_argument("--errors", action="store_true",
                    help="Only requests that failed (http_status >= 400)")
    ap.add_argument("-n", type=int, default=20, help="Show the last N events")
    args = ap.parse_args()

    if not LOG_PATH.exists():
        sys.exit(f"No log yet at {LOG_PATH.name} — start main.py and send "
                 "one /ask first.")

    # Parse defensively: one corrupt line must not kill the viewer.
    records = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if args.session and rec.get("session_id") != args.session:
            continue
        if args.errors and int(rec.get("http_status", 200)) < 400:
            continue
        records.append(rec)

    # Show the TAIL: the last N events after filtering (newest are appended).
    shown = records[-args.n:]
    if not shown:
        sys.exit("No events matched that filter.")

    # Fixed-width table — enough structure to skim, no dependency needed.
    print(f"{'ts':<21} {'event':<10} {'session':<10} "
          f"{'route/status':<28} {'ms':>7} {'http':>5}")
    print("-" * 86)
    for r in shown:
        ts = str(r.get("ts", "?"))[:19]                  # drop sub-seconds
        event = str(r.get("event", "?"))[:10]
        session = str(r.get("session_id", "-"))[:10]
        # 200s show the route+agent status; failures show why they failed.
        detail = (f"{r.get('route', '-')}/{r.get('status', '-')}"
                  if r.get("http_status", 200) < 400
                  else str(r.get("error", "failed")))
        latency = r.get("latency_ms", "-")
        print(f"{ts:<21} {event:<10} {session:<10} "
              f"{detail:<28} {str(latency):>7} {str(r.get('http_status', '-')):>5}")

    # The 3-number summary an engineer actually looks at first.
    asks = [r for r in records if r.get("event") == "ask"]
    ok = [r for r in asks if int(r.get("http_status", 200)) < 400]
    failed = [r for r in asks if int(r.get("http_status", 200)) >= 400]
    lat = [r["latency_ms"] for r in ok if isinstance(r.get("latency_ms"), int)]
    print("-" * 86)
    print(f"asks={len(asks)}  ok={len(ok)}  failed={len(failed)}  "
          f"avg_latency_ms={sum(lat) // len(lat) if lat else '-'}")


if __name__ == "__main__":
    main()
