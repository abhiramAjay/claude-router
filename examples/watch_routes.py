"""Follow the routing hook's decision log, one readable line per delegated task.

    python3 examples/watch_routes.py            # new decisions only
    python3 examples/watch_routes.py --all      # replay the log first

Standard library only. Reads CLAUDE_ROUTER_HOOK_LOG, like the hook itself.
"""
import argparse
import json
import os
import sys
import time

LOG = os.environ.get("CLAUDE_ROUTER_HOOK_LOG",
                     os.path.expanduser("~/Library/Logs/laya-cc-hook.jsonl"))

COLOR = {"haiku": "\033[32m", "sonnet": "\033[36m", "opus": "\033[35m"}
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def fmt(rec: dict, width: int) -> str:
    model = rec.get("chosen_model") or "default"
    when = time.strftime("%H:%M:%S", time.localtime(rec.get("ts", time.time())))
    desc = (rec.get("description") or "task")[:width]
    extra = ""
    if rec.get("context_chars"):
        extra = "  +%.1fk shared context" % (rec["context_chars"] / 1000)
    return "%s%s%s  %s%-6s%s  %-*s  %s%s%s%s" % (
        DIM, when, RESET, BOLD + COLOR.get(model, ""), model, RESET,
        width, desc, DIM, rec.get("reason", ""), extra, RESET)


def follow(path: str, replay: bool):
    while not os.path.exists(path):
        time.sleep(0.5)
    with open(path) as f:
        if not replay:
            f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", help="replay existing decisions first")
    ap.add_argument("--width", type=int, default=36, help="description column width")
    args = ap.parse_args()
    print("%swatching %s%s" % (DIM, LOG.replace(os.path.expanduser("~"), "~", 1), RESET), flush=True)
    try:
        for rec in follow(LOG, args.all):
            print(fmt(rec, args.width), flush=True)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
