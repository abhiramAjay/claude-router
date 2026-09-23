"""Routing sidecar: loads laya once, serves decisions over localhost HTTP.

    laya-routerd                      # or: python -m claude_router.server
    laya-routerd --port 8129 --host 127.0.0.1

Binds to loopback only by default -- this holds no secrets, but it also has no
auth, so do not expose it. The ~21 s model load happens once at startup; each
route is then ~75 ms. Threaded, and laya's own inference is not lock-guarded,
so concurrent requests share the checkpoint instead of serialising.
"""
import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .router import TIERS, ClaudeRouter, Route
from . import head as head_mod

_router = ClaudeRouter()
# The trained head lives here rather than in the client because it needs laya
# and sklearn. Keeping it server-side is what lets projects stay on the
# stdlib-only client -- the whole reason the sidecar exists.
_head = head_mod.HeadRouter()
_ready = threading.Event()
_stats = {"routes": 0, "by_tier": {}, "by_mode": {}}


def _route(prompt: str, mode: str) -> Route:
    """Route by the trained head, or by the difficulty score.

    Measured on 40 real delegated briefs: head 38/40, score 20/40. The score
    path stays because it needs no trained artefact -- a checkout with no
    routing_head.joblib still routes, just worse.
    """
    if mode == "head":
        d = _head.route(prompt, escalate=_router.escalate)
        return Route(tier=d["tier"], model=TIERS[d["tier"]][0], effort=d["effort"],
                     difficulty=float("nan"),      # the head predicts a class, not a scalar
                     domain="", reason=d["reason"], escalated=d["escalated"],
                     signals={"head": d["confidence"]})
    return _router.route(prompt)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            return self._send(200, {"ok": True, "loaded": _ready.is_set(),
                                    "head": head_mod.available(), **_stats})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/route":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n))
            prompt = body["prompt"]
            mode = body.get("mode") or self.server.default_mode
        except (ValueError, KeyError, TypeError) as e:
            return self._send(400, {"error": "bad request: %s" % e})
        if mode == "head" and not head_mod.available():
            mode = "score"                           # no trained head: degrade, do not fail
        try:
            d = _route(prompt, mode)
        except Exception as e:                      # never take the daemon down on one bad prompt
            return self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})
        _stats["routes"] += 1
        _stats["by_tier"][d.tier] = _stats["by_tier"].get(d.tier, 0) + 1
        _stats["by_mode"][mode] = _stats["by_mode"].get(mode, 0) + 1
        self._send(200, d.to_dict())

    def log_message(self, fmt, *a):                 # quiet by default; --verbose restores it
        if getattr(self.server, "verbose", False):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % a))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="laya-routerd")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8129)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--mode", choices=("head", "score"), default="head",
                    help="default routing mode when a request does not name one; "
                         "falls back to score if no head has been trained")
    args = ap.parse_args(argv)

    t0 = time.time()
    print("loading laya...", file=sys.stderr, flush=True)
    _router.agent                                   # pay the ~21 s load before we accept traffic
    if args.mode == "head" and head_mod.available():
        _head._agent = _router.agent                # one checkpoint, shared, not loaded twice
        _head.bundle                                # surface a broken artefact now, not mid-request
        print("routing head loaded", file=sys.stderr, flush=True)
    elif args.mode == "head":
        print("no trained head; falling back to the difficulty score", file=sys.stderr, flush=True)
    _ready.set()
    print("ready in %.1fs -- listening on http://%s:%d" % (time.time() - t0, args.host, args.port),
          file=sys.stderr, flush=True)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.verbose = args.verbose
    srv.default_mode = args.mode
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)
        srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
