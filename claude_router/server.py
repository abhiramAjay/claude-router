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

from .router import ClaudeRouter

_router = ClaudeRouter()
_ready = threading.Event()
_stats = {"routes": 0, "by_tier": {}}


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
            return self._send(200, {"ok": True, "loaded": _ready.is_set(), **_stats})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/route":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            prompt = json.loads(self.rfile.read(n))["prompt"]
        except (ValueError, KeyError, TypeError) as e:
            return self._send(400, {"error": "bad request: %s" % e})
        try:
            d = _router.route(prompt)
        except Exception as e:                      # never take the daemon down on one bad prompt
            return self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})
        _stats["routes"] += 1
        _stats["by_tier"][d.tier] = _stats["by_tier"].get(d.tier, 0) + 1
        self._send(200, d.to_dict())

    def log_message(self, fmt, *a):                 # quiet by default; --verbose restores it
        if getattr(self.server, "verbose", False):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % a))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="laya-routerd")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8129)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    print("loading laya...", file=sys.stderr, flush=True)
    _router.agent                                   # pay the ~21 s load before we accept traffic
    _ready.set()
    print("ready in %.1fs -- listening on http://%s:%d" % (time.time() - t0, args.host, args.port),
          file=sys.stderr, flush=True)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.verbose = args.verbose
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)
        srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
