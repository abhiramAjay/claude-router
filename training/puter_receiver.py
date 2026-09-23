"""Local sink for briefs generated in the browser by Puter.js.

    python training/puter_receiver.py          # listens on 127.0.0.1:8130

Puter.js is a browser SDK -- there is no API key and no server-side call, so the
generation loop has to run in a page. This is the other half: the page POSTs
each batch here and it lands in the same JSONL corpus every other provider
writes to, so nothing downstream needs to know where a brief came from.

Loopback only. CORS is wide open because the page is served from a file:// or
js.puter.com origin and this holds nothing worth protecting -- but that is also
why it must not be bound to a routable address.
"""
import argparse
import hashlib
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "seeded", "briefs-puter.jsonl")
TIERS = ("lookup", "edit", "design")

_seen = set()
_stats = {"written": 0, "duplicate": 0, "rejected": 0}


def load_seen() -> None:
    if not os.path.exists(OUT):
        return
    with open(OUT) as f:
        for line in f:
            try:
                _seen.add(hashlib.sha1(json.loads(line)["brief"].encode()).hexdigest())
            except Exception:
                pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        self._send(200, {"ok": True, **_stats, "total": len(_seen)})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            rows = json.loads(self.rfile.read(n))
        except (ValueError, TypeError) as e:
            return self._send(400, {"error": "bad json: %s" % e})
        if isinstance(rows, dict):
            rows = [rows]

        written = dup = bad = 0
        with open(OUT, "a") as f:
            for r in rows if isinstance(rows, list) else []:
                if not (isinstance(r, dict) and r.get("brief") and r.get("tier") in TIERS):
                    bad += 1
                    continue
                h = hashlib.sha1(str(r["brief"]).encode()).hexdigest()
                if h in _seen:
                    dup += 1
                    continue
                _seen.add(h)
                f.write(json.dumps({"brief": str(r["brief"]), "tier": r["tier"],
                                    "payload": r.get("payload", "?"),
                                    "source": r.get("source", "puter")}) + "\n")
                written += 1
        _stats["written"] += written
        _stats["duplicate"] += dup
        _stats["rejected"] += bad
        if written:
            print("+%d (total %d)" % (written, len(_seen)), file=sys.stderr, flush=True)
        self._send(200, {"written": written, "duplicate": dup, "rejected": bad,
                         "total": len(_seen)})

    def log_message(self, *a):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(prog="puter_receiver")
    ap.add_argument("--port", type=int, default=8130)
    args = ap.parse_args(argv)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    load_seen()
    print("%d brief(s) already on disk; listening on http://127.0.0.1:%d -> %s"
          % (len(_seen), args.port, OUT), file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
