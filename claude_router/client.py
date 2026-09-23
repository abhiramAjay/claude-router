"""Zero-dependency client for the laya routing sidecar.

Standard library only -- no torch, no httpx. This is what your projects import.

    from claude_router import RouterClient

    router = RouterClient()                      # localhost:8129
    d = router.route("prove that every finite integral domain is a field")
    client.messages.create(**d.messages_kwargs(), max_tokens=16000,
                           messages=[{"role": "user", "content": prompt}])

`mode` picks the classifier: "head" for the trained one, "score" for the
difficulty cuts, or None to take whatever the sidecar defaults to -- which means
switching every project over is a daemon restart, not a release of each client.

The default timeout is 5 s rather than 2 s: delegated briefs run to thousands of
characters, and a route that breaches the timeout is scored as a fallback to the
expensive model, which is a silent and expensive way to lose.

If the sidecar is down, slow, or erroring, `route()` returns `fallback_model`
with a reason saying so. Routing degrades to "always use the good model" --
you stop saving money, nothing breaks. Set `on_error="raise"` to surface it
loudly instead.
"""
import json
import os
import urllib.error
import urllib.request
from typing import Optional

from .router import TIERS, Route

DEFAULT_URL = os.environ.get("CLAUDE_ROUTER_URL", "http://127.0.0.1:8129")
FALLBACK_MODEL = os.environ.get("CLAUDE_ROUTER_FALLBACK", "claude-opus-5")


class RouterClient:
    def __init__(self, url: str = DEFAULT_URL, timeout: float = 5.0,
                 fallback_model: str = FALLBACK_MODEL, fallback_effort: Optional[str] = "high",
                 on_error: str = "fallback", mode: Optional[str] = None):
        if on_error not in ("fallback", "raise"):
            raise ValueError("on_error must be 'fallback' or 'raise'")
        if mode not in (None, "head", "score"):
            raise ValueError("mode must be None, 'head' or 'score'")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.fallback_model = fallback_model
        self.fallback_effort = fallback_effort
        self.on_error = on_error
        # None lets the sidecar decide, so switching every project to the trained
        # head is a daemon restart rather than a release of each client.
        self.mode = mode

    def _fallback(self, why: str) -> Route:
        tier = next((t for t, (m, _, _) in TIERS.items() if m == self.fallback_model), "opus")
        return Route(tier=tier, model=self.fallback_model, effort=self.fallback_effort,
                     difficulty=float("nan"), domain="", reason="router unavailable (%s); using fallback" % why,
                     escalated=False, signals={})

    def route(self, prompt: str) -> Route:
        payload = {"prompt": prompt}
        if self.mode:
            payload["mode"] = self.mode
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.url + "/route", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return Route.from_dict(json.loads(r.read()))
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            if self.on_error == "raise":
                raise
            return self._fallback(type(e).__name__)

    def health(self) -> dict:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=self.timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
