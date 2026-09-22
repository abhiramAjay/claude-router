"""Tests for the zero-dependency client. No laya, no network, no API key."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from claude_router import RouterClient, Route


@pytest.fixture
def stub_server():
    """A stand-in sidecar, so the client is tested without loading a 421M model."""
    payload = {"tier": "sonnet", "model": "claude-sonnet-5", "effort": "high",
               "difficulty": 1.8, "domain": "code", "reason": "stub", "escalated": False,
               "signals": {}}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()


def test_routes_through_sidecar(stub_server):
    d = RouterClient(url=stub_server).route("add pagination to this endpoint")
    assert d.model == "claude-sonnet-5"
    assert d.tier == "sonnet"


def test_falls_back_when_sidecar_is_down():
    d = RouterClient(url="http://127.0.0.1:9", timeout=0.3).route("anything")
    assert d.model == "claude-opus-5"
    assert "unavailable" in d.reason


def test_raise_mode_surfaces_the_failure():
    c = RouterClient(url="http://127.0.0.1:9", timeout=0.3, on_error="raise")
    with pytest.raises(OSError):
        c.route("anything")


def test_rejects_bad_on_error_value():
    with pytest.raises(ValueError):
        RouterClient(on_error="explode")


@pytest.mark.parametrize("tier,model,wants_effort", [
    ("haiku", "claude-haiku-4-5", False),      # Haiku 4.5 rejects effort + adaptive thinking
    ("sonnet", "claude-sonnet-5", True),
    ("opus", "claude-opus-5", True),
])
def test_messages_kwargs_are_correct_per_model(tier, model, wants_effort):
    kw = Route(tier=tier, model=model, effort="high", difficulty=1.0,
               domain="code", reason="").messages_kwargs()
    assert kw["model"] == model
    assert ("output_config" in kw) is wants_effort
    assert ("thinking" in kw) is wants_effort
    assert "budget_tokens" not in json.dumps(kw)   # 400 on every model we route to


def test_route_survives_a_serialisation_round_trip():
    a = Route(tier="opus", model="claude-opus-5", effort="xhigh", difficulty=2.3,
              domain="code", reason="hard", escalated=True)
    assert Route.from_dict(a.to_dict()) == a
