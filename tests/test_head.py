"""Tests for the trained-head inference path. No laya, no model file needed."""
import os

import pytest

from claude_router.head import DEFAULT_PATH, HeadNotTrained, HeadRouter, available


class FakeClf:
    classes_ = ["lookup", "edit", "design"]

    def predict(self, X):
        return ["lookup", "design"][: len(X)]

    def predict_proba(self, X):
        return [[0.7, 0.2, 0.1]] * len(X)


def head(tmp_path):
    r = HeadRouter(path=str(tmp_path / "none.joblib"))
    r._bundle = {"clf": FakeClf(), "classes": FakeClf.classes_,
                 "kind_tier": {"lookup": "haiku", "edit": "sonnet", "design": "opus"}}
    r._embed = lambda prompts: [[0.0]] * len(prompts)
    return r


def test_missing_head_says_what_to_run(tmp_path):
    r = HeadRouter(path=str(tmp_path / "absent.joblib"))
    with pytest.raises(HeadNotTrained) as e:
        r.tier("anything")
    assert "train_head.py" in str(e.value)


def test_available_reflects_the_file(tmp_path):
    p = tmp_path / "h.joblib"
    assert not available(str(p))
    p.write_bytes(b"x")
    assert available(str(p))


def test_tier_maps_kinds_to_tiers(tmp_path):
    assert head(tmp_path).tier("find the config") == "haiku"


def test_tiers_is_batched(tmp_path):
    # One encoder pass for many prompts: the eval loop embeds 40 at a time.
    assert head(tmp_path).tiers(["a", "b"]) == ["haiku", "opus"]


def test_confidence_is_keyed_by_tier(tmp_path):
    c = head(tmp_path).confidence("anything")
    assert set(c) == {"haiku", "sonnet", "opus"}
    assert abs(sum(c.values()) - 1.0) < 1e-6


def test_importing_the_module_does_not_load_torch(tmp_path):
    # Lazy loading is the point: projects import claude_router without paying
    # for the 590 MB of torch that only the sidecar host has.
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, claude_router.head; print('torch' in sys.modules, 'laya' in sys.modules)"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.stdout.strip() == "False False", out.stdout + out.stderr
