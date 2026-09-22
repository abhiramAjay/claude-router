"""Inference for the trained routing head.

The head is a small classifier over frozen laya embeddings (see
`training/train_head.py`). laya ships no `fit`, so the checkpoint itself is never
touched -- it is used as a ~38 ms feature extractor and the tier boundary is
learned on top.

    from claude_router.head import HeadRouter

    router = HeadRouter()           # raises if no head has been trained yet
    router.tier("Repo: foo. Find where parse_config is defined.")   # 'haiku'

Sits behind the same `tier`-shaped interface as the rest of the package, so it
can be dropped into `claude_router/agents.py` in place of the difficulty score
without touching the delegation hook.

This module deliberately does NOT strip the prompt before embedding. Stripping
was a patch for the difficulty score's length bias, and it breaks on exactly the
payloads that matter -- a stack trace or a log dump has no sentence punctuation,
so nothing is removed. The head is instead trained on data where payload size is
sampled independently of the label, which removes the shortcut at the source.
"""
import os
from typing import Any, Dict, List, Optional

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "routing_head.joblib")


class HeadNotTrained(RuntimeError):
    pass


class HeadRouter:
    """Classify a delegated brief into a tier with a trained head.

    Loading costs the laya checkpoint (~20 s) plus a negligible classifier, so
    hold one instance. In the sidecar that happens once at startup, same as the
    existing router.
    """

    def __init__(self, path: str = DEFAULT_PATH, agent: Any = None):
        self.path = path
        self._agent = agent
        self._bundle = None
        self._embed = None

    # -- lazy, so importing this module never pays for torch -----------------
    @property
    def bundle(self) -> Dict[str, Any]:
        if self._bundle is None:
            if not os.path.exists(self.path):
                raise HeadNotTrained(
                    "no trained head at %s -- run training/generate_data.py then "
                    "training/train_head.py" % self.path)
            import joblib
            self._bundle = joblib.load(self.path)
        return self._bundle

    @property
    def embed(self):
        if self._embed is None:
            from laya import embed_fn_from_agent, load
            self._embed = embed_fn_from_agent(self._agent or load())
        return self._embed

    def tiers(self, prompts: List[str]) -> List[str]:
        """Batched -- one encoder pass for many prompts, which is how the
        eval loop should call it."""
        kind_tier = self.bundle["kind_tier"]
        kinds = self.bundle["clf"].predict(self.embed(list(prompts)))
        return [kind_tier[str(k)] for k in kinds]

    def tier(self, prompt: str) -> str:
        return self.tiers([prompt])[0]

    def confidence(self, prompt: str) -> Dict[str, float]:
        """Per-tier probability. Useful for a "when unsure, spend more" policy:
        a low-margin prediction is a reasonable trigger to escalate a tier
        rather than guess, since underspending costs a retry and overspending
        costs cents."""
        clf = self.bundle["clf"]
        probs = clf.predict_proba(self.embed([prompt]))[0]
        kind_tier = self.bundle["kind_tier"]
        return {kind_tier[str(c)]: float(p) for c, p in zip(clf.classes_, probs)}


def available(path: str = DEFAULT_PATH) -> bool:
    """Whether a head has been trained, so callers can fall back quietly."""
    return os.path.exists(path)
