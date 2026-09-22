"""Route each prompt to a Claude model by complexity, using laya as a local classifier.

Two ways in:

    from claude_router import RouterClient      # thin, zero-dependency, talks to the sidecar
    from claude_router import ClaudeRouter      # in-process, needs the [server] extra (torch)

`RouterClient` is what projects should use. It has no dependencies beyond the
standard library and falls back to a fixed model if the sidecar is unreachable,
so a dead daemon degrades cost, never correctness.
"""
from .router import ClaudeRouter, Route, TIERS, DIFFICULTY_CUTS, CHEAP_DOMAINS, HIGH_STAKES, ROUTER_QUESTIONS
from .client import RouterClient, DEFAULT_URL

__all__ = ["RouterClient", "ClaudeRouter", "Route", "DEFAULT_URL", "TIERS",
           "DIFFICULTY_CUTS", "CHEAP_DOMAINS", "HIGH_STAKES", "ROUTER_QUESTIONS"]
__version__ = "0.1.0"
