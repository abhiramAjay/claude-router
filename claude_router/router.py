"""Route each prompt to a Claude model by complexity, using laya as the classifier.

laya runs a 421M ModernBERT locally (~140ms on CPU, no API call, no tokens) and
answers typed questions about the prompt in one forward pass. We use two of its
signals -- `difficulty` and `domain` -- and a deterministic guard for high-stakes
requests, because laya's score head does not discriminate stakes (measured: it
returns ~1.0 for everything from "hi there!" to "drop the users table in prod").

    from claude_router import ClaudeRouter

    router = ClaudeRouter()
    d = router.route("prove that every finite integral domain is a field")
    d.model    # 'claude-opus-5'
    d.effort   # 'high'
    d.reason   # 'difficulty 2.09 >= 1.95'

    # Splat straight into the SDK -- per-model params are already correct.
    client.messages.create(**d.messages_kwargs(), max_tokens=16000,
                           messages=[{"role": "user", "content": prompt}])
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import re

# Anthropic first-party pricing, $ per 1M tokens (input, output).
TIERS = {
    "haiku":  ("claude-haiku-4-5", 1.00,  5.00),
    "sonnet": ("claude-sonnet-5",  2.00, 10.00),
    "opus":   ("claude-opus-5",    5.00, 25.00),
}

# laya `difficulty` is the expected value over four ordered levels (0..3), so it
# is continuous, not a class. These cuts come from the labeled cases in
# tests/test_routing.py -- re-run that after changing them.
DIFFICULTY_CUTS = {"sonnet": 1.20, "opus": 1.95}

# Domains whose cheap-tier cut is raised: laya scores trivia and arithmetic
# around 1.35, well above the global Haiku cut, because the *question* reads as
# substantive even when the *answer* is a lookup.
CHEAP_DOMAINS = {"factual_lookup": 1.50, "chitchat": 1.50, "math_or_logic": 1.45}

# Deterministic escalation. laya cannot see stakes, so we do not ask it to.
# A match floors the decision at Opus regardless of difficulty.
HIGH_STAKES = re.compile(
    r"(?:"
    # destructive operations
    r"\bdrop\s+(?:the\s+)?(?:\w+\s+)?(?:table|database|schema)\b|\btruncate\s+table\b|"
    r"\brm\s+-rf\b|\bforce[- ]push\b|\bprod(?:uction)?\b[^.]{0,40}\b(?:delet|drop|wipe|purge|migrat|deploy)\w*|"
    # health and safety
    r"\b(?:swallow|overdos|poison|bleed|suicid|allerg|dosag|symptom)\w*|\bchest\s+pain\b|\bmg\s+of\b|"
    # legal
    r"\b(?:contract|lawsuit|liabilit|indemnif|litigat)\w*|\bnda\b|\bterms\s+of\s+service\b|"
    # money and secrets
    r"\bwire\s+transfer\b|\b(?:invoice|payroll|refund|credential)\w*|\btax(?:es)?\b|"
    r"\b(?:secret|private|api)\s+key\b"
    r")", re.I)

ROUTER_QUESTIONS: Dict[str, Any] = {
    "difficulty": {
        "type": "score",
        "instructions": "How much reasoning does `request` require from an AI assistant to answer well?",
        "criteria": [
            "trivial: a greeting, a single fact, or a one-line edit",
            "easy: a short self-contained answer with no real reasoning",
            "moderate: several dependent steps, or code spanning a few files",
            "hard: long multi-step reasoning, novel design, proofs, or deep specialist expertise",
        ],
    },
    "domain": {
        "type": "choice",
        "instructions": "What domain does `request` belong to?",
        "criteria": {
            "code": "software engineering, programming, refactoring, architecture, debugging",
            "math_or_logic": "mathematics, logic puzzles, proofs, complex calculation",
            "writing": "creative writing, essays, emails, blog posts, copywriting",
            "factual_lookup": "facts, definitions, trivia, history",
            "data_analysis": "statistics, SQL, data manipulation, metrics",
            "chitchat": "casual conversation, greetings, small talk",
        },
    },
}


@dataclass
class Route:
    tier: str
    model: str
    effort: Optional[str]          # None on Haiku 4.5 -- it rejects output_config.effort
    difficulty: float
    domain: str
    reason: str
    escalated: bool = False
    signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"tier": self.tier, "model": self.model, "effort": self.effort,
                "difficulty": self.difficulty, "domain": self.domain, "reason": self.reason,
                "escalated": self.escalated, "signals": self.signals}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Route":
        return cls(tier=d["tier"], model=d["model"], effort=d.get("effort"),
                   difficulty=d.get("difficulty", 0.0), domain=d.get("domain", ""),
                   reason=d.get("reason", ""), escalated=d.get("escalated", False),
                   signals=d.get("signals") or {})

    def messages_kwargs(self) -> Dict[str, Any]:
        """The model/thinking/effort params for this route, ready to splat.

        Haiku 4.5 predates adaptive thinking and errors on `output_config.effort`,
        so the cheap tier ships neither -- it is only ever chosen for work that
        does not need to think. Sonnet 5 and Opus 5 both take adaptive thinking;
        `budget_tokens` is a 400 on both.
        """
        kw: Dict[str, Any] = {"model": self.model}
        if self.tier != "haiku":
            kw["thinking"] = {"type": "adaptive"}
            kw["output_config"] = {"effort": self.effort}
        return kw

    def cost_per_mtok(self):
        _, i, o = TIERS[self.tier]
        return {"input": i, "output": o}


class ClaudeRouter:
    """Classify a prompt locally with laya, then pick a Claude model and effort."""

    def __init__(self, device: Optional[str] = None, cuts: Optional[Dict[str, float]] = None,
                 escalate_pattern: Optional[re.Pattern] = None, agent: Any = None):
        self.cuts = dict(DIFFICULTY_CUTS, **(cuts or {}))
        self.escalate = escalate_pattern if escalate_pattern is not None else HIGH_STAKES
        self._agent = agent
        self._device = device

    @property
    def agent(self):
        if self._agent is None:                 # ~90s on first ever call (model download), then ~3s
            from laya import load
            self._agent = load(device=self._device)
        return self._agent

    def route(self, prompt: str) -> Route:
        answers = self.agent.system_one({"request": prompt}, ROUTER_QUESTIONS)["answers"]
        diff = float(answers["difficulty"]["score"])
        domain = answers["domain"]["choice"]

        cheap_cut = CHEAP_DOMAINS.get(domain, self.cuts["sonnet"])
        if diff >= self.cuts["opus"]:
            tier, reason = "opus", "difficulty %.2f >= %.2f" % (diff, self.cuts["opus"])
        elif diff >= cheap_cut:
            tier, reason = "sonnet", "difficulty %.2f >= %.2f" % (diff, cheap_cut)
        else:
            tier, reason = "haiku", "difficulty %.2f < %.2f (%s)" % (diff, cheap_cut, domain)

        escalated = bool(self.escalate.search(prompt))
        if escalated and tier != "opus":
            tier, reason = "opus", "high-stakes guard matched; %s overridden" % reason

        model = TIERS[tier][0]
        # Effort is the second lever: within Opus, hard work earns `xhigh`.
        if tier == "haiku":
            effort = None
        elif tier == "opus":
            effort = "xhigh" if diff >= 2.15 else "high"
        else:
            effort = "medium" if diff < 1.60 else "high"

        return Route(tier=tier, model=model, effort=effort, difficulty=diff, domain=domain,
                     reason=reason, escalated=escalated, signals=answers)
