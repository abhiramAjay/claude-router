"""Train a routing head on frozen laya embeddings.

    python training/train_head.py                 # train, evaluate, save
    python training/train_head.py --curve         # accuracy vs training-set size

laya ships a runtime only (`predict`, `system_one`) -- there is no `fit`, so the
checkpoint itself cannot be fine-tuned. What it does expose is
`embed_fn_from_agent`, the mean-pooled encoder, at ~38 ms per prompt. That is
enough: freeze laya as a feature extractor and train a small classifier on top.
The result keeps the properties that made laya attractive (local, ~100 ms, $0
per decision, works offline) while learning the tier boundaries from data
instead of from prompt wording, which is where the hand-tuned approach stalled.

Honest evaluation
-----------------
Training data is synthetic (see generate_data.py). The test set is NOT: it is
the 40 briefs in `delegated_corpus.json`, captured from real Claude Code
sessions, and no synthetic example ever enters it. Training on generated data
and scoring on real data is the only way to find out whether the generator
produced something that looks like the real distribution -- a model that scores
well on held-out *synthetic* data has only proven the generator is consistent
with itself.

Baselines to beat, measured on those same 40 real briefs:

    current shipped router (difficulty score, raw)   20/40   50%
    strip + category question (hand-tuned)           27/40   68%
    head on laya embeddings, 27 real examples        60% +/- 18 (5-fold CV)
"""
import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BRIEFS = os.path.join(HERE, "briefs.jsonl")
MODEL_OUT = os.path.join(ROOT, "claude_router", "routing_head.joblib")
REAL = os.path.join(ROOT, "delegated_corpus.json")

# The generator speaks in task kinds; the router speaks in tiers.
KIND_TIER = {"lookup": "haiku", "edit": "sonnet", "design": "opus"}
CLASSES = ["lookup", "edit", "design"]


def load_training() -> Tuple[List[str], List[str], List[str]]:
    if not os.path.exists(BRIEFS):
        sys.exit("no training data. Run training/generate_data.py first.")
    texts, labels, payloads = [], [], []
    with open(BRIEFS) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("brief") and r.get("tier") in KIND_TIER:
                texts.append(r["brief"])
                labels.append(r["tier"])
                payloads.append(r.get("payload", "?"))
    return texts, labels, payloads


def load_real() -> Tuple[List[str], List[str]]:
    """The 40 real captures, as (brief, kind). Never used for training."""
    tier_kind = {v: k for k, v in KIND_TIER.items()}
    corpus = json.load(open(REAL))
    return [c["delegated"] for c in corpus], [tier_kind[c["want"]] for c in corpus]


def embedder():
    from laya import embed_fn_from_agent, load
    return embed_fn_from_agent(load())


def report(name: str, y_true: List[str], y_pred: List[str]) -> float:
    import collections
    hits = sum(a == b for a, b in zip(y_true, y_pred))
    rank = {"lookup": 0, "edit": 1, "design": 2}
    over = sum(rank[p] > rank[t] for t, p in zip(y_true, y_pred))
    under = sum(rank[p] < rank[t] for t, p in zip(y_true, y_pred))
    print("%-28s %2d/%-3d %5.0f%%   overspend %2d  underspend %2d   picks %s"
          % (name, hits, len(y_true), 100 * hits / len(y_true), over, under,
             dict(collections.Counter(y_pred))))
    return hits / len(y_true)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="train_head")
    ap.add_argument("--curve", action="store_true",
                    help="accuracy vs training-set size -- says whether more data would help")
    ap.add_argument("--C", type=float, default=0.3, help="inverse regularisation strength")
    args = ap.parse_args(argv)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import joblib

    texts, labels, payloads = load_training()
    if len(texts) < 60:
        print("only %d training briefs -- generate more before trusting any of this"
              % len(texts), file=sys.stderr)
    import collections
    print("training data: %d briefs  %s" % (len(texts), dict(collections.Counter(labels))))
    print("payload mix:   %s\n" % dict(collections.Counter(payloads)))

    emb = embedder()
    X = emb(texts)
    y = np.array(labels)
    Xr_texts, yr = load_real()
    Xr = emb(Xr_texts)

    if args.curve:
        rng = np.random.RandomState(0)
        print("%-10s %-22s %s" % ("n", "cross-val (synthetic)", "REAL 40 (held out)"))
        for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
            n = max(30, int(len(X) * frac))
            idx = rng.permutation(len(X))[:n]
            clf = LogisticRegression(max_iter=5000, C=args.C).fit(X[idx], y[idx])
            cv = cross_val_score(LogisticRegression(max_iter=5000, C=args.C),
                                 X[idx], y[idx], cv=min(5, n // 10 or 2)).mean()
            real = (clf.predict(Xr) == np.array(yr)).mean()
            print("%-10d %-22.0f%% %.0f%%" % (n, 100 * cv, 100 * real))
        return 0

    clf = LogisticRegression(max_iter=5000, C=args.C).fit(X, y)
    cv = cross_val_score(LogisticRegression(max_iter=5000, C=args.C), X, y, cv=5)
    print("cross-val on synthetic:  %.0f%% (+/- %.0f)   <- only says the generator is self-consistent"
          % (100 * cv.mean(), 100 * cv.std()))
    print()
    report("head, on the REAL 40", yr, list(clf.predict(Xr)))
    print("%-28s %2d/%-3d %5.0f%%" % ("  baseline: shipped router", 20, 40, 50))
    print("%-28s %2d/%-3d %5.0f%%" % ("  baseline: hand-tuned", 27, 40, 68))

    # Does it still lean on length? If the coefficient story were "longer means
    # harder", accuracy would collapse on the long-payload real briefs.
    lens = np.array([len(t) for t in Xr_texts])
    pred = clf.predict(Xr)
    for label, mask in (("short briefs (<median)", lens < np.median(lens)),
                        ("long briefs  (>median)", lens >= np.median(lens))):
        acc = (pred[mask] == np.array(yr)[mask]).mean()
        print("  %-26s %.0f%%" % (label, 100 * acc))

    joblib.dump({"clf": clf, "classes": CLASSES, "kind_tier": KIND_TIER}, MODEL_OUT)
    print("\nsaved -> %s" % MODEL_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
