"""
Smoke test: run hollow_learn with primitives_first=0 + the new maturity
gate ('root_log_prob' > -8.5) and confirm it produces metrics comparable
to the baseline that used PRIMITIVES_FIRST=200.

Runs three modes on the small grammar (fastest):
  1. baseline      — PRIMITIVES_FIRST=200, no maturity gate
  2. gate_skip     — PRIMITIVES_FIRST=0, gate='root_log_prob > -8.5', skip
  3. gate_allnone  — PRIMITIVES_FIRST=0, gate='root_log_prob > -8.5', all_or_none

Reports a comparison table. Pass = the gate modes produce non-trivial
F1 (≥ 0.5); fail = they degrade catastrophically.
"""
import os, sys, csv

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from util.cfg import TEST_GRAMMAR_SMALL, TEST_CORPUS_SMALL
from unittests.hollow_learn_test_mh import run_hollow_learn


GATE = ("root_log_prob", -8.5)
SEED = 13
CORPUS_DIR = os.path.join(_ROOT, "data", "cfg_grammar_small")
OUT_BASE   = os.path.join(_HERE, "smoke_hollow")


def compute_metrics(out_dir):
    """Read parse_accuracy.csv + step_pick_accuracy.csv → (F1, EM, step_pick)."""
    pa = os.path.join(out_dir, "parse_accuracy.csv")
    sp = os.path.join(out_dir, "step_pick_accuracy.csv")
    tp = fp = fn = em = n = 0
    with open(pa) as f:
        r = csv.reader(f); next(r)
        for row in r:
            t, p_, e = int(row[-3]), int(row[-2]), int(row[-1])
            tp += t; fp += p_; fn += e
            if p_ == 0 and e == 0 and t > 0: em += 1
            n += 1
    P = tp / max(tp+fp, 1); R = tp / max(tp+fn, 1)
    F = 2*P*R / max(P+R, 1e-9)
    EM = em / max(n, 1)
    sp_ok = sp_n = 0
    with open(sp) as f:
        for row in csv.DictReader(f):
            if row.get("is_gold") == "1": sp_ok += 1
            sp_n += 1
    return F, EM, sp_ok / max(sp_n, 1)


configs = [
    ("baseline",     {"primitives_first": 200, "maturity_gate": None}),
    ("gate_skip",    {"primitives_first":   0, "maturity_gate": GATE, "gate_mode": "skip"}),
    ("gate_allnone", {"primitives_first":   0, "maturity_gate": GATE, "gate_mode": "all_or_none"}),
]

results = {}
for name, kwargs in configs:
    out_dir = os.path.join(OUT_BASE, name)
    print(f"\n{'='*70}\n=== {name}  kwargs={kwargs}\n{'='*70}")
    run_hollow_learn(
        corpus_dir=CORPUS_DIR,
        out_dir=out_dir,
        grammar=TEST_GRAMMAR_SMALL,
        corpus=TEST_CORPUS_SMALL,
        seed=SEED,
        **kwargs,
    )
    f1, em, sp = compute_metrics(out_dir)
    results[name] = (f1, em, sp)

print(f"\n\n{'='*70}\n=== SMOKE TEST RESULTS ===\n{'='*70}")
print(f"{'Mode':<15s} {'F1':>10s} {'EM':>10s} {'StepPick':>10s}")
for name, (f, e, s) in results.items():
    print(f"{name:<15s} {100*f:>9.1f}% {100*e:>9.1f}% {100*s:>9.1f}%")
print()
