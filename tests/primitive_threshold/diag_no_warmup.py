"""
Diagnose the gate failure on grammar_med:
  A. baseline             primitives_first=200, no maturity_gate
  B. no_warmup_no_gate    primitives_first=0,   threshold="converge" (always stable)
  C. no_warmup_with_gate  primitives_first=0,   maturity_gate=('root_log_prob', -8.5)
  D. small_warmup_w_gate  primitives_first=50,  maturity_gate=('root_log_prob', -8.5)

If B beats C, the gate itself is the problem (it's too restrictive).
If B matches A, removing warmup is fine but the gate breaks it.
If B is worse than A, warmup matters even without a gate.

Runs ONE seed each (SEED=13) on grammar_med — fast.
"""
import os, sys, csv
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from util.cfg import TEST_GRAMMAR_MED, TEST_CORPUS_MED
from unittests.hollow_learn_test_mh import run_hollow_learn

CORPUS_DIR = os.path.join(_ROOT, "data", "cfg_grammar_med")
OUT_BASE   = os.path.join(_HERE, "diag_no_warmup")
SEED       = 13


def metrics(out_dir):
    pa = os.path.join(out_dir, "parse_accuracy.csv")
    sp = os.path.join(out_dir, "step_pick_accuracy.csv")
    tp = fp = fn = em = n = 0
    with open(pa) as f:
        r = csv.reader(f); next(r)
        for row in r:
            t, p_, e = int(row[-3]), int(row[-2]), int(row[-1])
            tp += t; fp += p_; fn += e
            if p_==0 and e==0 and t>0: em += 1
            n += 1
    P = tp/max(tp+fp,1); R = tp/max(tp+fn,1)
    F = 2*P*R/max(P+R, 1e-9)
    EM = em/max(n, 1)
    sp_ok = sp_n = 0
    with open(sp) as f:
        for row in csv.DictReader(f):
            if row.get("is_gold") == "1": sp_ok += 1
            sp_n += 1
    return F, EM, sp_ok/max(sp_n, 1)


configs = [
    ("A_baseline",            {"primitives_first": 200, "maturity_gate": None}),
    ("B_no_warmup_no_gate",   {"primitives_first":   0, "maturity_gate": None,
                                "gate_mode": "skip"}),
    ("C_no_warmup_with_gate", {"primitives_first":   0,
                                "maturity_gate": ("root_log_prob", -8.5),
                                "gate_mode": "skip"}),
    ("D_small_warmup_w_gate", {"primitives_first":  50,
                                "maturity_gate": ("root_log_prob", -8.5),
                                "gate_mode": "skip"}),
]

results = {}
for name, kwargs in configs:
    out_dir = os.path.join(OUT_BASE, name)
    print(f"\n{'='*70}\n=== {name}  kwargs={kwargs}\n{'='*70}")
    run_hollow_learn(
        corpus_dir=CORPUS_DIR, out_dir=out_dir,
        grammar=TEST_GRAMMAR_MED, corpus=TEST_CORPUS_MED,
        seed=SEED, viz_intermediates=False,
        **kwargs,
    )
    f1, em, sp = metrics(out_dir)
    results[name] = (f1, em, sp)

print(f"\n\n{'='*70}\n=== DIAG RESULTS (seed={SEED} on grammar_med) ===\n{'='*70}")
print(f"{'Mode':<25s} {'F1':>10s} {'EM':>10s} {'StepPick':>10s}")
for name, (f, e, s) in results.items():
    print(f"{name:<25s} {100*f:>9.1f}% {100*e:>9.1f}% {100*s:>9.1f}%")
print()
