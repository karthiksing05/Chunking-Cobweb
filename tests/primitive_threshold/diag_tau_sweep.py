"""
τ sweep on grammar_med (1 seed each). Goal: find a maturity-gate
threshold that beats no-gate (B = F1 67.9%) at primitives_first=0.

If best-F1 ≤ 67.9, the gate concept is not viable here — we should
keep the warmup OR find a different heuristic. If some τ beats 67.9,
that τ is our production setting.
"""
import os, sys, csv
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from util.cfg import TEST_GRAMMAR_MED, TEST_CORPUS_MED
from unittests.hollow_learn_test_mh import run_hollow_learn

CORPUS_DIR = os.path.join(_ROOT, "data", "cfg_grammar_med")
OUT_BASE   = os.path.join(_HERE, "diag_tau_sweep")
SEED       = 13

# Sweep across the operating points from the research test plus
# more permissive values, to see where the gate stops hurting.
TAUS = [-8.5, -10.0, -12.0, -15.0, -20.0, -50.0]


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


results = []
for tau in TAUS:
    name = f"tau_{tau}"
    out_dir = os.path.join(OUT_BASE, name)
    print(f"\n{'='*70}\n=== {name}  τ={tau}\n{'='*70}")
    run_hollow_learn(
        corpus_dir=CORPUS_DIR, out_dir=out_dir,
        grammar=TEST_GRAMMAR_MED, corpus=TEST_CORPUS_MED,
        seed=SEED, viz_intermediates=False,
        primitives_first=0,
        maturity_gate=("root_log_prob", tau),
        gate_mode="skip",
    )
    f1, em, sp = metrics(out_dir)
    results.append((tau, f1, em, sp))

print(f"\n\n{'='*70}\n=== τ SWEEP RESULTS (seed=13 on grammar_med, PF=0) ===\n{'='*70}")
print(f"  Baseline (A):       F1=81.7%  EM=65.0%  SP=91.7%")
print(f"  No-gate (B):        F1=67.9%  EM=40.0%  SP=89.7%")
print(f"{'τ':>10s} {'F1':>10s} {'EM':>10s} {'StepPick':>10s}")
for tau, f, e, s in results:
    flag = "  ← beats B" if f > 0.679 else ""
    print(f"{tau:>10.1f}  {100*f:>8.1f}%  {100*e:>8.1f}%  {100*s:>8.1f}%{flag}")
print()
