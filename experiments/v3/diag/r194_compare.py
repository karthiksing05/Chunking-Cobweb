"""Round 194 pre-test scoring: matched judged dumps, A vs B (same seed, n, prompts), with story_frontier.py's measures
(gram over new sentences, tri = share of new sentences with an unstored trigram, gram|tri).

Usage (compute node): r194_compare.py <label A> <glob A, with {s} for seed> <label B> <glob B> <seeds comma-separated> <n>
"""
import json
import sys

sys.path.insert(0, "experiments/v3")
from story_frontier import Train, grams, train_sentences  # noqa: E402


def stats(sents, judged, T):
    new = [(s, j) for s, j in zip(sents, judged) if s not in T.whole]
    tri = [any(g not in T.tri for g in grams(s, 3)) for s, _ in new]
    ok = [j > 0.5 for _, j in new]
    m = lambda v: sum(v) / len(v) if v else float("nan")
    return m(ok), m(tri), m([a for a, t in zip(ok, tri) if t]), len(new)


la, ga, lb, gb, seeds, n = sys.argv[1:7]
for s in map(int, seeds.split(",")):
    T = Train(train_sentences(s, int(n)))
    for lab, g in ((la, ga), (lb, gb)):
        f = g.format(s=s)
        rows = [json.loads(l) for l in list(open(f, encoding="utf-8"))[1:]]
        sc = json.load(open(f[:-len(".jsonl")] + ".scores.json", encoding="utf-8"))
        gr, t, gt, nn = stats([tuple(r["gen"].split()) for r in rows], sc, T)
        print(f"s{s} {lab:24s} new {nn:5d} gram {gr:.3f} tri {t:.3f} gram|tri {gt:.3f}", flush=True)
