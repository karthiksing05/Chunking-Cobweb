"""Round 234: recall by construction and sentence type for CCM (20 passes) and right-branching, as r234_diag does for TRELLIS.
usage: r234_ccmdiag.py <seed> [n list]"""
import json, os, sys
os.environ["R233_GRAMMAR"] = "g2"
sys.path.insert(0, "src"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r233_ccm as CC  # noqa: E402
import r233_goldparse as GP  # noqa: E402
import r234_grammar2 as G  # noqa: E402
from r234_diag import kind  # noqa: E402


def split(trees, held):
    by, sent = {}, {}
    for s, t in zip(held, trees):
        g = G.gold(s)
        for a, b in g:
            by.setdefault(kind(s, a, b), []).append((a, b) in t)
        if g:
            sent.setdefault("simple" if not any(w in ("with", "that", "at") for w in s) else "complex", []).append(len(t & g) / len(g))
    m = lambda v: (round(sum(v) / len(v), 3), len(v))
    return {"by_kind": {k: m(v) for k, v in sorted(by.items())}, "by_sentence": {k: m(v) for k, v in sent.items()}}


def main():
    seed = int(sys.argv[1])
    ns = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1000, 4000]
    C = G.load_corpus(seed, max(ns))
    stream = [s for s, _ in C["train"]]
    held = GP.held_unseen(C, max(ns))
    out = {"rb": split([GP.rb(len(s)) for s in held], held)}
    print(json.dumps({"seed": seed, "parser": "rb", **out["rb"]}), flush=True)
    for n in ns:
        m = CC.CCM(CC.Data(stream[:n]))
        m.fit(CC.ITERS)
        out[n] = split([m.viterbi(s) for s in held], held)
        print(json.dumps({"seed": seed, "parser": "ccm20", "n": n, **out[n]}), flush=True)
    os.makedirs("experiments/v3/results/story_gen/r233/unseen_g2/diag", exist_ok=True)
    json.dump(out, open(f"experiments/v3/results/story_gen/r233/unseen_g2/diag/ccmdiag_s{seed}.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
