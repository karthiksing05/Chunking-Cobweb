"""Round 234 diagnosis: where TRELLIS's parse of grammar-2 sentences fails. usage: r234_diag.py <seed> <arm full|nochunk> [n list]
Recall split by construction of the gold span and by sentence type (simple = no with/that/at); mean posterior of gold spans."""
import json, os, sys
os.environ["R233_GRAMMAR"] = "g2"
sys.path.insert(0, "src"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r233_goldparse as GP  # noqa: E402
import r234_grammar2 as G  # noqa: E402
from trellis2.trellis3_r225 import Trellis3R225  # noqa: E402

VERBS = {w for v, x in G.G.VERBS.items() for w in (v, x[0])}


def kind(s, a, b):
    w = s[a]
    if a == 0:
        return "subjNP" + ("+mod" if any(t in ("with", "that") for t in s[a:b]) else "")
    if w in ("with", "at", "that"):
        return {"with": "PP-with", "at": "PP-at", "that": "RC"}[w]
    if w in VERBS:
        return "VP-rc" if a > 0 and "that" in s[:a] and not any(t in VERBS for t in s[s.index("that") + 1:a]) else "VP"
    if w in G.G.PREPS:
        return "PP-verb"
    return "NP-in" + ("+with" if "with" in s[a:b] else "")


def main():
    seed, arm = int(sys.argv[1]), sys.argv[2]
    ns = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1000, 4000]
    cfg = json.load(open("experiments/v3/results/story_gen/r214p01/cfg_w5.json"))
    C = G.load_corpus(seed, max(ns))
    stream = [s for s, _ in C["train"]]
    held = GP.held_unseen(C, max(ns))
    ck, kw = GP.ARMS[arm]
    L = Trellis3R225(seed=seed, **{**cfg, **ck}, **kw)
    done, out = 0, {}
    for n in ns:
        for s in stream[done:n]:
            L.learn(s)
        done = n
        by, sent, post_g, dump = {}, {}, [], []
        for s in held:
            r = L.parse(s)
            t = GP.tree_spans(r["chosen"], len(s))
            g = G.gold(s)
            sp = {}
            for (i, k, j), p in r["post"].items():
                sp[(i, j)] = sp.get((i, j), 0) + p
            st = "simple" if not any(w in ("with", "that", "at") for w in s) else "complex"
            for a, b in g:
                kd = kind(s, a, b)
                by.setdefault(kd, []).append((a, b) in t)
                post_g.append(sp.get((a, b), 0))
            if g:
                sent.setdefault(st, []).append(len(t & g) / len(g))
            if len(dump) < 30:
                dump.append({"s": " ".join(s), "tree": sorted(t), "gold": sorted(g),
                             "gold_post": {f"{a},{b}": round(sp.get((a, b), 0), 2) for a, b in sorted(g)}})
        m = lambda v: (round(sum(v) / len(v), 3), len(v))
        out[n] = {"by_kind": {k: m(v) for k, v in sorted(by.items())}, "by_sentence": {k: m(v) for k, v in sent.items()},
                  "gold_post_mean": round(sum(post_g) / len(post_g), 3), "dump": dump}
        print(json.dumps({"seed": seed, "arm": arm, "n": n, "by_kind": out[n]["by_kind"], "by_sentence": out[n]["by_sentence"],
                          "gold_post_mean": out[n]["gold_post_mean"]}), flush=True)
    os.makedirs("experiments/v3/results/story_gen/r233/unseen_g2/diag", exist_ok=True)
    json.dump(out, open(f"experiments/v3/results/story_gen/r233/unseen_g2/diag/diag_s{seed}_{arm}.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
