"""Generation by substitution from TRELLIS v3 memory, per hierarchy and depth:
grammatical (target grammar), novel, both; unigram/bigram references for scale."""
import os, sys, json, random, time
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG
from trellis2.trellis3 import Trellis3
cond, cfg = sys.argv[1], json.loads(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
ntrain = int(sys.argv[4]) if len(sys.argv) > 4 else None
train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), 13)
train = train[:ntrain] if ntrain else train
L = Trellis3(seed=13, **cfg); t0 = time.time()
for e in train: L.learn(e.tokens)
g = CFG(target_grammar(cond)); seen = {tuple(e.tokens) for e in train}
mem = (L._members(L.ctx, "ctx"), L._members(L.cnt, "cnt"))
print(cond, "learn %.0fs" % (time.time() - t0), flush=True)
for up in (1, 2):
    rng = random.Random(0)
    S = [L.generate_expand(rng, members=mem[0], up=up) for _ in range(n)]; ok = [s for s in S if s]
    gram = [s for s in ok if g.recognizes(s)]
    print("  expand u%d generated %.2f grammatical %.3f novel %.2f gram+novel %.3f len %.1f distinct %.2f" % (
        up, len(ok) / n, len(gram) / max(len(ok), 1), sum(tuple(s) not in seen for s in ok) / max(len(ok), 1),
        sum(tuple(s) not in seen for s in gram) / max(len(ok), 1), sum(map(len, ok)) / max(len(ok), 1),
        len(set(map(tuple, ok))) / max(len(ok), 1)), flush=True)
    for s in ok[:3]: print("     ", " ".join(s), "✓" if g.recognizes(s) else "✗")
for via in ("ctx", "cnt"):
    for d in ("d2", "d3", "u1", "u2", "u3"):
        rng = random.Random(0)
        kw = {"depth": int(d[1])} if d[0] == "d" else {"up": int(d[1])}
        S = [L.generate_sub(rng, via=via, members=mem, **kw) for _ in range(n)]; ok = [s for s in S if s]
        gram = [s for s in ok if g.recognizes(s)]
        print("  %s %s generated %.2f grammatical %.3f novel %.2f gram+novel %.3f len %.1f" % (
            via, d, len(ok) / n, len(gram) / max(len(ok), 1), sum(tuple(s) not in seen for s in ok) / max(len(ok), 1),
            sum(tuple(s) not in seen for s in gram) / max(len(ok), 1), sum(map(len, ok)) / max(len(ok), 1)), flush=True)
        for s in ok[:3]: print("     ", " ".join(s), "✓" if g.recognizes(s) else "✗")
