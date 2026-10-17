"""Generation from TRELLIS v3 memory (top-down through both hierarchies):
fraction recognized by the target grammar, novel, examples."""
import os, sys, json, random, time
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar
from trellis2.evaluation import CFG
from trellis2.trellis3 import Trellis3
cond, cfg = sys.argv[1], json.loads(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), 13)
if len(sys.argv) > 4: train = train[:int(sys.argv[4])]
L = Trellis3(seed=13, **cfg); t0 = time.time()
for e in train: L.learn(e.tokens)
g = CFG(target_grammar(cond)); seen = {tuple(e.tokens) for e in train}; rng = random.Random(0)
S = [L.generate(rng) for _ in range(n)]; ok = [s for s in S if s]
gram = [s for s in ok if g.recognizes(s)]
print(cond, "learn %.0fs" % (time.time() - t0), "generated %.2f grammatical %.3f novel %.2f gram+novel %.3f mean len %.1f" % (
    len(ok) / n, len(gram) / max(len(ok), 1), sum(tuple(s) not in seen for s in ok) / max(len(ok), 1),
    sum(tuple(s) not in seen for s in gram) / max(len(ok), 1), sum(map(len, ok)) / max(len(ok), 1)))
for s in ok[:12]: print("  ", " ".join(s), "✓" if g.recognizes(s) else "✗")
# references for scale (not methods): unigram words at a training length; bigram chain
from collections import Counter, defaultdict
uni = Counter(w for e in train for w in e.tokens); lens = [len(e.tokens) for e in train]
big = defaultdict(Counter)
for e in train:
    for a, b in zip(["<s>"] + e.tokens, e.tokens + ["</s>"]): big[a][b] += 1
def draw(c):
    r = rng.random() * sum(c.values())
    for v, m in c.items():
        r -= m
        if r <= 0: return v
U = [[draw(uni) for _ in range(rng.choice(lens))] for _ in range(n)]
B = []
for _ in range(n):
    s, w = [], "<s>"
    while len(s) < 40:
        w = draw(big[w])
        if w == "</s>": break
        s.append(w)
    B.append(s)
for name, X in (("unigram", U), ("bigram", B)):
    gr = [s for s in X if g.recognizes(s)]
    print("  ref", name, "grammatical %.3f gram+novel %.3f" % (len(gr) / n, sum(tuple(s) not in seen for s in gr) / n))
