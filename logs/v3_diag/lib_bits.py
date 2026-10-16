"""Where the library's bits go: per held-out sentence, the full read vs reading
every word as its own piece (no chunks), split into naming / token / stop bits."""
import os, sys, math, random
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2.library import ChunkLibrary, BOS
domain = sys.argv[1]
if domain == "stories":
    from trellis2.stories import read_sentences, simple_sentences
    sim = simple_sentences(read_sentences(), 100, 3, 6); random.Random(13).shuffle(sim)
    test, train = sim[:200], sim[300:2300]
else:
    from trellis2 import load_corpus, v1_split
    from trellis2.data import CONDITIONS, default_data_root
    tr, te = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[domain])), 13)
    train, test = [e.tokens for e in tr], [e.tokens for e in te]
L = ChunkLibrary(seed=13, depth=6)
for x in train: L.learn(x)
full = name = tok = 0.0
for x in test:
    full += L.read(x)["bits"]
    for i, w in enumerate(x):
        pth = L.ctx.categorize(L._ctx_instance(x, i, i + 1))[:7]
        left = {f"l{k}": x[i - k] if i - k >= 0 else BOS for k in (1, 2)}
        reach = L._reach_bits(pth, left)
        best = min((reach[d] + L._ref_bits(v, w), d) for d, v in enumerate(pth))
        name += reach[best[1]]; tok += L._ref_bits(pth[best[1]], w)
n = len(test); words = sum(map(len, test))
print(domain, f"full read {full/n:.1f} | words alone: naming {name/n:.1f} + token {tok/n:.1f} + stop {words/n:.1f} = {(name+tok+words)/n:.1f} bits/sentence ({words/n:.1f} words)")
