"""How concentrated inside-outside posteriors are as learning proceeds: mean
posterior of the MBR-chosen composites and their share of the sentence's
total mass (n - 1), per block of training experiences."""
import os, sys, json
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root
from trellis2.trellis3 import Trellis3
cond, cfg = sys.argv[1], json.loads(sys.argv[2]); every = int(sys.argv[3]) if len(sys.argv) > 3 else 40
train, _ = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), 13)
L = Trellis3(seed=13, **cfg)
acc = []
for t, e in enumerate(train, 1):
    r = L.parse(e.tokens)
    if r["chosen"]:
        n = len(e.tokens); ch = [r["post"][k] for k in r["chosen"]]
        acc.append((sum(ch) / len(ch), sum(ch) / (n - 1)))
    L.learn(e.tokens)
    if t % every == 0:
        print(cond, t, "mean chosen posterior %.3f  chosen share of mass %.3f" % (
            sum(a for a, _ in acc) / len(acc), sum(b for _, b in acc) / len(acc)), flush=True)
        acc = []
