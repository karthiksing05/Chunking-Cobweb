"""Learning curve of TRELLIS v3: held-out omission (and right-branching) after
every ``every`` training experiences, to tell a bad start from a drift."""
import os, sys, json
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, Tree
from trellis2.evaluation import BracketTally
from trellis2.trellis3 import Trellis3
cond, cfg = sys.argv[1], json.loads(sys.argv[2])
every = int(sys.argv[3]) if len(sys.argv) > 3 else 50
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 13
train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
test = test[:60]
L = Trellis3(seed=seed, **cfg)
def ev(tag):
    te, rb = BracketTally(), BracketTally()
    for e in test:
        n = len(e.tokens); te.add(e.tree, L.parse(e.tokens)["tree"])
        rb.add(e.tree, Tree(n, {(i, n): i + 1 for i in range(n - 1)}, {}, [(0, n)]))
    print(cond, tag, "omission %.3f rb %.3f" % (te.omission, rb.omission), flush=True)
for t, e in enumerate(train, 1):
    L.learn(e.tokens)
    if t % every == 0 or t == len(train): ev(t)
for e in test[:4]:
    print("  ours", L.parse(e.tokens)["tree"].to_string(e.tokens)); print("  gold", e.tree.to_string(e.tokens))
