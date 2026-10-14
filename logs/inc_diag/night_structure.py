import os, sys, pickle
from collections import Counter
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/src")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root
from trellis2.unsupervised import UnsupervisedLearner
cond, seed = sys.argv[1], int(sys.argv[2])
ex = load_corpus(os.path.join(default_data_root(), CONDITIONS[cond]))
train, _ = v1_split(ex, seed)
L = UnsupervisedLearner(seed=seed, workers=int(sys.argv[3]) if len(sys.argv) > 3 else 1)
for e in train:
    L.observe(e.tokens)
g = L.sleep()
print(g.describe(top=6))
for h in L.history:
    print(h["stage"], h["move"], round(h["bits"], 1))
pickle.dump({"trees": L.trees, "experiences": L.experiences, "elem_symbol": list(map(int, L.model.grammar.elem_symbol)),
             "history": L.history}, open(f"logs/inc_pickles/night_{cond}_s{seed}.pkl", "wb"))
for t, toks in list(zip(L.trees, L.experiences))[:15]:
    print(t.to_string(toks))
