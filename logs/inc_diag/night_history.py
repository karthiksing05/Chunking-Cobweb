"""The night on the first N characters: its history (starts, moves, bits) and its symbols' members."""
import os, sys
from collections import Counter, defaultdict
snap = sys.argv[1]
sys.path.insert(0, os.path.join(snap, "src"))
from trellis2.characters import default_ids_path, load_characters  # noqa
from trellis2.unsupervised import UnsupervisedLearner  # noqa
n = int(sys.argv[2])
chars, _ = load_characters(default_ids_path(), seed=13)
L = UnsupervisedLearner(seed=13, workers=1)
for c in chars[:n]:
    L.observe(c.tokens)
g = L.sleep()
for h in L.history:
    print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in h.items() if k != "sentences"})
mem = L.model.memory
labels = L.model.labels if hasattr(L.model, "labels") else None
print("K", g.K, "total", g.info["total bits"])
try:
    sym = L.model.symbols_of_elements() if hasattr(L.model, "symbols_of_elements") else None
except Exception:
    sym = None
members = defaultdict(Counter)
for t, toks in zip(L.trees, L.experiences):
    for (i, j), A in t.label.items():
        members[A]["".join(toks[i:j]) if j - i == 1 else f"<{j - i}>"] += 1
for A in sorted(members, key=str):
    print(A, members[A].most_common(14))
