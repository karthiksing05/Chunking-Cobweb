"""Generation benchmark for TRELLIS v2: one pass over the training sentences,
then expansive generation (and, with k > 1, generation with the parser as
critic). Reports grammatical (target grammar), grammatical and new, distinct.
Held-out omission is printed as a side number."""
import os, sys, json, random, time
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from trellis2 import load_corpus, v1_split
from trellis2.data import CONDITIONS, default_data_root, target_grammar, Tree
from trellis2.evaluation import CFG, BracketTally
from trellis2.trellis3 import Trellis3
tag, cfg = sys.argv[1], json.loads(sys.argv[2])
conds = sys.argv[3].split(",") if len(sys.argv) > 3 else ["small", "med", "large", "term_low", "term_med", "term_high"]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 13
ks = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [1, 8]
by = sys.argv[6] if len(sys.argv) > 6 else "weight"
opts = sys.argv[7].split(",") if len(sys.argv) > 7 else []
rh = "hint" in opts
up = next((int(o[2:]) for o in opts if o.startswith("up")), 0)
mm = next((int(o[1:]) for o in opts if o.startswith("m") and o[1:].isdigit()), None)
tree = "tree" in opts
choose = next((o[7:] for o in opts if o.startswith("choose=")), "leaf")
su = next((int(o[5:]) for o in opts if o.startswith("level")), 0)
n = 300
for cond in conds:
    train, test = v1_split(load_corpus(os.path.join(default_data_root(), CONDITIONS[cond])), seed)
    L = Trellis3(seed=seed, **cfg); t0 = time.time()
    for e in train: L.learn(e.tokens)
    g = CFG(target_grammar(cond)); seen = {tuple(e.tokens) for e in train}
    te = BracketTally()
    for e in test[:60]: te.add(e.tree, L.parse(e.tokens)["tree"])
    mem = L._members(L.ctx, "ctx") if not tree else None
    ix = L.concept_index() if tree else None
    row = {"by": by + ("+hint" if rh else ""), "up": up, "min members": mm, "tree": tree, "level": su, "choose": choose, "tag": tag, "cond": cond, "seed": seed, "learn s": round(time.time() - t0), "omission (side)": round(te.omission, 3)}
    for k in ks:
        rng = random.Random(seed)
        def one():
            if k > 1:
                return L.generate_best(rng, members=mem, k=k, by=by)
            if tree:
                return L.generate_tree(rng, index=ix, level=su, choose=choose)
            return L.generate_expand(rng, members=mem, up=up, by=by, right_hint=rh, min_members=mm)
        S = [s for s in (one() for _ in range(n)) if s]
        gram = [s for s in S if g.recognizes(s)]
        row[f"k{k} grammatical"] = round(len(gram) / max(len(S), 1), 3)
        row[f"k{k} gram+new"] = round(sum(tuple(s) not in seen for s in gram) / max(len(S), 1), 3)
        row[f"k{k} distinct"] = round(len(set(map(tuple, S))) / max(len(S), 1), 3)
        row[f"k{k} examples"] = [" ".join(s) + (" ✓" if g.recognizes(s) else " ✗") for s in S[:4]]
    print(json.dumps(row), flush=True)
