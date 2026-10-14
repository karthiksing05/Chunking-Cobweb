"""Characters, night against incremental on the same first N characters:
metrics, analyses against gold, generated samples, the incremental symbols.
Usage: python chars_compare.py SNAPSHOT_DIR {night|inc} N [config-json]"""
import json, os, sys, time
import numpy as np
snap = sys.argv[1]
sys.path.insert(0, os.path.join(snap, "src")); sys.path.insert(0, os.path.join(snap, "experiments", "v2"))
from trellis2.characters import default_ids_path, load_characters, parse_prefix, placements  # noqa
import run_characters as RCH  # noqa

mode, n = sys.argv[2], int(sys.argv[3])
cfg = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
chars, _ = load_characters(default_ids_path(), seed=13)
train, test = chars[:n], chars[2000:2300]
real = {parse_prefix(c.tokens): c.char for c in chars}
attested = set()
for c in chars:
    placements(parse_prefix(c.tokens), attested)
t0 = time.time()
if mode == "night":
    from trellis2.unsupervised import UnsupervisedLearner
    L = UnsupervisedLearner(seed=13, workers=1)
    for c in train:
        L.observe(c.tokens)
    L.sleep()
    trees = L.trees
else:
    from trellis2.incremental import IncrementalLearner
    L = IncrementalLearner(seed=13, config=cfg)
    for k, c in enumerate(train, 1):
        L.observe(c.tokens)
        if k % 50 == 0:
            print(f"  {k} {L.tables.bits():.0f} bits K={L.tables.K} ({time.time() - t0:.0f}s)", flush=True)
    trees = L.stored_trees()
print(f"{mode} N={n} learned in {time.time() - t0:.0f}s", flush=True)
print("TOTAL BITS", (L.model.grammar.info["total bits"] if mode == "night" else L.grammar.info["total bits"]), "K", (L.model.grammar.K if mode == "night" else L.grammar.K), flush=True)
out = RCH.evaluate(L.chart, L.generate, train, test, real, attested, 500, 13)
g = out["generation"]
print("omission %.3f commission %.3f test %.2f" % (out["omission"], out["commission"], out["test_bits_per_character"]))
print("generation", json.dumps(g, ensure_ascii=False))
print("whole trees", np.mean([len(t.roots) == 1 for t in trees]),
      "gold-exact", np.mean([t.brackets() == c.tree.brackets() and len(t.roots) == 1 for t, c in zip(trees, train)]))


def br(tree, toks):
    spans = set(tree.composite_spans()) | {r for r in tree.roots if r[1] - r[0] >= 2}
    opens = {i: sum(1 for a, b in spans if a == i) for i in range(len(toks))}
    closes = {j: sum(1 for a, b in spans if b == j) for j in range(1, len(toks) + 1)}
    out = []
    for i, t in enumerate(toks):
        out.append("[" * opens[i] + t + "]" * closes[i + 1])
    return " ".join(out) + ("  (forest %d)" % len(tree.roots) if len(tree.roots) > 1 else "")


for t, c in list(zip(trees, train))[:25]:
    print(c.char, " ".join(c.tokens), "| gold", br(c.tree, c.tokens), "| learned", br(t, c.tokens))
print("examples:")
for e in out.get("examples", [])[:20] if isinstance(out.get("examples"), list) else list(out.get("examples", {}).items())[:20]:
    print("  ", e)
if mode == "inc":
    print("stats", dict(L.stats))
    for A in sorted(L.members):
        keys = sorted({k for k in L.rec_by_sym.get(A, ())}, key=str)
        print("S%d" % A, len(keys), [k[1] if k[0] == "P" else k for k in keys][:14])
