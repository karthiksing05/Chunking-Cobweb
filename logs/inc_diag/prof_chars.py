import sys, cProfile, pstats, time
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/src")
from trellis2.characters import default_ids_path, load_characters
from trellis2.incremental import IncrementalLearner
chars, _ = load_characters(default_ids_path(), seed=13)
L = IncrementalLearner(seed=13, config={"lookahead_dissolve": True})
for c in chars[:40]:
    L.observe(c.tokens)
pr = cProfile.Profile(); pr.enable(); t = time.time()
for c in chars[40:80]:
    L.observe(c.tokens)
pr.disable(); print("40 chars", time.time() - t, "K", L.tables.K, "len", sum(len(c.tokens) for c in chars[40:80]) / 40)
pstats.Stats(pr).sort_stats("cumulative").print_stats(40)
