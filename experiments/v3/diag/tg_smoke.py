import sys, time, json
sys.path.insert(0, "src")
from trellis2.chess import load_positions
from trellis2.trellis_graph import GraphTrellis, ChessDomain
cfg = json.loads(sys.argv[1]); n = int(sys.argv[2])
P = load_positions("data/chess/positions_1500_ply30.fen", n)
L = GraphTrellis(ChessDomain(), seed=13, **cfg)
t0 = time.time()
for k, p in enumerate(P, 1):
    t = time.time(); r = L.learn(p)
    if k in (1, 2, 5, 10, 20, 40, 80, 150, 300) or k == n:
        print(k, "pieces", len(r["words"]), "candidates", len(r["comps"]), "s", round(time.time() - t, 2),
              "top chunks", len(L.chunks(r)), L.chunks(r)[:4], flush=True)
print("total s", round(time.time() - t0), L.stats, flush=True)
