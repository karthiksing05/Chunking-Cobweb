"""Debug: where gap queries sort (_query_path) and what the concepts on the path hold."""
import json
import random
import sys
sys.path.insert(0, "src")
from trellis2.stories import read_sentences, simple_sentences  # noqa: E402
from trellis2.trellis3 import Trellis3, FILL  # noqa: E402

cfg, seed, n = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
corpus = simple_sentences(read_sentences(), vocab_size=250)
random.Random(seed).shuffle(corpus)
L = Trellis3(seed=seed, **cfg)
for s in corpus[:n]:
    L.learn(s)
def desc(nd):
    f = L.ctx.passive(nd, FILL)
    top = sorted(f.items(), key=lambda kv: -kv[1])[:6]
    show = [(k[1] if k[0] == "w" else " ".join(L._type_key(L.records[k[1]])), round(v, 1)) for k, v in top]
    win = {a: sorted(nd.av.get(a, {}).items(), key=lambda kv: -kv[1])[:3] for a in ("l2", "l1", "r1", "r2")}
    return {"id": nd.id, "count": round(nd.count, 1), "fills": show, "win": {a: [(u, round(c, 1)) for u, c in v] for a, v in win.items()}}
for q in ({"l2": "<s>", "l1": "the", "r1": "</s>", "r2": "</s>"}, {"l2": "he", "l1": "was", "r1": "</s>", "r2": "</s>"},
          {"l2": "<s>", "l1": "lily", "r1": "and", "r2": "laughed"}):
    path = L._query_path(q)
    print(json.dumps(q))
    for nd in path[-4:]:
        print("   ", json.dumps(desc(nd)))
