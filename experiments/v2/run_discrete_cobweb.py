"""TRELLIS v2 on cobweb-private's CobwebDiscreteTree (the information-theoretic
Cobweb that v1 used) instead of cobweb_cu (Fisher's category utility): the
synthetic experiment of run_unsupervised.py, unsupervised and from gold trees.

CobwebDiscreteTree is placed behind the interface v2 uses of cobweb_cu
(ifit returning the stored leaf, nodes() in pre-order, node id/children/count,
concept_codes as cobweb_cu defines them). It has no instance weights, so an
integer weight w is w insertions (exact for the integer weights here). Build
its module in cobweb-private (cmake --build build --target cobweb_discrete)
and put the build directory on the Python path.

    PYTHONPATH=cobweb-private/build python experiments/v2/run_discrete_cobweb.py
"""
import json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "cobweb-private", "build"))
import cobweb_discrete as cd  # noqa: E402

ALPHA = 1e-4
STATS = {"weights_ignored": 0}


class Node:
    __slots__ = ("_n", "id")

    def __init__(self, n):
        self._n = n
        self.id = int(n.concept_hash().rsplit("_", 1)[1])

    @property
    def children(self):
        return [Node(c) for c in self._n.children]

    @property
    def count(self):
        return self._n.count


class Tree:
    def __init__(self, attrs, seed=0):
        cd.set_random_seed(seed)
        self._t = cd.CobwebDiscreteTree(alpha=ALPHA, weight_attr=False)
        self.attr_id = {a: i for i, a in enumerate(attrs)}
        self.val_id = {}

    def _encode(self, inst):
        out = {}
        for a, v in inst.items():
            if isinstance(v, dict):
                out[self.attr_id[a]] = {self.val_id.setdefault(k, len(self.val_id)): float(w) for k, w in v.items()}
            else:
                out[self.attr_id[a]] = {self.val_id.setdefault(v, len(self.val_id)): 1.0}
        return out

    def ifit(self, inst, weight=1.0):
        # No instance weights here: an integer weight w is w insertions of the
        # instance (each after the first lands on the same leaf, an exact match).
        reps = int(round(weight))
        if abs(weight - reps) > 1e-9 or reps < 1:
            STATS["non_integer_weights"] = STATS.get("non_integer_weights", 0) + 1
            reps = max(reps, 1)
        enc = self._encode(inst)
        leaf = self._t.ifit(enc)[0]
        for _ in range(reps - 1):
            self._t.ifit(enc)
        return Node(leaf)

    def nodes(self):
        out, stack = [], [self._t.root]
        while stack:
            n = stack.pop()
            out.append(Node(n))
            stack.extend(reversed(list(n.children)))
        return out

    def concept_codes(self, alpha):
        root_av = self._t.root.av_count
        n_values = {a: max(len(d), 1) for a, d in root_av.items()}
        lg_alpha = math.lgamma(alpha)
        out = []
        for node in self.nodes():
            tot = 0.0
            for a, d in node._n.av_count.items():
                s = sum(d.values())
                terms = sum(math.lgamma(c + alpha) - lg_alpha for c in d.values())
                a_tot = n_values[a] * alpha
                tot += math.lgamma(s + a_tot) - math.lgamma(a_tot) - terms
            out.append(tot)
        return out


import trellis2.memory, trellis2.grammar  # noqa: E402
trellis2.memory.CobwebTree = Tree        # swapped in every process that imports this file
trellis2.grammar.CobwebTree = Tree

import run_unsupervised as ru  # noqa: E402
from trellis2.data import CONDITIONS, default_data_root  # noqa: E402

SEARCH = {"beam": 4, "patience": 3, "levels": 12, "sampling": (1.0, 1.0, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2)}


def job(args):
    c, s = args
    return ru.run_one(c, s, 1000, default_data_root(), SEARCH)


if __name__ == "__main__":
    out = os.path.join(HERE, "results", "discrete_cobweb")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(job, [(c, s) for c in CONDITIONS for s in (13, 17)]))
    json.dump(rows, open(os.path.join(out, "results.json"), "w"), indent=1)
    table = ru.summarise(rows)
    open(os.path.join(out, "summary.md"), "w").write(
        "TRELLIS v2 on CobwebDiscreteTree (alpha 1e-4, weight_attr False). "
        "Each cell: unsupervised / supervised on gold trees.\n\n" + table + "\n")
    print(table, f"\n{time.time() - t0:.0f}s")
