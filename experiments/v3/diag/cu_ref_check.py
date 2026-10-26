"""Default cobweb_cu (insert-only ifit, as built from the worktree) against
tests/trellis2/reference_cobweb.py: the test of tests/trellis2/test_cobweb.py,
plus random weights and bag shares (squares that glibc's pow rounds
differently from x*x)."""
import random
import sys

sys.path.insert(0, "/coc/flash10/ksingara3/cobweb-private-trellis/build-trellis")
sys.path.insert(0, "tests/trellis2")
import cobweb_cu  # noqa: E402
from reference_cobweb import CobwebTree as Reference  # noqa: E402

print("cobweb_cu from", cobweb_cu.__file__)


def shape(tree):
    return [(n.id, n.parent.id if n.parent is not None else None, [c.id for c in n.children],
             n.count, {a: dict(d) for a, d in n.av.items()}) for n in tree.nodes()]


def instances(n, seed, rough):
    rng = random.Random(seed)
    for _ in range(n):
        words = [rng.choice("pqrst") for _ in range(rng.randint(1, 4))]
        bag = {}
        for w_ in words:
            bag[w_] = bag.get(w_, 0.0) + (rng.random() if rough else 1.0 / len(words))
        if rough:
            t = sum(bag.values())
            bag = {u: p / t for u, p in bag.items()}
        w = rng.random() + 0.01 if rough else rng.choice([1.0, 0.5, 2.0])
        yield {"a": rng.choice("xyz"), "b": rng.choice([1, 2, "q"]), "bag": bag}, w


for rough in (False, True):
    for seed in range(5 if rough else 1):
        data = list(instances(2000, 5 + seed, rough))
        ref, cu = Reference(("a", "b", "bag"), seed=7), cobweb_cu.CobwebCUTree(["a", "b", "bag"], seed=7)
        same = [ref.ifit(x, w).id for x, w in data] == [cu.ifit(x, w).id for x, w in data] and shape(ref) == shape(cu)
        print(f"{'random weights' if rough else 'test_cobweb data'} seed {seed}: identical={same}", flush=True)
