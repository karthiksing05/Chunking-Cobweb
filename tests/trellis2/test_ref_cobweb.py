"""RefCobwebTree (update on read): with a fixed name table it is the base tree;
after names change, a synced node's counts are its records' current names summed."""
import random
import sys

sys.path.insert(0, "src")
from trellis2.incremental_cobweb import IncrementalCobwebTree  # noqa: E402
from trellis2.ref_cobweb import NameTable, Ref, RefCobwebTree  # noqa: E402

ATTRS = ("a", "b", "L", "R")


def rand_bag(rng, k=6):
    ids = rng.sample(range(k), rng.randint(1, 3))
    ws = [rng.random() + 0.1 for _ in ids]
    s = sum(ws)
    return {f"c{i}": w / s for i, w in zip(ids, ws)}


def stream(rng, n, keys):
    out = []
    for _ in range(n):
        x = {"a": rng.choice("xyz"), "b": rand_bag(rng),
             "L": Ref(rng.choice(keys)), "R": Ref(rng.choice(keys))}
        out.append((x, rng.choice([1.0, 1.0, 0.5, 0.05])))
    return out


def same_tree(n1, n2):
    if abs(n1.count - n2.count) > 1e-9 or len(n1.children) != len(n2.children):
        return False
    for a, d in n1.av.items():
        d2 = n2.av.get(a, {})
        if any(abs(d2.get(u, 0.0) - m) > 1e-9 for u, m in d.items()) or \
                any(abs(d.get(u, 0.0) - m) > 1e-9 for u, m in d2.items()):
            return False
    return all(same_tree(c1, c2) for c1, c2 in zip(n1.children, n2.children))


def test_fixed_table_is_base_tree():
    for seed in range(20):
        rng = random.Random(seed)
        keys = list(range(8))
        table = NameTable()
        for k in keys:
            table.set(k, rand_bag(rng))
        data = stream(rng, 120, keys)
        ref = RefCobwebTree(ATTRS, seed=seed, table=table)
        base = IncrementalCobwebTree(ATTRS, seed=seed)
        base.fast_ec = ref.fast_ec = False
        ref.sample_restructure = base.sample_restructure = True
        for x, w in data:
            ref.ifit(x, w)
            base.ifit(ref.translate(x), w)
        assert same_tree(ref.root, base.root), seed


def check_counts(tree, stored):
    """Every synced node's av equals the current translated values of the records below it."""
    tree.sync_all()
    under = {}
    for leaf, x, w in stored:
        node = leaf
        while node is not None:
            under.setdefault(id(node), []).append((x, w))
            node = node.parent
    for node in tree.nodes():
        exp = {}
        for x, w in under.get(id(node), []):
            for a, v in tree.translate(x).items():
                d = exp.setdefault(a, {})
                for u, p in (v.items() if isinstance(v, dict) else ((v, 1.0),)):
                    d[u] = d.get(u, 0.0) + w * p
        for a in set(exp) | set(node.av):
            e, g = exp.get(a, {}), node.av.get(a, {})
            for u in set(e) | set(g):
                assert abs(e.get(u, 0.0) - g.get(u, 0.0)) < 1e-7, (node.id, a, u, e.get(u), g.get(u))
            ss = sum(m * m for m in g.values())
            assert abs(ss - node.ss.get(a, 0.0)) < 1e-6 * max(1.0, ss), (node.id, a)


def test_counts_follow_names():
    for seed in range(20):
        rng = random.Random(100 + seed)
        keys = list(range(10))
        table = NameTable()
        for k in keys:
            table.set(k, rand_bag(rng))
        tree = RefCobwebTree(ATTRS, seed=seed, table=table)
        tree.sample_restructure = True
        stored = []
        for t, (x, w) in enumerate(stream(rng, 200, keys)):
            leaf = tree.ifit(x, w)
            stored.append([leaf, x, w])
            if rng.random() < 0.3:                       # a name changes
                table.set(rng.choice(keys), rand_bag(rng))
            if rng.random() < 0.1 and len(stored) > 5:   # a record moves (unfit + ifit)
                i = rng.randrange(len(stored))
                leaf, x, w = stored[i]
                tree.unfit(leaf, x, w)
                stored[i][0] = tree.ifit(x, w)
                for s in stored:                         # a leaf emptied and removed never holds records
                    assert s[0].count > 0
            if rng.random() < 0.05:                      # a record relabelled in place
                i = rng.randrange(len(stored))
                leaf, x, w = stored[i]
                new = dict(x, L=Ref(rng.choice(keys)))
                tree.relabel(leaf, x, new, w)
                stored[i][1] = new
            if t % 50 == 49:
                tree.categorize(stream(rng, 1, keys)[0][0])
        check_counts(tree, stored)


if __name__ == "__main__":
    test_fixed_table_is_base_tree()
    print("fixed table: identical to the base tree (20 streams)")
    test_counts_follow_names()
    print("changing names: synced counts = current names summed (20 streams)")
