"""Is the compiled Cobweb (cu_cobweb.CUCobwebTree, cobweb_cu) the same tree as
the Python IncrementalCobwebTree?

Random streams of instances (single values and bags, random weights) with
sample_restructure on, interleaving ifit / unfit / relabel / add_passive (and
its removal) / categorize; after EVERY operation the two trees are compared
node by node: pre-order ids, parents, children, counts, av (values, counts and
insertion order), ss, sstot, pav, restructured sets; floats compared with ==.

Modes: "ref" = Python fast_ec=False vs C++ fast=False (reference arithmetic),
"fast" = Python fast_ec=True vs C++ fast=True: C++ fast mode also updates sums of
squares on unfit instead of recomputing them, so floats are compared with a
relative tolerance of 1e-9 and a structural divergence (after a near-tie) is
reported, not failed.
Usage: cu_equiv.py [streams] [ops] [modes]
"""
import random
import sys

sys.path.insert(0, "src")
from trellis2.cu_cobweb import CUCobwebTree  # noqa: E402
from trellis2.incremental_cobweb import IncrementalCobwebTree  # noqa: E402

streams = int(sys.argv[1]) if len(sys.argv) > 1 else 20
n_ops = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
modes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["ref", "fast"]


def snapshot(t):
    out = []
    for n in t.nodes():
        p = n.parent
        out.append((n.id, None if p is None else p.id, [c.id for c in n.children], n.count, n.sstot,
                    [(a, list(d.items())) for a, d in n.av.items()],
                    list(n.ss.items()),
                    [(a, list(d.items())) for a, d in n.pav.items()]))
    return out


class Diverged(Exception):
    pass


def close(a, b, tol=1e-9):
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    return a == b


def first_diff(a, b):
    for x, y in zip(a, b):
        if x != y:
            return x, y
    return (len(a), len(b))


def run(seed, mode):
    rng = random.Random(1000 + seed)
    n_attrs = rng.choice([2, 3, 4, 6])
    attrs = [f"a{k}" for k in range(n_attrs)]
    vocab = [f"v{k}" for k in range(rng.choice([3, 5, 12, 40]))]
    py = IncrementalCobwebTree(attrs, seed=seed)
    cu = CUCobwebTree(attrs, seed=seed, fast=(mode == "fast"))
    py.fast_ec = (mode == "fast")
    py.sample_restructure = cu.sample_restructure = True
    bag_p = rng.random()

    def same(ok, msg):
        if not ok:
            raise (AssertionError if mode == "ref" else Diverged)(msg)

    def value():
        if rng.random() < bag_p:
            k = rng.randint(1, min(len(vocab), 20 if rng.random() < 0.1 else 4))
            us = rng.sample(vocab, k)
            ws = [rng.choice([1.0, rng.random() + 0.01]) for _ in us]
            t = sum(ws)
            return {u: x / t for u, x in zip(us, ws)}
        return rng.choice(vocab)

    def inst():
        return {a: value() for a in attrs}

    def weight():
        r = rng.random()
        return 1.0 if r < 0.4 else (rng.choice([0.5, 0.25, 0.1]) if r < 0.6 else rng.random() * 0.999 + 0.001)

    recs = []          # [x, w, py leaf, cu leaf, passive list]
    ops = 0
    stats = {"merge/split events": 0, "leaf removed": 0, "unfit raised": 0}
    kinds = {}
    n_restr = 0
    sp = []
    for step in range(n_ops):
        r = rng.random()
        kind = ("ifit" if r < 0.5 or not recs else "unfit" if r < 0.62 else "relabel" if r < 0.77
                else "passive" if r < 0.88 else "categorize")
        if kind == "ifit":
            x, w = inst(), weight()
            lp, lc = py.ifit(x, w), cu.ifit(x, w)
            same(lp.id == lc.id, (seed, mode, step, "ifit leaf", lp.id, lc.id))
            recs.append([x, w, lp, lc, []])
        elif kind == "unfit":
            i = rng.randrange(len(recs))
            x, w, lp, lc, pas = recs.pop(i)
            for a, v, m in pas:
                py.add_passive(lp, a, v, -m)
                cu.add_passive(lc, a, v, -m)
            ep = ec = None
            try:
                py.unfit(lp, x, w)
            except Exception as e:   # noqa: BLE001
                ep = type(e).__name__
            try:
                cu.unfit(lc, x, w)
            except Exception as e:   # noqa: BLE001
                ec = type(e).__name__
            same((ep is None) == (ec is None), (seed, mode, step, "unfit raised", ep, ec))
            stats["unfit raised"] += ep is not None
        elif kind == "relabel":
            rec = recs[rng.randrange(len(recs))]
            x, w, lp, lc, _ = rec
            new = dict(x)
            for a in rng.sample(attrs, rng.randint(1, n_attrs)):
                new[a] = value()
            if rng.random() < 0.1:
                new = dict(x)          # nothing differs
            py.relabel(lp, x, new, w)
            cu.relabel(lc, x, new, w)
            rec[0] = new
        elif kind == "passive":
            rec = recs[rng.randrange(len(recs))]
            if rec[4] and rng.random() < 0.3:
                a, v, m = rec[4].pop(rng.randrange(len(rec[4])))
                py.add_passive(rec[2], a, v, -m)
                cu.add_passive(rec[3], a, v, -m)
            else:
                a, v, m = rng.choice(["P", "Q"]), rng.choice(vocab + [("t", 1), 7]), weight()
                py.add_passive(rec[2], a, v, m)
                cu.add_passive(rec[3], a, v, m)
                rec[4].append((a, v, m))
        else:
            x = inst()
            if rng.random() < 0.3:
                x = {a: v for a, v in x.items() if rng.random() < 0.6}
            if rng.random() < 0.1 and x:
                x[next(iter(x))] = "unseen-value"
            w = weight()
            pp, pc = [n.id for n in py.categorize(x, w)], [n.id for n in cu.categorize(x, w)]
            same(pp == pc, (seed, mode, step, "categorize", pp, pc))
            for n in py.categorize(x, w)[-1:]:
                assert [m.id for m in py.ancestors(n)] == [m.id for m in cu.ancestors(cu._node(n.id))]
        ops += 1
        kinds[kind] = kinds.get(kind, 0) + 1
        before = len(sp) if step else 0
        sp, sc = snapshot(py), snapshot(cu)
        if kind == "unfit" and len(sp) < before:
            stats["leaf removed"] += 1
        if (sp != sc) if mode == "ref" else not close(sp, sc):
            same(False, f"seed {seed} mode {mode} step {step} op {kind}: first difference {str(first_diff(sp, sc))[:1500]}")
        rp = sorted(n.id for n in py.restructured)
        rc = sorted(n.id for n in cu.restructured)
        same(rp == rc, (seed, mode, step, "restructured", rp, rc))
        stats["merge/split events"] += len(rp) > n_restr
        n_restr = len(rp)
        if rng.random() < 0.05:
            py.restructured.clear()
            cu.restructured.clear()
            n_restr = 0
    py.check_invariants()
    cu.check_invariants()
    return ops, sum(1 for _ in py.nodes()), kinds, stats


total = {}
for mode in modes:
    for s in range(streams):
        try:
            ops, nodes, kinds, stats = run(s, mode)
        except Diverged as e:
            print(f"mode {mode} stream {s}: DIVERGED {e}", flush=True)
            total.setdefault(mode + " diverged streams", 0)
            total[mode + " diverged streams"] += 1
            continue
        total[mode] = total.get(mode, 0) + ops
        print(f"mode {mode} stream {s}: {ops} ops {'identical' if mode == 'ref' else 'equal within 1e-9'} (final tree {nodes} nodes) {kinds} {stats}", flush=True)
print(f"DONE (ref: exact ==; fast: rel 1e-9): operations compared per mode {total}", flush=True)
