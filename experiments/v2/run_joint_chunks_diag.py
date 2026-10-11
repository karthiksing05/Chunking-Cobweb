"""Step 1 of the joint-chunk work: does the code reward whole components when
they are applied jointly, but not step by step?

On the night's final analyses of 2,000 characters from sequences alone (a
saved ``state_baseline_s<seed>.pkl`` of ``run_joint_chunks.py night``), the
most frequent IDS components with 2-4 atomic parts (gold subtrees: an oracle,
used for this diagnostic only) are made constituents with their gold internal
brackets at every occurrence (``joint_chunks.restructure``: brackets crossing
them dropped, what is left joined right-branching; each new node a fresh
category per pair of child categories). The exact change of the plain code
(``mdl_search.code_bits``) and of the full code (the night's consolidation of
the changed analyses, ``Learner.fit``; no re-analysis) against the same
consolidation of the unchanged analyses:

(a) one binary join at a time: each of the component's gold brackets alone,
    at all its occurrences; and the bottom-up path, one bracket after another;
(b) the whole nested component at once;
(c) all of the top-K components together, K = 10, 50, 200;
(d) every operator chunk joined with its second part ([[op A] B], with
    [op A]) in every character at once; (e) the whole gold structure, for
    reference.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

from trellis2.characters import OPERATORS, gold_tree, load_characters, parse_prefix
from trellis2.mdl_search import code_bits, to_tree
from trellis2.unsupervised import UnsupervisedLearner

import joint_chunks as JC
import run_characters_memory as RCM

_W = {}


def _init(seed, seqs):
    learner = UnsupervisedLearner(seed=seed)
    for s in seqs:
        learner.observe(s)
    _W["learner"] = learner


def _fit(analyses):
    t0 = time.time()
    model = _W["learner"].fit([to_tree(a) for a in analyses])
    g = model.grammar
    return {"total": g.info["total bits"], "model": g.info.get("model bits"), "data": g.info.get("data bits"),
            "symbols": g.K, "chunk types": g.info["chunk types"], "seconds": time.time() - t0}


def components(train, valid):
    """Occurrences of database components with 2-4 atomic parts as proper
    parts of the training characters: yield -> [(character, span, gold
    brackets inside it)]."""
    occ = defaultdict(list)
    for ci, c in enumerate(train):
        st = parse_prefix(c.tokens)
        gold = gold_tree(c.tokens).brackets()
        nodes, _ = RCM.subtrees(st)
        for span, node in nodes[1:]:
            if isinstance(node, str):
                continue
            y = tuple(c.tokens[span[0]:span[1]])
            parts = sum(t not in OPERATORS for t in y)
            if y in valid and 2 <= parts <= 4:
                inside = {b for b in gold if span[0] <= b[0] and b[1] <= span[1]}
                occ[y].append((ci, span, inside))
    return occ


def operator_joins(train):
    """Per character, [op A] and [[op A] B] for every operator node."""
    out = []
    for c in train:
        st = parse_prefix(c.tokens)
        nodes, _ = RCM.subtrees(st)
        req = set()
        for span, node in nodes:
            if isinstance(node, str):
                continue
            # The parts start right after the operator.
            pos, ends = span[0] + 1, []
            for part in node[1:]:
                pos += _size(part)
                ends.append(pos)
            req.add((span[0], ends[0]))
            req.add((span[0], ends[1]))
        out.append(req)
    return out


def _size(node) -> int:
    return 1 if isinstance(node, str) else 1 + sum(_size(p) for p in node[1:])


def apply(analyses, req_by_char, prefix):
    lab = JC.Labeler(prefix)
    out = []
    for i, tops in enumerate(analyses):
        r = req_by_char.get(i)
        out.append(JC.restructure(tops, r, lab) if r else tops)
    return out


def present(analyses, occs):
    """Share of occurrences whose gold brackets are all constituents already."""
    hit = 0
    for ci, span, inside in occs:
        t = to_tree(analyses[ci])
        b = t.brackets() | set(t.roots)
        hit += inside <= b
    return hit / max(len(occs), 1)


def run(args):
    out_dir = args.out
    with open(os.path.join(out_dir, f"state_baseline_s{args.seed}.pkl"), "rb") as f:
        state = pickle.load(f)
    chars, _ = load_characters(args.path, seed=args.seed)
    train = chars[:args.train]
    analyses = state["analyses"]
    n_tokens = state["n_tokens"]
    alpha = 0.001
    plain_valid, _ = RCM.database_components(args.path)
    occ = components(train, plain_valid)
    ranked = sorted(occ, key=lambda y: (-len(occ[y]), " ".join(y)))
    print(f"{len(ranked)} components with 2-4 parts; top counts "
          f"{[len(occ[y]) for y in ranked[:10]]}", flush=True)
    jobs = []           # (name, analyses)
    meta = []

    def add(name, req_by_char, info):
        new = apply(analyses, req_by_char, f"jc:{len(jobs)}")
        changed = sum(a != b for a, b in zip(new, analyses))
        info = dict(info, name=name, changed_characters=changed,
                    plain_delta=code_bits(new, n_tokens, alpha) - base_plain)
        meta.append(info)
        jobs.append(new if changed else None)

    base_plain = code_bits(analyses, n_tokens, alpha)
    jobs.append(analyses)
    meta.append({"name": "unchanged", "plain_delta": 0.0, "changed_characters": 0})
    for rank, y in enumerate(ranked[:args.top]):
        occs = occ[y]
        rel = sorted({(b[0] - span[0], b[1] - span[0]) for _, span, inside in occs for b in inside},
                     key=lambda r: (r[1] - r[0], r[0]))
        info = {"component": " ".join(y), "entries": plain_valid[y][:3], "rank": rank, "occurrences": len(occs),
                "already constituent (all its brackets)": present(analyses, occs), "brackets": rel}

        def req_for(rels):
            req = defaultdict(set)
            for ci, span, inside in occs:
                for a, b in rels:
                    if (span[0] + a, span[0] + b) in inside:
                        req[ci].add((span[0] + a, span[0] + b))
            return req
        add(f"(b) whole: {' '.join(y)}", req_for(rel), dict(info, kind="b"))
        for r in rel:
            add(f"(a) single {r}: {' '.join(y)}", req_for([r]), dict(info, kind="a single", bracket=r))
        for k in range(1, len(rel)):
            add(f"(a) path {k}/{len(rel)}: {' '.join(y)}", req_for(rel[:k]),
                dict(info, kind="a path", step=k, of=len(rel)))
    for K in (10, 50, 200):
        req = defaultdict(set)
        for y in ranked[:K]:
            for ci, span, inside in occ[y]:
                req[ci] |= inside
        add(f"(c) top {K} jointly", req, {"kind": "c", "K": K, "available": len(ranked[:K])})
    ops = operator_joins(train)
    add("(d) every operator chunk with its second part", dict(enumerate(ops)), {"kind": "d"})
    gold = {i: c.tree.brackets() for i, c in enumerate(train)}
    add("(e) the whole gold structure", gold, {"kind": "e"})
    todo = [i for i, j in enumerate(jobs) if j is not None]
    print(f"{len(meta)} variants, {len(todo)} to refit", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.seed, [c.tokens for c in train])) as pool:
        fits = dict(zip(todo, pool.map(_fit, [jobs[i] for i in todo])))
    base = fits[0]["total"]
    for i, m in enumerate(meta):
        if i in fits:
            m["full"] = fits[i]
            m["full_delta"] = fits[i]["total"] - base
        else:
            m["full_delta"] = 0.0
    res = {"seed": args.seed, "night total bits": state["total bits"], "refit of the night's analyses": fits[0],
           "plain bits of the night's analyses": base_plain, "variants": meta, "seconds": time.time() - t0}
    with open(os.path.join(out_dir, f"diag_s{args.seed}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    for m in meta:
        print(f"{m['name'][:70]:70s} plain {m['plain_delta']:+9.1f}  full {m['full_delta']:+9.1f}  "
              f"({m['changed_characters']} chars)", flush=True)
