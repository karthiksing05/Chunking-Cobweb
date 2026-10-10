"""Track 3 of the TRELLIS v2 handoff: Chinese characters, are the chunks strong?

Three measurements on the characters of ``run_characters.py`` (seed 13, the
first 2,000 learned and the next 500 held out), for the three ways TRELLIS v2
learns them (operators as relations, operators as tokens, sequences alone):

1. Memorization and compression. Bits per character, two-part (the grammar's
   total code of the training characters: model + data) and held out,
   against gzip and xz, token n-grams (add-alpha, alpha by 5-fold
   cross-validation on the training characters) and a small GRU
   (``run_characters_memory_neural.py``, torch). The same on shuffled
   characters, where each character keeps its operator skeleton and the
   atomic components are permuted across all the corpus' slots
   (``shuffled``), or only among slots of the same operator and position
   (``slot_shuffled``, which keeps positional statistics and breaks only
   co-occurrence).
2. Chunk validity. Each chunk type (symbol, left symbol, relation, right
   symbol: the grammar's ``chunk types``) is represented by its modal
   training yield; it is valid when that yield is the full decomposition of
   some entry of the IDS database (a character or a listed component).
   Recall: of the K most frequent multi-part components (proper parts of the
   training characters that are database entries), how many are the modal
   yield of a chunk type, of a category, and how often they are analysed as
   one constituent.
3. Completion and cued recall on the held-out characters. (a) Given the
   operator and the first part, complete the character: open generation
   (prompting, 100 draws, ranked by frequency) and closed-set ranking among
   the real characters of the corpus with the same cue. (b) One atomic
   component masked: rank every training component by the probability of
   the filled character. Top-1 / top-5 exact match against the truth.

Usage (one job per call; see logs/submit_characters_memory.sh):
    python experiments/v2/run_characters_memory.py prepare
    python experiments/v2/run_characters_memory.py trellis --mode relational --cond real
    python experiments/v2/run_characters_memory.py baselines
    python experiments/v2/run_characters_memory.py summarise
"""
from __future__ import annotations

import argparse
import gzip
import json
import lzma
import math
import os
import random
import sys
import time
import zlib
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import Trellis2  # noqa: E402
from trellis2.characters import (ARITY, OPERATORS, TWO_PART, CharacterMemory, canonical,  # noqa: E402
                                 default_ids_path, expand, from_relational, load_characters,
                                 parse_prefix, structure_tokens, to_relational, token_slots)
from trellis2.grammar import inside_of_analysis  # noqa: E402
from trellis2.unsupervised import UnsupervisedLearner  # noqa: E402

LN2 = math.log(2)
OUT = os.path.join(HERE, "results", "characters_memory")
CONDS = ["real", "shuffled", "slot_shuffled"]
MODES = ["relational", "supervised", "unsupervised"]
EOS = "</s>"
UNK = "<unk>"


# --------------------------------------------------------------------------- #
# Data: splits, shuffled corpora, queries
# --------------------------------------------------------------------------- #
def prepare(args) -> dict:
    chars, ids = load_characters(args.path, seed=args.seed)
    base = chars[:args.train + args.test]
    rng = random.Random(args.seed)
    seqs = {"real": [list(c.tokens) for c in base]}
    # Global shuffle: every atomic component of the corpus permuted across all slots.
    pos = [(ci, pi) for ci, c in enumerate(base) for pi, t in enumerate(c.tokens) if t not in OPERATORS]
    vals = [base[ci].tokens[pi] for ci, pi in pos]
    rng.shuffle(vals)
    s = [list(c.tokens) for c in base]
    for (ci, pi), v in zip(pos, vals):
        s[ci][pi] = v
    seqs["shuffled"] = s
    # Slot shuffle: permuted only among slots of the same (operator, position).
    groups = defaultdict(list)
    for ci, c in enumerate(base):
        for pi, sl in enumerate(token_slots(c.tokens)):
            if c.tokens[pi] not in OPERATORS:
                groups[sl].append((ci, pi))
    s = [list(c.tokens) for c in base]
    for key in sorted(groups, key=str):
        members = groups[key]
        vals = [base[ci].tokens[pi] for ci, pi in members]
        rng.shuffle(vals)
        for (ci, pi), v in zip(members, vals):
            s[ci][pi] = v
    seqs["slot_shuffled"] = s
    for k, v in seqs.items():
        assert all(parse_prefix(t) is not None for t in v), k
    data = {"seed": args.seed, "train": args.train, "test": args.test,
            "conds": {k: {"train": v[:args.train], "test": v[args.train:]} for k, v in seqs.items()},
            "test_chars": [c.char for c in base[args.train:]]}
    # Closed-set candidates for the cue task: every character of the corpus
    # (unique token sequences), marked whether it was learned.
    train_set = {tuple(c.tokens) for c in base[:args.train]}
    corpus, seen = [], set()
    for c in chars:
        t = tuple(c.tokens)
        if t not in seen:
            seen.add(t)
            corpus.append({"tokens": list(t), "char": c.char, "trained": t in train_set})
    data["corpus"] = corpus
    # Cue queries: operator + first part of every held-out real character.
    cue = []
    for i, c in enumerate(base[args.train:]):
        st = parse_prefix(c.tokens)
        if isinstance(st, tuple):
            cue.append({"test_index": i, "op": st[0], "prompt": [st[0]] + structure_tokens(st[1]),
                        "truth": list(c.tokens)})
    data["cue"] = cue
    # Masked queries: one atomic component of every held-out real character.
    vocab = sorted({t for c in base[:args.train] for t in c.tokens if t not in OPERATORS})
    mrng = random.Random(args.seed + 1)
    mask = []
    for i, c in enumerate(base[args.train:]):
        leaves = [p for p, t in enumerate(c.tokens) if t not in OPERATORS]
        if len(c.tokens) < 2 or not leaves:
            continue
        p = mrng.choice(leaves)
        mask.append({"test_index": i, "position": p, "truth": c.tokens[p], "tokens": list(c.tokens)})
    data["mask"] = mask
    data["mask_vocab"] = vocab
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"prepared: {len(base)} characters, {len(corpus)} corpus sequences, {len(cue)} cue and "
          f"{len(mask)} mask queries, {len(vocab)} components", flush=True)
    return data


def load_data(out: str) -> dict:
    with open(os.path.join(out, "data.json"), encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# The IDS database's components
# --------------------------------------------------------------------------- #
def database_components(path: str) -> Tuple[Dict[tuple, List[str]], Dict[tuple, List[str]]]:
    """Every database entry with a multi-part decomposition: its full prefix
    expansion -> the entries (characters or components) that have it; and the
    same in canonical (relational) form."""
    from trellis2.characters import read_ids
    ids = read_ids(path)
    plain, canon = defaultdict(list), defaultdict(list)
    for c in ids:
        if ids[c] == c:
            continue
        t = expand(c, ids)
        if t is None or len(t) < 2:
            continue
        st = parse_prefix(t)
        if st is None:
            continue
        plain[tuple(t)].append(c)
        canon[tuple(structure_tokens(canonical(st)))].append(c)
    return dict(plain), dict(canon)


def subtrees(structure, start: int = 0, out=None):
    """(span, structure) of every node of a structure, in prefix order."""
    out = [] if out is None else out
    if isinstance(structure, str):
        out.append(((start, start + 1), structure))
        return out, start + 1
    me = len(out)
    out.append(None)
    pos = start + 1
    for p in structure[1:]:
        _, pos = subtrees(p, pos, out)
    out[me] = ((start, pos), structure)
    return out, pos


def relational_subtrees(node, out=None):
    out = [] if out is None else out
    out.append(node)
    if not isinstance(node, str):
        relational_subtrees(node[1], out)
        relational_subtrees(node[2], out)
    return out


def complete_parts(tokens: Sequence[str]):
    """How many complete structures ``tokens`` is, read in a row (None if it
    ends inside one)."""
    pos, n = 0, 0
    while pos < len(tokens):
        def node():
            nonlocal pos
            if pos >= len(tokens):
                raise ValueError
            t = tokens[pos]
            pos += 1
            if t in OPERATORS:
                for _ in range(ARITY[t]):
                    node()
        try:
            node()
        except ValueError:
            return None
        n += 1
    return n


def classify(y: tuple, valid: Dict[tuple, List[str]]) -> str:
    """A multi-token yield: a database component; an operator with some of
    its leading parts (an open component, which the gold trees of operators
    as tokens group); a complete structure the database does not list; or
    something that crosses component boundaries."""
    if y in valid:
        return "component"
    if parse_prefix(y) is not None:
        return "complete, not listed"
    if y[0] in OPERATORS:
        k = complete_parts(y[1:])
        if k is not None and 1 <= k < ARITY[y[0]]:
            return "operator + leading parts"
    return "crosses components"


CLASSES = ["component", "operator + leading parts", "complete, not listed", "crosses components"]


def chunk_report(memory, g, mode: str, train_seqs, valid, top_k=(50, 100, 200), recall: bool = True) -> dict:
    """Validity of the learned chunk types, and recall of frequent components."""
    n = len(memory.kind)
    relational = mode == "relational"
    # Yield of every element (canonical tokens for relational trees).
    ys: List[tuple] = []
    if relational:
        struct = []
        for e in range(n):
            if memory.left[e] < 0:
                struct.append(memory.token[e])
            else:
                struct.append((memory.relation[e], struct[memory.left[e]], struct[memory.right[e]]))
        ys = [tuple(structure_tokens(from_relational(st))) for st in struct]
    else:
        for e in range(n):
            i, j = memory.span[e]
            ys.append(tuple(memory.experiences[memory.experience_of[e]][i:j]))
    s = g.elem_symbol
    comp = [e for e in range(n) if memory.left[e] >= 0]
    # Yields that occur as a proper part of some training character (not the whole of it).
    wholes = {tuple(t) if not relational else tuple(structure_tokens(canonical(parse_prefix(t)))) for t in train_seqs}
    proper = set()
    for e in comp:
        sid = memory.experience_of[e]
        exp = memory.experiences[sid]
        full = (tuple(structure_tokens(from_relational(exp))) if relational else tuple(exp))
        if ys[e] != full:
            proper.add(ys[e])
    types: Dict[tuple, Counter] = defaultdict(Counter)
    for e in comp:
        key = (int(s[e]), int(s[memory.left[e]]), str(memory.relation[e]), int(s[memory.right[e]]))
        types[key][ys[e]] += memory.weight[e]
    assert len(types) == g.info["chunk types"], (len(types), g.info["chunk types"])
    rows = []
    for key, cnt in types.items():
        y, c = cnt.most_common(1)[0]
        tot = sum(cnt.values())
        rows.append({"type": key, "modal": " ".join(y), "purity": c / tot, "count": tot,
                     "distinct": len(cnt), "class": classify(y, valid), "entries": valid.get(y, [])[:5],
                     "proper part": y in proper, "whole training character": y in wholes})
    rows.sort(key=lambda r: -r["count"])
    share = lambda f, rs: (sum(1 for r in rs if f(r)) / len(rs)) if rs else float("nan")
    pure = [r for r in rows if r["purity"] >= 0.5]
    rep = {"chunk types": len(rows),
           "type modal class": {k: share(lambda r, k=k: r["class"] == k, rows) for k in CLASSES},
           "mean purity": float(np.mean([r["purity"] for r in rows])) if rows else float("nan"),
           "types with purity >= 0.5": len(pure),
           "type modal class (purity >= 0.5)": {k: share(lambda r, k=k: r["class"] == k, pure) for k in CLASSES},
           "share of types: purity >= 0.5 and a database component used as a proper part":
               share(lambda r: r["purity"] >= 0.5 and r["class"] == "component" and r["proper part"], rows),
           "share of types: modal yield a database component used as a proper part":
               share(lambda r: r["class"] == "component" and r["proper part"], rows),
           "share of types: modal yield only ever a whole character":
               share(lambda r: not r["proper part"], rows),
           "count-weighted modal class": {k: (sum(r["count"] for r in rows if r["class"] == k)
                                              / max(sum(r["count"] for r in rows), 1)) for k in CLASSES}}
    inst = Counter(classify(ys[e], valid) for e in comp)
    rep["composite instances"] = len(comp)
    rep["instance class"] = {k: inst[k] / max(len(comp), 1) for k in CLASSES}
    distinct = {ys[e] for e in comp}
    dc = Counter(classify(y, valid) for y in distinct)
    rep["distinct composite yields"] = len(distinct)
    rep["distinct yield class"] = {k: dc[k] / max(len(distinct), 1) for k in CLASSES}
    rep["top types"] = [{**r, "type": list(r["type"])} for r in rows[:40]]
    rep["valid modal examples"] = [{"modal": r["modal"], "entries": r["entries"], "purity": r["purity"],
                                    "count": r["count"]} for r in rows if r["class"] == "component"][:40]
    if not recall:
        return rep
    # Symbols' modal yields (multi-token).
    sym = defaultdict(Counter)
    for e in comp:
        sym[int(s[e])][ys[e]] += memory.weight[e]
    sym_modal = {cnt.most_common(1)[0][0] for cnt in sym.values()}
    type_modal = {tuple(r["modal"].split()) for r in rows}
    type_modal_pure = {tuple(r["modal"].split()) for r in rows if r["purity"] >= 0.5}
    sym_modal_pure = {cnt.most_common(1)[0][0] for cnt in sym.values()
                      if cnt.most_common(1)[0][1] >= 0.5 * sum(cnt.values())}
    # Frequency of database components as proper parts of the training characters.
    freq, occ = Counter(), defaultdict(list)
    for sid, t in enumerate(train_seqs):
        st = parse_prefix(t)
        if relational:
            for node in relational_subtrees(to_relational(st))[1:]:
                if isinstance(node, str):
                    continue
                y = tuple(structure_tokens(from_relational(node)))
                if y in valid:
                    freq[y] += 1
        else:
            nodes, _ = subtrees(st)
            for span, node in nodes[1:]:
                if isinstance(node, str):
                    continue
                y = tuple(t[span[0]:span[1]])
                if y in valid:
                    freq[y] += 1
                    occ[y].append((sid, span))
    ranked = sorted(freq, key=lambda y: (-freq[y], " ".join(y)))
    spans_of = defaultdict(set)
    if not relational:
        for e in comp:
            spans_of[memory.experience_of[e]].add(tuple(memory.span[e]))
    rep["recall"] = {}
    for K in top_k:
        top = ranked[:K]
        r = {"K": K, "available": len(top),
             "modal yield of a chunk type": sum(y in type_modal for y in top) / max(len(top), 1),
             "modal yield of a category": sum(y in sym_modal for y in top) / max(len(top), 1),
             "modal yield of a chunk type with purity >= 0.5": sum(y in type_modal_pure for y in top) / max(len(top), 1),
             "modal yield of a category with purity >= 0.5": sum(y in sym_modal_pure for y in top) / max(len(top), 1)}
        if not relational:
            frac = [sum(sp in spans_of[sid] for sid, sp in occ[y]) / len(occ[y]) for y in top]
            r["occurrences analysed as one constituent (mean)"] = float(np.mean(frac))
            r["components analysed as one constituent in >= half their occurrences"] = float(np.mean([f >= 0.5 for f in frac]))
        else:
            r["occurrences analysed as one constituent (mean)"] = 1.0   # gold relational trees
        rep["recall"][str(K)] = r
    rep["top components"] = [{"yield": " ".join(y), "count": freq[y], "entries": valid[y][:3],
                              "type modal": y in type_modal, "category modal": y in sym_modal}
                             for y in ranked[:50]]
    return rep


# --------------------------------------------------------------------------- #
# Completion
# --------------------------------------------------------------------------- #
def topk_hits(ranked: List[tuple], truth: tuple) -> Tuple[bool, bool, int]:
    rank = ranked.index(truth) + 1 if truth in ranked else 0
    return rank == 1, 0 < rank <= 5, rank


def rank_stats(hits, sizes) -> dict:
    """Beyond top-k: on queries with two or more candidates, top-1 against
    chance and the truth's percentile rank (0 = first, 1 = last or absent;
    chance 0.5), and the mean reciprocal rank over all queries."""
    multi = [(h, n) for h, n in zip(hits, sizes) if n >= 2]
    pct = [((h[2] - 1) / (n - 1)) if h[2] > 0 else 1.0 for h, n in multi]
    return {"queries with >= 2 candidates": len(multi),
            "top1 (>= 2 candidates)": float(np.mean([h[0] for h, _ in multi])) if multi else float("nan"),
            "chance top1 (>= 2 candidates)": float(np.mean([1 / n for _, n in multi])) if multi else float("nan"),
            "mean percentile rank (>= 2 candidates; chance 0.5)": float(np.mean(pct)) if multi else float("nan"),
            "MRR": float(np.mean([1 / h[2] if h[2] > 0 else 0.0 for h in hits])),
            "ranks": [int(h[2]) for h in hits], "sizes": [int(n) for n in sizes]}


def closed_cue(score, data, form) -> dict:
    """Rank the corpus' characters with the same cue by the model's probability."""
    corpus = data["corpus"]
    cand = defaultdict(list)          # prompt -> corpus indices
    prompts = {tuple(q["prompt"]) for q in data["cue"]}
    for i, c in enumerate(corpus):
        t = tuple(c["tokens"])
        if t[0] not in OPERATORS:
            continue
        st = parse_prefix(t)
        p = tuple([st[0]] + structure_tokens(st[1]))
        if p in prompts:
            cand[p].append(i)
    out = {"all": [], "unseen": []}
    sizes = {"all": [], "unseen": []}
    for q in data["cue"]:
        truth = form(q["truth"])
        idx = cand[tuple(q["prompt"])]
        for which in ("all", "unseen"):
            ids = [i for i in idx if which == "all" or not corpus[i]["trained"]]
            forms = {}
            for i in ids:
                f = form(corpus[i]["tokens"])
                forms.setdefault(f, score(corpus[i]["tokens"]))
            ranked = sorted(forms, key=lambda f: -forms[f])
            out[which].append(topk_hits(ranked, truth))
            sizes[which].append(len(forms))
    return {w: {"top1": float(np.mean([h[0] for h in out[w]])), "top5": float(np.mean([h[1] for h in out[w]])),
                "mean candidates": float(np.mean(sizes[w])), "median candidates": float(np.median(sizes[w])),
                "chance top1": float(np.mean([1 / max(n, 1) for n in sizes[w]])),
                "chance top5": float(np.mean([min(5, n) / max(n, 1) for n in sizes[w]])),
                "queries": len(out[w]), **rank_stats(out[w], sizes[w])} for w in out}


def masked(score, data, form, n_queries: int) -> dict:
    vocab = data["mask_vocab"]
    hits, sizes = [], []
    for q in data["mask"][:n_queries]:
        truth = form(q["tokens"])
        cands = {}
        for v in vocab:
            t = list(q["tokens"])
            t[q["position"]] = v
            f = form(t)
            cands.setdefault(f, score(t))
        ranked = sorted(cands, key=lambda f: -cands[f])
        hits.append(topk_hits(ranked, truth) if truth in cands else (False, False, 0))
        sizes.append(len(cands))
    return {"top1": float(np.mean([h[0] for h in hits])), "top5": float(np.mean([h[1] for h in hits])),
            **rank_stats(hits, sizes),
            "truth in vocabulary": float(np.mean([q["truth"] in set(vocab) for q in data["mask"][:n_queries]])),
            "queries": len(hits), "candidates": len(vocab)}


def open_cue(draw, data, form, n_draws: int, n_queries: int, seed: int) -> dict:
    """``draw(query, n, rng)`` returns n completions (token lists, None if it
    failed); ranked by how often each was drawn."""
    rng = np.random.default_rng(seed)
    hits, failed, wellformed = [], 0, []
    real_forms = {form(c["tokens"]) for c in data["corpus"]}
    learned = {form(c["tokens"]) for c in data["corpus"] if c["trained"]}
    real, real_new = [], []
    for q in data["cue"][:n_queries]:
        try:
            samples = draw(q, n_draws, rng)
        except (ValueError, RuntimeError):
            samples = []
            failed += 1
        cnt = Counter(form(s) for s in samples if s is not None)
        wellformed.append(np.mean([parse_prefix(s) is not None for s in samples]) if samples else 0.0)
        ranked = [f for f, _ in cnt.most_common()]
        hits.append(topk_hits(ranked, form(q["truth"])))
        fs = [form(x) for x in samples if x is not None]
        real.append(np.mean([f in real_forms for f in fs]) if fs else 0.0)
        real_new.append(np.mean([f in real_forms and f not in learned for f in fs]) if fs else 0.0)
    return {"top1": float(np.mean([h[0] for h in hits])), "top5": float(np.mean([h[1] for h in hits])),
            "draws that are real characters": float(np.mean(real)),
            "draws that are real unlearned characters": float(np.mean(real_new)),
            "truth drawn at all": float(np.mean([h[2] > 0 for h in hits])),
            "well formed draws": float(np.mean(wellformed)), "failed prompts": failed,
            "queries": len(hits), "draws": n_draws}


def cue_bits(lp_full, lp_prefix, data, n_queries) -> dict:
    """-log2 P(character | it begins with the cue): the held-out code of the
    completion given the operator and the first part."""
    bits = [-(lp_full(q["truth"]) - lp_prefix(q)) / LN2 for q in data["cue"][:n_queries]]
    return {"mean bits": float(np.mean(bits)), "median bits": float(np.median(bits)), "queries": len(bits)}


def unambiguous(structure) -> bool:
    """Whether the relational (canonical) form names this structure alone: no
    three-part operator, and no ⿰ (⿱) whose second part is again a ⿰ (⿱)
    join, which ⿲ (⿳) would also be written as."""
    if isinstance(structure, str):
        return True
    op = structure[0]
    if op in TWO_PART:
        return False
    if op in ("⿰", "⿱") and isinstance(structure[2], tuple) and structure[2][0] in (op, {"⿰": "⿲", "⿱": "⿳"}[op]):
        return False
    return all(unambiguous(p) for p in structure[1:])


def exact_query(q) -> bool:
    """A cue whose relational form is the same event as its sequence form (an
    operator with no three-part counterpart), for a character whose relational
    form names it alone: on these the relational cue bits are like-for-like."""
    return q["op"] not in ("⿰", "⿱", "⿲", "⿳") and unambiguous(parse_prefix(q["truth"]))


def run_cue(args):
    """Per-query cue bits of every model available in ``--out`` (TRELLIS
    grammars and token n-grams), and their means on all queries and on the
    exact subset (``exact_query``). Written to cue_exact.json."""
    import pickle
    from trellis2.chart import Chart
    from trellis2.prompt import PromptChart
    data = load_data(args.out)
    queries = data["cue"][:args.cue_queries]
    mask = [exact_query(q) for q in queries]
    out = {"queries": len(queries), "exact queries": int(sum(mask)), "exact mask": mask, "models": {}}

    def record(name, lp_full, lp_prefix):
        bits = [-(lp_full(q["truth"]) - lp_prefix(q)) / LN2 for q in queries]
        ex = [b for b, m in zip(bits, mask) if m]
        out["models"][name] = {"mean bits": float(np.mean(bits)), "exact subset mean bits": float(np.mean(ex)),
                               "exact subset median bits": float(np.median(ex)), "per query": bits}
        print(name, out["models"][name]["mean bits"], out["models"][name]["exact subset mean bits"], flush=True)
    for mode in MODES:
        path = os.path.join(args.out, f"grammar_{mode}.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            g = pickle.load(f)
        if mode == "relational":
            mem = CharacterMemory()
            record(mode, lambda t: mem.log_prob(g, to_relational(parse_prefix(t))), relational_prefix(g))
        else:
            record(mode, lambda t: Chart(g, t).log_prob, lambda q: PromptChart(g, q["prompt"]).log_prefix)
    tr = data["conds"]["real"]["train"]
    for order in (2, 3):
        m = NGram(tr, order, cv_alpha(tr, order))
        record(f"{order}-gram", m.logp, lambda q, m=m: m.logp_prefix(q["prompt"]))
    with open(os.path.join(args.out, "cue_exact.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


def relational_prefix(g):
    """ln P(a character's relational tree has root relation r and first part A)."""
    parts = lambda n: None if isinstance(n, str) else (n[1], n[2])

    def lp(q):
        op = q["op"]
        rel = TWO_PART.get(op, op)
        first = to_relational(parse_prefix(q["prompt"][1:]))
        scale, vA = inside_of_analysis(g, first, parts=parts, token=lambda x: x, relation=lambda x: x[0])
        gamma = g.qk * g.Rel[:, g.relations.index(rel)] * (g.Lt @ vA)
        return float(scale + np.log(g.S @ (g.U @ gamma)) + g.log_whole)
    return lp


def relational_draw(g):
    K, M = g.K, g.M
    parts = lambda n: None if isinstance(n, str) else (n[1], n[2])

    def expand_sym(sym, depth, rng):
        c = int(rng.choice(M, p=g.U[sym]))
        if rng.random() < g.pk[c] or depth >= 12:
            return g.vocab[int(rng.choice(len(g.vocab), p=g.E[c]))]
        b, d = int(rng.choice(K, p=g.Lt[c])), int(rng.choice(K, p=g.Rt[c]))
        rel = g.relations[int(rng.choice(len(g.relations), p=g.Rel[c]))]
        return (rel, expand_sym(b, depth + 1, rng), expand_sym(d, depth + 1, rng))

    def draw(q, n, rng):
        op = q["op"]
        rel = TWO_PART.get(op, op)
        first = to_relational(parse_prefix(q["prompt"][1:]))
        _, vA = inside_of_analysis(g, first, parts=parts, token=lambda x: x, relation=lambda x: x[0])
        lw = g.qk * g.Rel[:, g.relations.index(rel)] * (g.Lt @ vA)
        W = (g.S[:, None] * g.U * lw[None, :]).ravel()
        if W.sum() <= 0:
            raise ValueError("cue impossible")
        p = W / W.sum()
        out = []
        for _ in range(n):
            c = int(rng.choice(p.size, p=p)) % M
            C = int(rng.choice(K, p=g.Rt[c]))
            out.append(structure_tokens(from_relational((rel, first, expand_sym(C, 1, rng)))))
        return out
    return draw


# --------------------------------------------------------------------------- #
# TRELLIS jobs
# --------------------------------------------------------------------------- #
def run_trellis(args):
    data = load_data(args.out)
    split = data["conds"][args.cond]
    train, test = split["train"], split["test"]
    seed = data["seed"]
    t0 = time.time()
    if args.mode == "relational":
        model = Trellis2(seed=seed, memory=CharacterMemory())
        for t in train:
            model.learn(to_relational(parse_prefix(t)))
        g = model.consolidate()
        memory = model.memory
        lp = lambda t: model.log_prob(to_relational(parse_prefix(t)))
        form = lambda t: tuple(structure_tokens(canonical(parse_prefix(t)))) if parse_prefix(t) is not None else tuple(t)
    elif args.mode == "supervised":
        from trellis2.characters import gold_tree
        model = Trellis2(seed=seed)
        for t in train:
            model.learn(t, gold_tree(t))
        g = model.consolidate()
        memory = model.memory
        lp = lambda t: model.chart(t).log_prob
        form = tuple
    else:
        learner = UnsupervisedLearner(seed=seed, workers=args.workers)
        for t in train:
            learner.observe(t)
        g = learner.sleep()
        model = learner.model
        memory = model.memory
        lp = lambda t: learner.chart(t).log_prob
        form = tuple
    seconds = time.time() - t0
    print(f"[{args.mode} {args.cond}] learned in {seconds:.0f}s: total {g.info['total bits']:.0f} bits", flush=True)
    cache: Dict[tuple, float] = {}

    def score(t):
        k = tuple(t)
        if k not in cache:
            cache[k] = lp(list(t))
        return cache[k]
    test_bits = -sum(score(t) for t in test) / LN2
    res = {"mode": args.mode, "cond": args.cond, "seed": seed, "train": len(train), "test": len(test),
           "seconds learn": seconds, "total bits": g.info["total bits"], "model bits": g.info["model bits"],
           "data bits": g.info["data bits"], "symbols": g.K, "rule classes": g.M,
           "chunk types": g.info["chunk types"],
           "total bits per character": g.info["total bits"] / len(train),
           "model bits per character": g.info["model bits"] / len(train),
           "data bits per character": g.info["data bits"] / len(train),
           "held-out bits per character": test_bits / len(test)}
    print(f"  held-out {res['held-out bits per character']:.2f} b/char", flush=True)
    plain, canon = database_components(args.path)
    valid = canon if args.mode == "relational" else plain
    t1 = time.time()
    res["validity"] = chunk_report(memory, g, args.mode, train, valid, recall=args.cond == "real")
    print(f"  validity {res['validity']['type modal class']} ({time.time() - t1:.0f}s)", flush=True)
    if args.validity_only:
        path = os.path.join(args.out, f"trellis_{args.mode}_{args.cond}.json")
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        for k in ("total bits", "held-out bits per character", "chunk types"):
            if abs(float(old[k]) - float(res[k])) > 1e-6:
                print(f"  WARNING: {k} differs from the saved run: {old[k]} vs {res[k]}", flush=True)
        old["validity"] = res["validity"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(old, f, indent=1, ensure_ascii=False)
        print(path, "(validity updated)", flush=True)
        return
    if args.cond == "real":
        import dataclasses
        import pickle
        with open(os.path.join(args.out, f"grammar_{args.mode}.pkl"), "wb") as f:
            pickle.dump(dataclasses.replace(g, symbol_nodes=[], rule_nodes=[]), f)
    if args.cond == "real" and not args.no_completion:
        res.update(completion_tasks(g, args.mode, data, args))
    res["seconds total"] = time.time() - t0
    path = os.path.join(args.out, f"trellis_{args.mode}_{args.cond}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(path, flush=True)


def completion_tasks(g, mode, data, args) -> dict:
    """The completion tasks from the grammar alone."""
    from trellis2.chart import Chart
    from trellis2.prompt import complete
    seed = data["seed"]
    res = {}
    if mode == "relational":
        mem = CharacterMemory()
        lp = lambda t: mem.log_prob(g, to_relational(parse_prefix(t)))
        form = lambda t: tuple(structure_tokens(canonical(parse_prefix(t)))) if parse_prefix(t) is not None else tuple(t)
        draw = relational_draw(g)
        lp_prefix = relational_prefix(g)
    else:
        from trellis2.prompt import PromptChart
        lp = lambda t: Chart(g, t).log_prob
        form = tuple
        lp_prefix = lambda q: PromptChart(g, q["prompt"]).log_prefix

        def draw(q, n, rng):
            return [c.tokens for c in complete(g, q["prompt"], n, rng, max_len=20)]
    cache: Dict[tuple, float] = {}

    def score(t):
        k = tuple(t)
        if k not in cache:
            cache[k] = lp(list(t))
        return cache[k]
    if True:
        t1 = time.time()
        res["masked"] = masked(score, data, form, args.mask_queries)
        print(f"  masked { {k: v for k, v in res['masked'].items() if k not in ('ranks', 'sizes')} } ({time.time() - t1:.0f}s)", flush=True)
        t1 = time.time()
        res["cue closed"] = closed_cue(score, data, form)
        print(f"  cue closed { {w: {k: v for k, v in d.items() if k not in ('ranks', 'sizes')} for w, d in res['cue closed'].items()} } ({time.time() - t1:.0f}s)", flush=True)
        t1 = time.time()
        res["cue bits"] = cue_bits(score, lp_prefix, data, args.cue_queries)
        print(f"  cue bits {res['cue bits']} ({time.time() - t1:.0f}s)", flush=True)
        t1 = time.time()
        res["cue open"] = open_cue(draw, data, form, args.draws, args.cue_queries, seed)
        print(f"  cue open {res['cue open']} ({time.time() - t1:.0f}s)", flush=True)
    return res


def run_complete(args):
    """Completion tasks from a saved grammar (``grammar_<mode>.pkl``); the
    results are merged into ``trellis_<mode>_real.json``."""
    import pickle
    data = load_data(args.out)
    with open(os.path.join(args.out, f"grammar_{args.mode}.pkl"), "rb") as f:
        g = pickle.load(f)
    path = os.path.join(args.out, f"trellis_{args.mode}_real.json")
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    res.update(completion_tasks(g, args.mode, data, args))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(path, flush=True)


# --------------------------------------------------------------------------- #
# Baselines: compressors and n-grams
# --------------------------------------------------------------------------- #
class NGram:
    def __init__(self, train, order: int, alpha: float):
        self.order, self.alpha = order, alpha
        self.vocab = sorted({t for s in train for t in s} | {EOS, UNK})
        self.V = len(self.vocab)
        self.known = set(self.vocab)
        self.counts, self.ctx = Counter(), Counter()
        self.nexts = defaultdict(Counter)
        for s in train:
            self.update(s)

    def hist(self, seq, i):
        return tuple(seq[max(0, i - self.order + 1):i]) if self.order > 1 else ()

    def update(self, s):
        seq = ["<s>"] * (self.order - 1) + list(s) + [EOS]
        for i in range(self.order - 1, len(seq)):
            h = self.hist(seq, i)
            self.counts[(h, seq[i])] += 1
            self.ctx[h] += 1
            self.nexts[h][seq[i]] += 1

    def logp(self, s) -> float:
        seq = ["<s>"] * (self.order - 1) + [t if t in self.known else UNK for t in s] + [EOS]
        total = 0.0
        for i in range(self.order - 1, len(seq)):
            h = self.hist(seq, i)
            total += math.log((self.counts[(h, seq[i])] + self.alpha) / (self.ctx[h] + self.alpha * self.V))
        return total

    def logp_prefix(self, prompt) -> float:
        seq = ["<s>"] * (self.order - 1) + [t if t in self.known else UNK for t in prompt]
        total = 0.0
        for i in range(self.order - 1, len(seq)):
            h = self.hist(seq, i)
            total += math.log((self.counts[(h, seq[i])] + self.alpha) / (self.ctx[h] + self.alpha * self.V))
        return total

    def sample(self, prompt, rng, max_len=20):
        seq = ["<s>"] * (self.order - 1) + list(prompt)
        out = list(prompt)
        vocab = [v for v in self.vocab if v != UNK]
        while len(out) < max_len:
            h = self.hist(seq, len(seq))
            w = np.array([self.counts[(h, v)] + self.alpha for v in vocab])
            v = vocab[int(rng.choice(len(vocab), p=w / w.sum()))]
            if v == EOS:
                return out
            out.append(v)
            seq.append(v)
        return out


def prequential_bits(train, order, alpha, vocab_size) -> float:
    """Adaptive add-alpha code of the training corpus, in order: a one-part
    code (the Dirichlet-multinomial marginal) with no separate model bits."""
    counts, ctx = Counter(), Counter()
    total = 0.0
    for s in train:
        seq = ["<s>"] * (order - 1) + list(s) + [EOS]
        for i in range(order - 1, len(seq)):
            h = tuple(seq[max(0, i - order + 1):i]) if order > 1 else ()
            total -= math.log2((counts[(h, seq[i])] + alpha) / (ctx[h] + alpha * vocab_size))
            counts[(h, seq[i])] += 1
            ctx[h] += 1
    return total


ALPHAS = [0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]


def cv_alpha(train, order, folds=5) -> float:
    best, best_bits = None, float("inf")
    for a in ALPHAS:
        bits = 0.0
        for k in range(folds):
            tr = [s for i, s in enumerate(train) if i % folds != k]
            te = [s for i, s in enumerate(train) if i % folds == k]
            m = NGram(tr, order, a)
            bits -= sum(m.logp(s) for s in te)
        if bits < best_bits:
            best, best_bits = a, bits
    return best


def compressors(train, test) -> dict:
    tr = [("".join(s)).encode("utf-8") for s in train]
    te = [("".join(s)).encode("utf-8") for s in test]
    corpus = lambda xs: b"\n".join(xs) + b"\n"

    def raw_deflate(b):
        c = zlib.compressobj(9, zlib.DEFLATED, -15, 9)
        return c.compress(b) + c.flush()

    def raw_lzma(b):
        return lzma.compress(b, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}])
    methods = {"gzip -9": lambda b: gzip.compress(b, 9, mtime=0),
               "xz -9e": lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME),
               "raw deflate": raw_deflate, "raw lzma2": raw_lzma}
    out = {"utf-8 bits per character": 8 * float(np.mean([len(b) + 1 for b in tr]))}
    for name, f in methods.items():
        whole = len(f(corpus(tr)))
        both = len(f(corpus(tr + te)))
        out[name] = {"corpus bits per character": 8 * whole / len(tr),
                     "per-character bits (each compressed alone)": 8 * float(np.mean([len(f(b)) for b in tr])),
                     "held-out bits per character (C(train+test) - C(train))": 8 * (both - whole) / len(te)}
    return out


def run_baselines(args):
    data = load_data(args.out)
    res = {}
    for cond in CONDS:
        tr, te = data["conds"][cond]["train"], data["conds"][cond]["test"]
        r = {"compressors": compressors(tr, te), "ngrams": {}}
        for order in (1, 2, 3):
            a = cv_alpha(tr, order)
            m = NGram(tr, order, a)
            r["ngrams"][str(order)] = {
                "alpha (5-fold CV)": a,
                "held-out bits per character": -sum(m.logp(s) for s in te) / LN2 / len(te),
                "held-out bits per character (alpha 0.5)": -sum(NGram(tr, order, 0.5).logp(s) for s in te) / LN2 / len(te),
                "prequential total bits per character": prequential_bits(tr, order, a, m.V) / len(tr),
                "training bits per character (fitted, no model cost)": -sum(m.logp(s) for s in tr) / LN2 / len(tr),
                "parameters (observed n-gram types)": len(m.counts)}
        res[cond] = r
        print(cond, json.dumps(r["ngrams"], indent=0), flush=True)
    # Completion with n-grams (real characters).
    tr = data["conds"]["real"]["train"]
    res["completion"] = {}
    for order in (2, 3):
        m = NGram(tr, order, res["real"]["ngrams"][str(order)]["alpha (5-fold CV)"])
        cache = {}

        def score(t, m=m, cache=cache):
            k = tuple(t)
            if k not in cache:
                cache[k] = m.logp(t)
            return cache[k]
        draw = lambda q, n, rng, m=m: [m.sample(q["prompt"], rng) for _ in range(n)]
        res["completion"][f"{order}-gram"] = {
            "masked": masked(score, data, tuple, args.mask_queries),
            "cue closed": closed_cue(score, data, tuple),
            "cue bits": cue_bits(score, lambda q, m=m: m.logp_prefix(q["prompt"]), data, args.cue_queries),
            "cue open": open_cue(draw, data, tuple, args.draws, args.cue_queries, data["seed"])}
        print(order, res["completion"][f"{order}-gram"], flush=True)
    with open(os.path.join(args.out, "baselines.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
LABEL = {"relational": "TRELLIS, structures given (operators as relations)",
         "supervised": "TRELLIS, structures given (operators as tokens)",
         "unsupervised": "TRELLIS, sequences alone"}


def run_summarise(args):
    def load(name):
        p = os.path.join(args.out, name)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    data = load_data(args.out)
    base = load("baselines.json")
    neural = load("neural.json")
    tr = {(m, c): load(f"trellis_{m}_{c}.json") for m in MODES for c in CONDS}
    f2 = lambda x: "–" if x is None else f"{x:.1f}"
    L = [f"# Chinese characters: are the chunks strong? (Track 3)\n",
         f"Seed {data['seed']}, {data['train']} learned / {data['test']} held-out characters "
         f"(`run_characters.py` defaults). Bits per character. Shuffled: atomic components permuted across all "
         f"slots of the corpus (operator skeletons kept); slot-shuffled: permuted only among slots of the same "
         f"operator and position.\n",
         "## 1. Memorization and compression\n",
         "| Method | " + " | ".join(f"{c}: total | {c}: held-out" for c in CONDS) + " | real − shuffled (held-out) | real − slot-shuffled (held-out) |",
         "|---|" + "---|---|" * len(CONDS) + "---|---|"]

    def row(name, vals):
        # vals: cond -> (total, held)
        cells = []
        for c in CONDS:
            t, h = vals.get(c, (None, None))
            cells += [f2(t), f2(h)]
        gap = lambda c: (f2(vals["real"][1] - vals[c][1]) if vals.get("real", (None, None))[1] is not None
                         and vals.get(c, (None, None))[1] is not None else "–")
        L.append(f"| {name} | " + " | ".join(cells) + f" | {gap('shuffled')} | {gap('slot_shuffled')} |")
    if base:
        for name in ("gzip -9", "xz -9e", "raw deflate", "raw lzma2"):
            row(f"{name} (corpus; held-out = C(train+test) − C(train))",
                {c: (base[c]["compressors"][name]["corpus bits per character"],
                     base[c]["compressors"][name]["held-out bits per character (C(train+test) - C(train))"]) for c in CONDS})
        for name in ("gzip -9", "xz -9e"):
            row(f"{name}, each character alone", {c: (base[c]["compressors"][name]["per-character bits (each compressed alone)"], None) for c in CONDS})
        for o in ("1", "2", "3"):
            row(f"token {o}-gram (add-α, α by CV; total = prequential code)",
                {c: (base[c]["ngrams"][o]["prequential total bits per character"],
                     base[c]["ngrams"][o]["held-out bits per character"]) for c in CONDS})
    if neural:
        for h, r in sorted(neural["models"].items(), key=lambda kv: int(kv[0])):
            row(f"GRU, hidden {h} (total = train NLL + 8-bit parameters)",
                {c: (r[c]["total bits per character (8-bit)"], r[c]["held-out bits per character"])
                 for c in CONDS if c in r})
    for m in MODES:
        row(LABEL[m] + " (total = model + data)",
            {c: (tr[(m, c)]["total bits per character"], tr[(m, c)]["held-out bits per character"])
             for c in CONDS if tr[(m, c)]})
    if base:
        uni = {c: base[c]["ngrams"]["1"]["held-out bits per character"] for c in CONDS}
        L.append("\nHeld-out gain over the token unigram of the same corpus (bits per character; the unigram codes all "
                 "three corpora alike, so this separates what a method takes from structure from how badly it "
                 "overfits a corpus without any):\n")
        L.append("| Method | real | shuffled | slot-shuffled | real − shuffled | real − slot-shuffled |")
        L.append("|---|---|---|---|---|---|")
        held = {}
        for name in ("xz -9e", "gzip -9"):
            held[name] = {c: base[c]["compressors"][name]["held-out bits per character (C(train+test) - C(train))"] for c in CONDS}
        for o in ("2", "3"):
            held[f"token {o}-gram"] = {c: base[c]["ngrams"][o]["held-out bits per character"] for c in CONDS}
        if neural:
            for h, r in neural["models"].items():
                held[f"GRU, hidden {h}"] = {c: r[c]["held-out bits per character"] for c in CONDS}
        for m in MODES:
            if all(tr[(m, c)] for c in CONDS):
                held[LABEL[m]] = {c: tr[(m, c)]["held-out bits per character"] for c in CONDS}
        for name, h in held.items():
            gain = {c: uni[c] - h[c] for c in CONDS}
            L.append(f"| {name} | {gain['real']:.1f} | {gain['shuffled']:.1f} | {gain['slot_shuffled']:.1f} | "
                     f"{gain['real'] - gain['shuffled']:.1f} | {gain['real'] - gain['slot_shuffled']:.1f} |")
    L.append("\nTRELLIS two-part codes (bits per training character: model + data = total):\n")
    L.append("| Model | " + " | ".join(CONDS) + " |")
    L.append("|---|" + "---|" * len(CONDS))
    for m in MODES:
        cells = []
        for c in CONDS:
            r = tr[(m, c)]
            cells.append("–" if not r else f"{r['model bits per character']:.1f} + {r['data bits per character']:.1f} = "
                         f"{r['total bits per character']:.1f} ({r['symbols']} symbols, {r['chunk types']} chunk types)")
        L.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    L.append("\n## 2. Chunk validity\n")
    L.append("Each chunk type (symbol, left symbol, relation, right symbol) is represented by its modal training yield.\n")
    L.append("| Model | cond | chunk types | modal yield a database component | operator + leading parts | complete, not listed | crosses components | mean purity | types with purity ≥ 0.5 (of which component) | specific reused unit: purity ≥ 0.5, a component used as a proper part | modal yield only ever a whole character | composite instances that are components |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    pc = lambda x: f"{100 * x:.1f}%"
    for m in MODES:
        for c in CONDS:
            r = tr[(m, c)]
            if not r:
                continue
            v = r["validity"]
            k = v["type modal class"]
            L.append(f"| {LABEL[m]} | {c} | {v['chunk types']} | {pc(k['component'])} | {pc(k['operator + leading parts'])} | "
                     f"{pc(k['complete, not listed'])} | {pc(k['crosses components'])} | {v['mean purity']:.2f} | "
                     f"{v['types with purity >= 0.5']} ({pc(v['type modal class (purity >= 0.5)']['component'])}) | "
                     f"{pc(v.get('share of types: purity >= 0.5 and a database component used as a proper part', float('nan')))} | "
                     f"{pc(v.get('share of types: modal yield only ever a whole character', float('nan')))} | "
                     f"{pc(v['instance class']['component'])} |")
    L.append("\nRecall of the K most frequent multi-part database components (proper parts of the training characters):\n")
    L.append("| Model | K | modal yield of a chunk type (any purity) | … with purity ≥ 0.5 | modal yield of a category (any purity) | … with purity ≥ 0.5 | occurrences analysed as one constituent |")
    L.append("|---|---|---|---|---|---|---|")
    for m in MODES:
        r = tr[(m, "real")]
        if not r:
            continue
        for K, x in r["validity"]["recall"].items():
            L.append(f"| {LABEL[m]} | {K} ({x['available']}) | {pc(x['modal yield of a chunk type'])} | "
                     f"{pc(x.get('modal yield of a chunk type with purity >= 0.5', float('nan')))} | "
                     f"{pc(x['modal yield of a category'])} | "
                     f"{pc(x.get('modal yield of a category with purity >= 0.5', float('nan')))} | "
                     f"{pc(x['occurrences analysed as one constituent (mean)'])} |")
    L.append("\n## 3. Completion and cued recall (held-out real characters)\n")
    L.append("Masked: one atomic component of each held-out character hidden, every learned component ranked by the "
             "probability of the filled character (246 candidates; chance top-1 0.4%). Cue: the operator and the first "
             "part. Cue bits = −log2 P(character | cue). Open: 100 draws given the cue, ranked by frequency. Closed: "
             "the real corpus characters sharing the cue, ranked by probability. Percentile rank: 0 = first, chance 0.5.\n")
    L.append("| Method | masked top-1 / top-5 | masked percentile rank | cue bits (mean / median) | open: top-1 / top-5 | open: draws real / real unlearned / well formed | closed (all): top-1 / top-5 (chance) | closed: percentile rank |")
    L.append("|---|---|---|---|---|---|---|---|")
    tt = lambda d: f"{pc(d['top1'])} / {pc(d['top5'])}" if d else "–"
    PR = "mean percentile rank (>= 2 candidates; chance 0.5)"
    comp_rows = []
    if base:
        for k, r in base["completion"].items():
            comp_rows.append((f"token {k}", r))
    if neural and "completion" in neural:
        comp_rows.append((f"GRU, hidden {neural['completion']['hidden']}", neural["completion"]))
    for m in MODES:
        r = tr[(m, "real")]
        if r and "masked" in r:
            comp_rows.append((LABEL[m], r))
    for name, r in comp_rows:
        mk, cb, op, cc = r["masked"], r.get("cue bits"), r.get("cue open"), r["cue closed"]["all"]
        cbs = f"{cb['mean bits']:.1f} / {cb['median bits']:.1f}" if cb else "–"
        ops = (f"{pc(op.get('draws that are real characters', float('nan')))} / "
               f"{pc(op.get('draws that are real unlearned characters', float('nan')))} / {pc(op['well formed draws'])}") if op else "–"
        L.append(f"| {name} | {tt(mk)} | {mk.get(PR, float('nan')):.3f} | {cbs} | {tt(op)} | {ops} | "
                 f"{tt(cc)} ({pc(cc.get('chance top1', float('nan')))} / {pc(cc.get('chance top5', float('nan')))}) | "
                 f"{cc.get(PR, float('nan')):.3f} |")
    seeds_path = os.path.join(args.out, "seeds.json")
    if os.path.exists(seeds_path):
        with open(seeds_path, encoding="utf-8") as f:
            per = json.load(f)
        keys = ["total · TRELLIS supervised · real", "total · TRELLIS unsupervised · real", "total · TRELLIS relational · real",
                "total · token 2-gram (adaptive code) · real", "total · token 3-gram (adaptive code) · real",
                "total · xz -9e · real", "total · GRU 8 (8-bit) · real", "total · GRU 8 (entropy-coded, best bit depth) · real",
                "held-out · GRU 128 · real", "held-out · TRELLIS supervised · real", "held-out · TRELLIS unsupervised · real",
                "held-out · token 3-gram · real", "held-out · TRELLIS relational · real",
                "gain lost when slot_shuffled · TRELLIS unsupervised", "gain lost when slot_shuffled · TRELLIS supervised",
                "gain lost when slot_shuffled · TRELLIS relational", "gain lost when slot_shuffled · GRU 128",
                "gain lost when slot_shuffled · token 3-gram",
                "validity: modal yield a component · unsupervised · real", "validity: modal yield a component · unsupervised · shuffled",
                "validity: modal yield a component · unsupervised · slot_shuffled",
                "validity: specific reused unit · unsupervised · real", "validity: specific reused unit · supervised · real",
                "validity: specific reused unit · relational · real",
                "recall@50, chunk type purity >= 0.5 · unsupervised", "recall@50, chunk type purity >= 0.5 · supervised",
                "recall@50, chunk type purity >= 0.5 · relational",
                "masked top-1 · TRELLIS relational", "masked top-1 · TRELLIS supervised", "masked top-1 · TRELLIS unsupervised",
                "masked top-1 · token 3-gram", "masked top-1 · GRU (best validation size)",
                "open draws real · TRELLIS supervised", "open draws real · TRELLIS unsupervised", "open draws real · TRELLIS relational",
                "open draws real · GRU (best validation size)",
                "cue bits, exact subset · relational", "cue bits, exact subset · supervised", "cue bits, exact subset · unsupervised",
                "cue bits, exact subset · 3-gram", "cue bits, exact subset · GRU (best validation size)"]
        seeds = list(per)
        L.append(f"\n## Across seeds ({', '.join(seeds)}): key numbers\n")
        L.append("Full table: `summary_seeds.md`. Bits per character; shares in %. Cue bits on the exact subset: cues "
                 "whose operator has no three-part counterpart (⿴…⿻) and characters whose relational form names them "
                 "alone, where the relational number is like-for-like (42–48 queries per seed).\n")
        L.append("| Quantity | " + " | ".join(f"seed {x}" for x in seeds) + " | mean | min–max |")
        L.append("|---|" + "---|" * len(seeds) + "---|---|")
        pctk = lambda k: any(t in k for t in ("validity: modal", "specific", "recall@", "masked", "open draws"))
        for k in keys:
            vals = [per[x].get(k) for x in seeds]
            if any(v is None for v in vals):
                continue
            f = (lambda v: f"{100 * v:.1f}") if pctk(k) else (lambda v: f"{v:.1f}")
            L.append(f"| {k} | " + " | ".join(f(v) for v in vals) +
                     f" | {f(float(np.mean(vals)))} | {f(min(vals))}–{f(max(vals))} |")
    L.append("""
## Caveats

- Tables 1–3 above are seed 13; the across-seeds table covers seeds 13, 17 and 23 (each seed is its own split and shuffle). Three seeds give a spread, not confidence intervals.
- With structures given, every composite is a gold IDS subtree, so a chunk type's modal yield is a database component almost by construction (relational: 91% of composite instances already are). The shuffled rows give the floor; the measurement is informative mainly for sequences alone, and through purity and recall for all modes.
- Recall "modal yield of a chunk type (any purity)" counts a component even when it is a small share of its type's instances (e.g. ⿱十一 at purity 0.01 with operators as tokens); the purity ≥ 0.5 column is the strict version.
- Closed-set cue ranking is at chance for every method, by construction: the candidates are all real characters with the same cue and the held-out truth is a random one of them. Read cue bits instead.
- Relational cue bits on all queries condition on a coarser cue (⿲ A is written ⿰ A …, so binary and ternary cues merge, and a canonical tree can stand for two sequences); only the exact-subset cue bits are like-for-like.
- GRU totals: entropy-coded weights (adaptive KT code over 2–8-bit levels, best depth per model) shorten the GRU's total a lot (hidden 8: 46 → 39 bits) but not below TRELLIS; hidden sizes below 8 and MDL-oriented training were not tried.
- The n-gram total is the prequential (adaptive add-α) code; the GRU total is training NLL at 8-bit weights plus 8 bits per parameter; TRELLIS's total is its own model + data code. gzip/xz include their headers; per-character compression is dominated by them.
""")
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# --------------------------------------------------------------------------- #
# Several seeds
# --------------------------------------------------------------------------- #
def seed_metrics(d: str) -> dict:
    """The key numbers of one seed's directory, flat: name -> value."""
    def load(name):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    out = {}
    base, cue = load("baselines.json"), load("cue_exact.json")
    neural = load("neural_v2.json") or load("neural.json")
    tr = {(m, c): load(f"trellis_{m}_{c}.json") for m in MODES for c in CONDS}
    held, total = defaultdict(dict), defaultdict(dict)
    if base:
        for c in CONDS:
            held["token 1-gram"][c] = base[c]["ngrams"]["1"]["held-out bits per character"]
            for o in ("2", "3"):
                held[f"token {o}-gram"][c] = base[c]["ngrams"][o]["held-out bits per character"]
                total[f"token {o}-gram (adaptive code)"][c] = base[c]["ngrams"][o]["prequential total bits per character"]
            held["xz -9e"][c] = base[c]["compressors"]["xz -9e"]["held-out bits per character (C(train+test) - C(train))"]
            total["xz -9e"][c] = base[c]["compressors"]["xz -9e"]["corpus bits per character"]
    if neural:
        for h, r in neural["models"].items():
            for c in CONDS:
                held[f"GRU {h}"][c] = r[c]["held-out bits per character"]
                total[f"GRU {h} (8-bit)"][c] = r[c]["total bits per character (8-bit)"]
                if "best entropy-coded total bits per character" in r[c]:
                    total[f"GRU {h} (entropy-coded, best bit depth)"][c] = r[c]["best entropy-coded total bits per character"]
    for m in MODES:
        for c in CONDS:
            r = tr[(m, c)]
            if r:
                held[f"TRELLIS {m}"][c] = r["held-out bits per character"]
                total[f"TRELLIS {m}"][c] = r["total bits per character"]
    for name, h in held.items():
        for c in CONDS:
            if c in h:
                out[f"held-out · {name} · {c}"] = h[c]
        if "token 1-gram" in held and name != "token 1-gram":
            u = held["token 1-gram"]
            for c in ("shuffled", "slot_shuffled"):
                if c in h and "real" in h:
                    out[f"gain lost when {c} · {name}"] = (u["real"] - h["real"]) - (u[c] - h[c])
    for name, t in total.items():
        for c in CONDS:
            if c in t:
                out[f"total · {name} · {c}"] = t[c]
    for m in MODES:
        for c in CONDS:
            r = tr[(m, c)]
            if not r:
                continue
            v = r["validity"]
            out[f"validity: modal yield a component · {m} · {c}"] = v["type modal class"]["component"]
            out[f"validity: mean purity · {m} · {c}"] = v["mean purity"]
            out[f"validity: specific reused unit · {m} · {c}"] = v.get(
                "share of types: purity >= 0.5 and a database component used as a proper part", float("nan"))
            out[f"validity: chunk types · {m} · {c}"] = v["chunk types"]
            if c == "real" and "recall" in v:
                for K, x in v["recall"].items():
                    out[f"recall@{K}, chunk type purity >= 0.5 · {m}"] = x.get("modal yield of a chunk type with purity >= 0.5", float("nan"))
                    out[f"recall@{K}, chunk type any purity · {m}"] = x["modal yield of a chunk type"]
    comp = {}
    if base:
        for k, r in base["completion"].items():
            comp[f"token {k}"] = r
    if neural and "completion" in neural:
        comp["GRU (best validation size)"] = neural["completion"]
    for m in MODES:
        r = tr[(m, "real")]
        if r and "masked" in r:
            comp[f"TRELLIS {m}"] = r
    for name, r in comp.items():
        out[f"masked top-1 · {name}"] = r["masked"]["top1"]
        out[f"masked top-5 · {name}"] = r["masked"]["top5"]
        op = r.get("cue open") or {}
        if "draws that are real characters" in op:
            out[f"open draws real · {name}"] = op["draws that are real characters"]
            out[f"open draws real unlearned · {name}"] = op["draws that are real unlearned characters"]
        if r.get("cue bits"):
            out[f"cue bits, all queries · {name}"] = r["cue bits"]["mean bits"]
    if cue:
        mask = cue["exact mask"]
        out["cue: exact queries"] = cue["exact queries"]
        for name, r in cue["models"].items():
            out[f"cue bits, exact subset · {name}"] = r["exact subset mean bits"]
        if neural and "per query" in neural.get("completion", {}).get("cue bits", {}):
            pq = neural["completion"]["cue bits"]["per query"]
            out["cue bits, exact subset · GRU (best validation size)"] = float(np.mean([b for b, m in zip(pq, mask) if m]))
    return out


def run_summarise_seeds(args):
    dirs = {13: args.out}
    for s in (17, 23):
        d = os.path.join(args.out, f"seed{s}")
        if os.path.exists(os.path.join(d, "data.json")):
            dirs[s] = d
    per = {s: seed_metrics(d) for s, d in dirs.items()}
    keys = []
    for s in per:
        for k in per[s]:
            if k not in keys:
                keys.append(k)
    seeds = list(per)
    L = [f"## Several seeds ({', '.join(map(str, seeds))})\n",
         "Each seed draws its own split of learned and held-out characters (the seed of `run_characters.py`) and its own "
         "shuffles. Mean over the seeds with the value, then min–max; n = number of seeds.\n",
         "| Quantity | " + " | ".join(f"seed {s}" for s in seeds) + " | mean | min–max | n |",
         "|---|" + "---|" * len(seeds) + "---|---|---|"]
    fmt = lambda k, x: ("–" if x is None or (isinstance(x, float) and math.isnan(x)) else
                        (f"{100 * x:.1f}%" if any(t in k for t in ("validity: modal", "specific", "recall@", "masked", "open draws"))
                         else (f"{x:.0f}" if "chunk types" in k or "exact queries" in k else f"{x:.2f}")))
    for k in keys:
        vals = [per[s].get(k) for s in seeds]
        ok = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not ok:
            continue
        L.append(f"| {k} | " + " | ".join(fmt(k, v) for v in vals) +
                 f" | {fmt(k, float(np.mean(ok)))} | {fmt(k, min(ok))}–{fmt(k, max(ok))} | {len(ok)} |")
    with open(os.path.join(args.out, "seeds.json"), "w", encoding="utf-8") as f:
        json.dump({str(s): per[s] for s in seeds}, f, indent=1, ensure_ascii=False)
    with open(os.path.join(args.out, "summary_seeds.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "trellis", "baselines", "summarise", "complete", "cue", "seeds"])
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--cond", choices=CONDS, default="real")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--mask-queries", type=int, default=500)
    ap.add_argument("--cue-queries", type=int, default=500)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--no-completion", action="store_true")
    ap.add_argument("--validity-only", action="store_true",
                    help="re-learn and update only the validity section of an existing result")
    ap.add_argument("--path", default=default_ids_path())
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    {"prepare": prepare, "trellis": run_trellis, "baselines": run_baselines, "summarise": run_summarise, "complete": run_complete, "cue": run_cue, "seeds": run_summarise_seeds}[args.command](args)


if __name__ == "__main__":
    main()
