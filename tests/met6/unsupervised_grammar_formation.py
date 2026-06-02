"""
met6 — Unsupervised grammar formation.

GOAL (per the project owner): learn parse trees fully UNSUPERVISED — the
only signal is the raw sentence stream (NO gold parse trees, NO merges) —
and show the learner forms a grammar that is

    CONSISTENT  +  RECURSIVE  +  SIMPLE  +  GENERATIVE,

without any requirement that it match the source grammar.

How the unsupervised loop works (existing engine machinery):

    for epoch: for sentence:
        parse_sentence(sent, learning=True, climb_count_threshold=τ)
          build()        → commit a merge ONLY when the merged chunk's
                            climbing-ancestor count in the content tree > τ
                            (the parser's own decision — no gold merges)
          add_parse_tree → committed chunks → BOTH hierarchies; leftover
                            candidate pairs → CONTENT tree (counts grow)
                            [parse_mh.py "fit orphan candidate pairs"]

Candidate counts accumulate across epochs → frequent patterns cross τ →
graduate into committed chunks → higher-order / recursive chunks then
accumulate in turn. After the warmup epochs we FREEZE (learning off): the
formed grammar is the object we evaluate.

WHAT WE MEASURE (and what we learned measuring it — see README):

  CONSISTENCY = determinism. The frozen grammar parses the same input the
    same way every time (`determinism`). Coverage = fraction chunked to a
    single root. NOTE: epoch-to-epoch parse churn DURING formation is high
    (Cobweb re-categorizes as it learns) — that is a property of the
    online formation process, NOT of the formed grammar. It is reported as
    a `formation_churn` diagnostic, not as a consistency failure.

  RECURSION = `self_embed` (fraction of recursive-probe parses where a
    chunk TYPE — its context-concept label — recurs at a strictly deeper
    node = a recursive rule), plus mean/max parse depth.

  SIMPLICITY = grammar SIZE, measured at the right granularity:
    `n_categories` = distinct content-tree BASIC-LEVEL classes (the
    generalized phrase categories — NOT the over-fine context labels), and
    `n_phrasal_prods` = distinct (parentCategory → leftCategory,
    rightCategory) productions where both children are themselves chunks
    (the recursive backbone, excluding lexical productions). τ is the
    simplicity knob: higher τ → fewer categories.

  GENERATION = generate_via_chunk_replay: `gen_gram` (CYK legality oracle —
    a legality check, not a similarity metric), `gen_novel`, `roundtrip`
    (generate → re-parse → still chunks to one root = closed under its own
    generation).

The sweep over τ answers the project question: at what threshold is
unsupervised grammar formation facilitated — the τ giving the SIMPLEST
grammar that is still consistent, recursive, and generative.

Usage:
    PYTHONHASHSEED=0 python tests/met6/unsupervised_grammar_formation.py
"""
import os, sys, csv, random, re
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cobweb.cobweb_discrete import set_random_seed as cobweb_set_seed
from util.cfg import TEST_GRAMMAR_MED, TEST_CORPUS_MED, generate
from parse_mh import (WEBSTER, CompositeParseNode, PrimitiveParseNode,
                      _categorize)
from unittests.learning_curves_test import _build_cyk_recognizer, _bracket_set

# ───────────────────────────── config ─────────────────────────────────
CONTEXT_LENGTH = 3
MATURITY_GATE  = ("root_log_prob", -12.0)   # primitive gate, held fixed
GATE_MODE      = "skip"
CONTENT_BL_ALPHA = 10

THRESHOLDS = [2, 5, 10, 20, 40]   # τ — the simplicity knob
N_TRAIN    = 70
N_EPOCHS   = 4
N_RECUR    = 40
N_GEN      = 60
SEED       = 13

GRAMMAR = TEST_GRAMMAR_MED
CORPUS  = TEST_CORPUS_MED
OUT_DIR = os.path.join(_HERE, "unsup_grammar")


# ───────────────────────────── helpers ────────────────────────────────

def make_webster():
    return WEBSTER(
        CORPUS, context_length=CONTEXT_LENGTH, threshold=30,
        content_alpha=1e-4, context_alpha=1e-4,
        content_bl_alpha=CONTENT_BL_ALPHA, context_bl_alpha=10,
        bow=False, empty_weighting=True, chunk_context=False,
        weighting="binary", categorization_mode="dfs",
        depth_max_content=1000, depth_max_context=1000,
        branch_max_content=1000, branch_max_context=1000,
        content_top_k=7, content_pool_depth=4,
    )


def _ntok(s):
    return len(re.findall(r"[\w']+|[.,!?;]", s))


def gen_unique(n, seed, min_words=3, recursive_only=False, avoid=None):
    """Generate ``n`` unique sentences. ``recursive_only`` keeps only
    sentences with surface recursion (≥2 adjacent adjectives → AdjP
    recursion, a PP, or a relative pronoun)."""
    avoid = avoid or set()
    adjs = set(sum(GRAMMAR.get("Adj", []), []))
    preps = set(sum(GRAMMAR.get("P", []), []))
    relpros = set(sum(GRAMMAR.get("RelPro", []), []))
    random.seed(seed)
    out, seen = [], set()
    attempts = 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        s = generate("S", GRAMMAR).strip()
        if not s or s in seen or s in avoid or _ntok(s) < min_words:
            continue
        if recursive_only:
            toks = s.split()
            two_adj = any(toks[i] in adjs and toks[i + 1] in adjs
                          for i in range(len(toks) - 1))
            if not (two_adj or (preps & set(toks)) or (relpros & set(toks))):
                continue
        seen.add(s)
        out.append(s)
    return out


def parse_no_learn(web, sent, tau):
    return web.parse_sentence(
        sent, threshold=tau, climb_count_threshold=tau,
        new_vocab=False, learning=False,
        maturity_gate=MATURITY_GATE, gate_mode=GATE_MODE)


def n_frontier_roots(pt):
    return len(list(pt.global_root_node.children))


def fully_parsed(pt):
    kids = list(pt.global_root_node.children)
    return n_frontier_roots(pt) == 1 and kids and isinstance(kids[0][1], CompositeParseNode)


def tree_depth_and_selfembed(pt):
    """(max_depth, n_composites, self_embeds) — self_embeds True if a
    composite shares its context-concept label with a strictly deeper
    descendant composite (a recursive phrase type)."""
    max_d = [0]; n_comp = [0]; self_embed = [False]

    def label_of(n):
        lab = getattr(n, "label", None)
        return next(iter(lab.keys())) if isinstance(lab, dict) and lab else None

    def rec(n, depth, anc):
        if isinstance(n, CompositeParseNode) and not getattr(n, "is_global_root", False):
            n_comp[0] += 1; max_d[0] = max(max_d[0], depth)
            lab = label_of(n)
            if lab is not None and lab in anc:
                self_embed[0] = True
            anc = anc | ({lab} if lab is not None else set())
        for _, ch in getattr(n, "children", []):
            rec(ch, depth + 1, anc)

    for _, ch in pt.global_root_node.children:
        rec(ch, 1, frozenset())
    return max_d[0], n_comp[0], self_embed[0]


def make_category_fn(web):
    """Return a memoized fn: composite → content basic-level class hash
    (the generalized phrase category)."""
    cache = {}

    def category(comp):
        ci = comp.get_content_instance()
        if not ci:
            return None
        leaf, _, node_path, _ = _categorize(ci, web.ltm.content_hierarchy, mode="dfs")
        if not node_path:
            return None
        h = node_path[-1].concept_hash()
        if h in cache:
            return cache[h]
        bl = node_path[-1].get_basic(200, 100, debug=False,
                                     eval_alpha=CONTENT_BL_ALPHA, use_root=True)
        cache[h] = str(bl.concept_hash())
        return cache[h]
    return category


def grammar_size(web, sents, tau, category):
    """Distinct generalized categories + distinct PHRASAL productions
    (parentCat → leftCat rightCat, both children composites)."""
    cats = set(); phrasal = set()
    for s in sents:
        pt = parse_no_learn(web, s, tau)

        def rec(n):
            if isinstance(n, CompositeParseNode) and not getattr(n, "is_global_root", False):
                pc = category(n)
                cats.add(pc)
                kids = sorted(getattr(n, "children", []),
                              key=lambda y: y[0] if y[0] is not None else 0)
                if len(kids) == 2:
                    lk, rk = kids[0][1], kids[1][1]
                    if isinstance(lk, CompositeParseNode) and isinstance(rk, CompositeParseNode):
                        phrasal.add((pc, category(lk), category(rk)))
            for _, c in getattr(n, "children", []):
                rec(c)
        for _, c in pt.global_root_node.children:
            rec(c)
    return len(cats), len(phrasal)


# ───────────────────────────── one τ run ──────────────────────────────

def run_one(tau, train_sents, recur_sents, recog, word_to_pos, train_set):
    random.seed(SEED); np.random.seed(SEED); cobweb_set_seed(SEED)
    web = make_webster()

    # ── unsupervised formation: sentences only, τ drives learning ──
    last_train_brackets = {}
    trained_trees = []
    for epoch in range(N_EPOCHS):
        trained_trees = []
        for s in train_sents:
            pt = web.parse_sentence(
                s, threshold=tau, climb_count_threshold=tau,
                new_vocab=(epoch == 0), learning=True,
                maturity_gate=MATURITY_GATE, gate_mode=GATE_MODE)
            trained_trees.append(pt)
            if epoch == N_EPOCHS - 1:
                last_train_brackets[s] = _bracket_set(pt)

    # ── FREEZE — everything below is learning=False on the formed grammar ──

    # consistency: determinism (re-parse identical) + coverage; plus a
    # formation-churn diagnostic (did the frozen parse match the last
    # learning-epoch parse? low = formation still churning, NOT a grammar
    # inconsistency).
    det = cov = settled = 0
    for s in train_sents:
        pt1 = parse_no_learn(web, s, tau)
        pt2 = parse_no_learn(web, s, tau)
        b1 = _bracket_set(pt1)
        if b1 == _bracket_set(pt2): det += 1
        if fully_parsed(pt1): cov += 1
        if b1 == last_train_brackets.get(s): settled += 1
    n = len(train_sents)
    determinism = det / n
    coverage = cov / n
    formation_churn = 1.0 - settled / n

    # recursion (held-out recursive probes)
    depths, embeds, rcov = [], 0, 0
    for s in recur_sents:
        pt = parse_no_learn(web, s, tau)
        d, nc, se = tree_depth_and_selfembed(pt)
        depths.append(d)
        if se: embeds += 1
        if fully_parsed(pt): rcov += 1
    mean_depth = float(np.mean(depths)) if depths else 0.0
    max_depth = int(np.max(depths)) if depths else 0
    self_embed = embeds / max(len(recur_sents), 1)
    recur_cov = rcov / max(len(recur_sents), 1)

    # simplicity (grammar size at category granularity)
    category = make_category_fn(web)
    n_categories, n_phrasal = grammar_size(web, train_sents, tau, category)

    # generation
    web.learn_leaf_transitions(trained_trees)
    web.learn_chunk_records(trained_trees)
    web.ltm.chunk_pool_weight = 5.0
    g_made = g_lex = g_gram = g_novel = g_total = 0
    rt_total = rt_closed = 0
    if getattr(web, "sentence_root_chunks", None):
        for _ in range(N_GEN):
            try:
                text, _ = web.generate_via_chunk_replay()
            except Exception:
                continue
            g_total += 1
            toks = text.split()
            if not toks:
                continue
            g_made += 1
            l_ok = all(t in word_to_pos for t in toks)
            if l_ok: g_lex += 1
            if l_ok and recog(toks): g_gram += 1
            if text.strip() not in train_set: g_novel += 1
            if _ntok(text) >= 2:
                rt_total += 1
                if fully_parsed(parse_no_learn(web, text.strip(), tau)):
                    rt_closed += 1

    return {
        "tau": tau,
        "determinism": determinism,
        "coverage": coverage,
        "formation_churn": formation_churn,
        "mean_depth": mean_depth,
        "max_depth": max_depth,
        "self_embed": self_embed,
        "recur_cov": recur_cov,
        "n_categories": n_categories,
        "n_phrasal_prods": n_phrasal,
        "gen_gram": g_gram / max(g_made, 1),
        "gen_novel": g_novel / max(g_made, 1),
        "roundtrip": rt_closed / max(rt_total, 1),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    recog, word_to_pos = _build_cyk_recognizer(GRAMMAR)
    train_sents = gen_unique(N_TRAIN, seed=SEED, min_words=3)
    train_set = {s.strip() for s in train_sents}
    recur_sents = gen_unique(N_RECUR, seed=SEED + 1, min_words=4,
                             recursive_only=True, avoid=train_set)
    print("=== met6 unsupervised grammar formation (warmup → freeze) ===")
    print(f"  train={len(train_sents)} (sentences only)  recursive-probe={len(recur_sents)}  "
          f"epochs={N_EPOCHS}  primitive-gate={MATURITY_GATE}")
    print(f"  sweeping τ={THRESHOLDS}\n")

    rows = []
    for tau in THRESHOLDS:
        print(f"--- τ = {tau} ---", flush=True)
        r = run_one(tau, train_sents, recur_sents, recog, word_to_pos, train_set)
        rows.append(r)
        print(f"    CONSISTENT det={r['determinism']:.2f} cover={r['coverage']:.2f} "
              f"(formation_churn={r['formation_churn']:.2f}) | "
              f"RECURSIVE selfEmb={r['self_embed']:.2f} meanD={r['mean_depth']:.1f} | "
              f"SIMPLE cats={r['n_categories']} prods={r['n_phrasal_prods']} | "
              f"GEN gram={r['gen_gram']:.2f} novel={r['gen_novel']:.2f} "
              f"rt={r['roundtrip']:.2f}", flush=True)

    # CSV
    csv_path = os.path.join(OUT_DIR, "unsup_grammar.csv")
    cols = ["tau", "determinism", "coverage", "formation_churn",
            "mean_depth", "max_depth", "self_embed", "recur_cov",
            "n_categories", "n_phrasal_prods",
            "gen_gram", "gen_novel", "roundtrip"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nCSV → {csv_path}")

    # chart: consistency / recursion+simplicity / generation vs τ
    taus = [r["tau"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(taus, [r["determinism"] for r in rows], "o-", label="determinism (consistency)")
    axes[0].plot(taus, [r["coverage"] for r in rows], "s-", label="coverage")
    axes[0].plot(taus, [r["formation_churn"] for r in rows], "^--", color="#999999",
                 label="formation churn (diag)")
    axes[0].set_title("Consistency"); axes[0].set_xlabel("τ"); axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3); axes[0].legend()

    axes[1].plot(taus, [r["self_embed"] for r in rows], "o-", color="#d62728", label="self-embed (recursion)")
    axes[1].plot(taus, [r["recur_cov"] for r in rows], "s-", color="#e6820f", label="recursive-probe coverage")
    axb = axes[1].twinx()
    axb.plot(taus, [r["n_categories"] for r in rows], "^--", color="#1f77b4", label="# categories")
    axb.plot(taus, [r["n_phrasal_prods"] for r in rows], "v:", color="#9467bd", label="# phrasal prods")
    axes[1].set_title("Recursion (left) & grammar size (right)"); axes[1].set_xlabel("τ")
    axes[1].set_xscale("log"); axes[1].set_ylim(0, 1.05); axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper left"); axb.set_ylabel("grammar size"); axb.legend(loc="upper right")

    axes[2].plot(taus, [r["gen_gram"] for r in rows], "o-", color="#2ca02c", label="grammatical")
    axes[2].plot(taus, [r["gen_novel"] for r in rows], "s-", color="#1f77b4", label="novel")
    axes[2].plot(taus, [r["roundtrip"] for r in rows], "^--", color="#9467bd", label="round-trip closure")
    axes[2].set_title("Generation"); axes[2].set_xlabel("τ"); axes[2].set_xscale("log")
    axes[2].set_ylim(0, 1.05); axes[2].grid(alpha=0.3); axes[2].legend()

    fig.suptitle("Unsupervised grammar formation — consistent / recursive / simple / generative "
                 f"(MED, {N_EPOCHS} epochs→freeze, seed={SEED})", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    png = os.path.join(OUT_DIR, "unsup_grammar.png")
    plt.savefig(png, dpi=140, bbox_inches="tight"); plt.close()
    print(f"PNG → {png}")


if __name__ == "__main__":
    main()
