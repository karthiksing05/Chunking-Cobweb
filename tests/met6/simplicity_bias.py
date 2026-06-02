"""
met6 — Simplicity bias: can we condense the symbol/category inventory
while keeping each chunk meaningful?

The unsupervised grammar (unsupervised_grammar_formation.py) is consistent,
recursive, and generative, but uses 12–20 categories formed under the
default fine-grained Cobweb clustering (content_alpha=1e-4). The question
here: is there a SIMPLICITY BIAS that motivates FEWER chunk categories
(symbols) while each remaining category stays MEANINGFUL — i.e. reused
across many contexts, not a one-off, and the grammar still covers and
generates?

The principled lever is the content-tree clustering granularity
``content_alpha`` (the Cobweb smoothing prior). Higher α makes attribute
differences look noisier → Cobweb splits less → coarser tree → fewer
basic-level categories. That is a structural simplicity bias applied at
LEARNING time. We retrain at each α and measure two things in tension:

  SYMBOL ECONOMY (want simpler)
    n_categories   : distinct content basic-level classes used (symbols)
    mean_reuse     : chunk instances / n_categories (uses per symbol —
                     higher = each symbol earns its keep)
    singleton_frac : fraction of categories used exactly once (not
                     meaningful — pure description-length overhead)

  MEANINGFULNESS / UTILITY (want preserved)
    coverage       : fraction of sentences chunked to a single root
    self_embed     : recursion retained
    gen_gram       : generations still grammatical (CYK legality oracle)

Scored by a two-part MDL code (lower = simpler grammar that still
explains the data):
    DL_grammar = n_productions * 3 * log2(n_symbols)     # write the rules
    DL_data    = n_internal_nodes * log2(n_productions)  # pick a rule per node
    MDL        = DL_grammar + DL_data

A successful simplicity bias REDUCES n_categories and MDL and RAISES
mean_reuse while holding coverage / self_embed / gen_gram roughly fixed.
The α minimizing MDL is the "simplest meaningful grammar".

Usage:
    PYTHONHASHSEED=0 python tests/met6/simplicity_bias.py
"""
import os, sys, csv, math, random
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cobweb.cobweb_discrete import set_random_seed as cobweb_set_seed
from util.cfg import generate
from parse_mh import WEBSTER, CompositeParseNode, PrimitiveParseNode, _categorize
from unittests.learning_curves_test import _build_cyk_recognizer, _bracket_set
import unsupervised_grammar_formation as M

# ───────────────────────────── config ─────────────────────────────────
ALPHAS   = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]   # content clustering granularity (simplicity bias)
TAU      = 5
BL_ALPHA = 10
N_TRAIN  = 60
N_EPOCHS = 4
N_GEN    = 60
SEED     = 13
GRAMMAR  = M.GRAMMAR
CORPUS   = M.CORPUS
OUT_DIR  = os.path.join(_HERE, "simplicity_bias")


def make_webster(content_alpha):
    return WEBSTER(
        CORPUS, context_length=M.CONTEXT_LENGTH, threshold=30,
        content_alpha=content_alpha, context_alpha=1e-4,
        content_bl_alpha=BL_ALPHA, context_bl_alpha=10,
        bow=False, empty_weighting=True, chunk_context=False,
        weighting="binary", categorization_mode="dfs",
        depth_max_content=1000, depth_max_context=1000,
        branch_max_content=1000, branch_max_context=1000,
        content_top_k=7, content_pool_depth=4,
    )


def make_category_fn(web):
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
                                     eval_alpha=BL_ALPHA, use_root=True)
        cache[h] = str(bl.concept_hash())
        return cache[h]
    return category


def child_sym(category, ch):
    if isinstance(ch, PrimitiveParseNode):
        return ("w", ch.word_id)
    return ("c", category(ch))


def run_alpha(alpha, train_sents, recog, word_to_pos, train_set):
    random.seed(SEED); np.random.seed(SEED); cobweb_set_seed(SEED)
    web = make_webster(alpha)

    trained_trees = []
    for epoch in range(N_EPOCHS):
        trained_trees = []
        for s in train_sents:
            pt = web.parse_sentence(
                s, threshold=TAU, climb_count_threshold=TAU,
                new_vocab=(epoch == 0), learning=True,
                maturity_gate=M.MATURITY_GATE, gate_mode=M.GATE_MODE)
            trained_trees.append(pt)

    # ── freeze; measure symbol economy + grammar over the corpus ──
    category = make_category_fn(web)
    cat_uses = {}                 # category → usage count (chunk instances)
    cat_prods = {}                # category → {production-signature: count}
    phrasal = set(); lexical = set()
    terminals = set()
    n_internal = 0
    cov = 0; depths = []; emb = 0
    for s in train_sents:
        pt = M.parse_no_learn(web, s, TAU)
        if M.fully_parsed(pt):
            cov += 1
        d, nc, se = M.tree_depth_and_selfembed(pt)
        depths.append(d)
        if se: emb += 1

        def rec(n):
            nonlocal n_internal
            if isinstance(n, PrimitiveParseNode):
                terminals.add(n.word_id); return
            if isinstance(n, CompositeParseNode) and not getattr(n, "is_global_root", False):
                n_internal += 1
                pc = category(n)
                cat_uses[pc] = cat_uses.get(pc, 0) + 1
                kids = sorted(getattr(n, "children", []),
                              key=lambda y: y[0] if y[0] is not None else 0)
                if len(kids) == 2:
                    ls, rs = child_sym(category, kids[0][1]), child_sym(category, kids[1][1])
                    cat_prods.setdefault(pc, {})
                    cat_prods[pc][(ls, rs)] = cat_prods[pc].get((ls, rs), 0) + 1
                    if kids[0][1].__class__ is CompositeParseNode and kids[1][1].__class__ is CompositeParseNode:
                        phrasal.add((pc, ls, rs))
                    else:
                        lexical.add((pc, ls, rs))
            for _, c in getattr(n, "children", []):
                rec(c)
        for _, c in pt.global_root_node.children:
            rec(c)

    n_categories = len(cat_uses)
    total_uses = sum(cat_uses.values())
    mean_reuse = total_uses / max(n_categories, 1)
    singleton_frac = sum(1 for v in cat_uses.values() if v == 1) / max(n_categories, 1)

    # category PURITY = how meaningful each category is: fraction of a
    # category's uses that take its single most-common expansion, averaged
    # over categories weighted by usage. 1.0 = every category deterministically
    # predicts its own expansion (maximally meaningful); low = the category
    # blurs distinct structures together (condensed too far / not meaningful).
    purity_num = purity_den = 0.0
    for c, prods in cat_prods.items():
        tot = sum(prods.values())
        if tot > 0:
            purity_num += max(prods.values())
            purity_den += tot
    category_purity = (purity_num / purity_den) if purity_den else 0.0
    n_prod = len(phrasal) + len(lexical)
    n_symbols = n_categories + len(terminals)

    # two-part MDL
    _l2 = lambda x: math.log2(x) if x > 1 else 0.0
    dl_grammar = n_prod * 3 * _l2(n_symbols + 1)
    dl_data = n_internal * _l2(n_prod + 1)
    mdl = dl_grammar + dl_data

    coverage = cov / len(train_sents)
    self_embed = emb / len(train_sents)
    mean_depth = float(np.mean(depths)) if depths else 0.0

    # generation grammaticality (utility preserved?)
    web.learn_leaf_transitions(trained_trees)
    web.learn_chunk_records(trained_trees)
    web.ltm.chunk_pool_weight = 5.0
    g_made = g_gram = 0
    if getattr(web, "sentence_root_chunks", None):
        for _ in range(N_GEN):
            try:
                text, _ = web.generate_via_chunk_replay()
            except Exception:
                continue
            toks = text.split()
            if not toks:
                continue
            g_made += 1
            if all(t in word_to_pos for t in toks) and recog(toks):
                g_gram += 1
    gen_gram = g_gram / max(g_made, 1)

    return {
        "alpha": alpha,
        "n_categories": n_categories,
        "mean_reuse": round(mean_reuse, 2),
        "singleton_frac": round(singleton_frac, 2),
        "category_purity": round(category_purity, 2),
        "n_phrasal": len(phrasal),
        "n_prod": n_prod,
        "coverage": round(coverage, 2),
        "self_embed": round(self_embed, 2),
        "mean_depth": round(mean_depth, 1),
        "gen_gram": round(gen_gram, 2),
        "dl_grammar": round(dl_grammar),
        "dl_data": round(dl_data),
        "mdl": round(mdl),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    recog, word_to_pos = _build_cyk_recognizer(GRAMMAR)
    train_sents = M.gen_unique(N_TRAIN, seed=SEED, min_words=3)
    train_set = {s.strip() for s in train_sents}
    print("=== met6 simplicity bias — content_alpha sweep (τ=%d, %d epochs→freeze) ===" % (TAU, N_EPOCHS))
    print(f"  train={len(train_sents)} sentences only; sweeping content_alpha={ALPHAS}\n")
    print(f"{'alpha':>7} {'#cats':>5} {'reuse':>6} {'singl':>6} {'purity':>6} {'#prod':>6} "
          f"{'cover':>6} {'selfEmb':>7} {'genGram':>7} {'MDL':>8}")

    rows = []
    for a in ALPHAS:
        r = run_alpha(a, train_sents, recog, word_to_pos, train_set)
        rows.append(r)
        print(f"{a:>7g} {r['n_categories']:>5d} {r['mean_reuse']:>6.2f} "
              f"{r['singleton_frac']:>6.2f} {r['category_purity']:>6.2f} {r['n_prod']:>6d} "
              f"{r['coverage']:>6.2f} {r['self_embed']:>7.2f} {r['gen_gram']:>7.2f} "
              f"{r['mdl']:>8d}", flush=True)

    best = min(rows, key=lambda r: r["mdl"])
    print(f"\nMin-MDL (simplest grammar that explains the data): "
          f"alpha={best['alpha']:g}  #cats={best['n_categories']}  "
          f"reuse={best['mean_reuse']}  MDL={best['mdl']}  "
          f"(coverage={best['coverage']} gen_gram={best['gen_gram']})")

    csv_path = os.path.join(OUT_DIR, "simplicity_bias.csv")
    cols = ["alpha", "n_categories", "mean_reuse", "singleton_frac", "category_purity",
            "n_phrasal", "n_prod", "coverage", "self_embed", "mean_depth", "gen_gram",
            "dl_grammar", "dl_data", "mdl"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"CSV → {csv_path}")

    alphas = [r["alpha"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(alphas, [r["n_categories"] for r in rows], "o-", color="#1f77b4", label="# categories")
    axes[0].plot(alphas, [r["n_prod"] for r in rows], "v:", color="#9467bd", label="# productions")
    axb = axes[0].twinx()
    axb.plot(alphas, [r["mean_reuse"] for r in rows], "s--", color="#2ca02c", label="reuse/category")
    axes[0].set_xscale("log"); axes[0].set_xlabel("content_alpha (simplicity bias)")
    axes[0].set_title("Symbol economy"); axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left"); axb.set_ylabel("reuse per category"); axb.legend(loc="upper right")

    axes[1].plot(alphas, [r["coverage"] for r in rows], "o-", color="#1f77b4", label="coverage")
    axes[1].plot(alphas, [r["category_purity"] for r in rows], "D-", color="#8c564b", label="category purity")
    axes[1].plot(alphas, [r["self_embed"] for r in rows], "s-", color="#d62728", label="self-embed")
    axes[1].plot(alphas, [r["gen_gram"] for r in rows], "^-", color="#2ca02c", label="gen grammatical")
    axes[1].plot(alphas, [r["singleton_frac"] for r in rows], "v--", color="#999999", label="singleton frac")
    axes[1].set_xscale("log"); axes[1].set_xlabel("content_alpha"); axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Meaningfulness / utility"); axes[1].grid(alpha=0.3); axes[1].legend()

    axes[2].plot(alphas, [r["dl_grammar"] for r in rows], "o-", color="#e6820f", label="DL(grammar)")
    axes[2].plot(alphas, [r["dl_data"] for r in rows], "s-", color="#1f77b4", label="DL(data|grammar)")
    axes[2].plot(alphas, [r["mdl"] for r in rows], "^-", color="#d62728", linewidth=2, label="MDL total")
    axes[2].set_xscale("log"); axes[2].set_xlabel("content_alpha")
    axes[2].set_title("MDL two-part code (lower = simpler)"); axes[2].grid(alpha=0.3); axes[2].legend()

    fig.suptitle("Simplicity bias — condensing categories while keeping chunks meaningful "
                 f"(MED, τ={TAU}, seed={SEED})", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    png = os.path.join(OUT_DIR, "simplicity_bias.png")
    plt.savefig(png, dpi=140, bbox_inches="tight"); plt.close()
    print(f"PNG → {png}")


if __name__ == "__main__":
    main()
