"""
met6 — gen_learn_test: validate the unsupervised grammar against the
supervised reference, render parse trees, and ANALYZE the discovered
nonterminals (chunk categories) and how consistent they are.

For a chosen grammar it:
  1. Trains a SUPERVISED reference (gold merges) and an UNSUPERVISED model
     (sentences only, τ-driven) on the SAME corpus + config.
  2. Head-to-head table: parse F1 / EM vs gold, generation grammaticality
     / novelty (CYK legality oracle).
  3. RENDERS parse trees (PNG+HTML via FiniteParseTree.visualize):
       - held-out test sentences: unsupervised parse vs gold parse
       - freshly GENERATED sentences: their unsupervised parse trees
  4. NONTERMINAL ANALYSIS — the discovered "nonterminals" are the content
     basic-level categories the unsupervised parser commits chunks into.
     We report, per nonterminal:
        - usage count and the POS-span(s) it covers (what it "means")
        - REPRESENTATION consistency : fraction of its uses that share its
          single dominant POS-span (does this category mean one thing?)
        - EXPANSION consistency      : fraction of its uses taking its
          single most-common production (does it expand one way?)
     and corpus-level:
        - ASSIGNMENT consistency     : for each POS-span pattern, fraction
          of its occurrences assigned to its dominant nonterminal (does the
          same surface structure always get the same category?)

Best settings (from the research loop): SMALL τ=8 (matches supervised),
MED τ=2 (generation beats supervised; parse-F1-vs-gold trails — its own
binarization). This is a standalone test (no other met6 scripts needed).

Usage:
    PYTHONHASHSEED=0 python tests/met6/gen_learn_test.py MED
    PYTHONHASHSEED=0 python tests/met6/gen_learn_test.py SMALL
"""
import os, sys, re, random
from collections import Counter, defaultdict
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))

from cobweb.cobweb_discrete import set_random_seed as cobweb_set_seed
from util.cfg import (TEST_GRAMMAR_SMALL, TEST_CORPUS_SMALL,
                      TEST_GRAMMAR_MED, TEST_CORPUS_MED, generate_with_merges)
from parse_mh import (WEBSTER, FiniteParseTree, CompositeParseNode,
                      PrimitiveParseNode, _categorize)
from unittests.learning_curves_test import _build_cyk_recognizer, _bracket_set

MATURITY_GATE = ("root_log_prob", -12.0)
CONTEXT_LENGTH = 3
CONTENT_ALPHA = 1e-4
CONTENT_BL_ALPHA = 10
SEED = 13

SETTINGS = {
    "SMALL": dict(grammar=TEST_GRAMMAR_SMALL, corpus=TEST_CORPUS_SMALL,
                  tau=8, n_train=150, epochs=6, flatten=("VP",)),
    "MED":   dict(grammar=TEST_GRAMMAR_MED, corpus=TEST_CORPUS_MED,
                  tau=2, n_train=150, epochs=6, flatten=("VP",)),
}
N_SHOW_TREES = 4
N_GEN_SHOW   = 4
N_GEN_EVAL   = 60


# ───────────────────────── train / eval helpers ───────────────────────

def make_webster(corpus):
    return WEBSTER(
        corpus, context_length=CONTEXT_LENGTH, threshold=30,
        content_alpha=CONTENT_ALPHA, context_alpha=1e-4,
        content_bl_alpha=CONTENT_BL_ALPHA, context_bl_alpha=10,
        bow=False, empty_weighting=True, chunk_context=False,
        weighting="binary", categorization_mode="dfs",
        depth_max_content=1000, depth_max_context=1000,
        branch_max_content=1000, branch_max_context=1000,
        content_top_k=7, content_pool_depth=4,
    )


def gen_corpus(grammar, n, seed, flatten):
    random.seed(seed)
    seen, out, attempts = set(), [], 0
    while len(out) < n and attempts < n * 60:
        attempts += 1
        text, merges = generate_with_merges("S", grammar, flatten_at_parent=flatten)
        text = text.strip()
        if not text or text in seen or len(re.findall(r"[\w']+|[.,!?;]", text)) < 3:
            continue
        seen.add(text); out.append({"sentence": text, "merges": merges})
    return out


def gold_brackets(webster, h):
    gt = FiniteParseTree(webster.ltm, context_length=CONTEXT_LENGTH)
    gt.build_primitives(h["sentence"], threshold="converge")
    for m in h["merges"]:
        try: gt.apply_candidate(m["left"], m["right"])
        except Exception: pass
    return _bracket_set(gt)


def train_supervised(corpus, train):
    random.seed(SEED); np.random.seed(SEED); cobweb_set_seed(SEED)
    web = make_webster(corpus)
    for h in train:
        for tok in re.findall(r"[\w']+|[.,!?;]", h["sentence"]):
            web.ltm.add_to_vocab(tok)
    trees = []
    for h in train:
        t = FiniteParseTree(web.ltm, context_length=CONTEXT_LENGTH)
        t.build_primitives(h["sentence"], threshold="converge")
        for m in h["merges"]:
            try: t.apply_candidate(m["left"], m["right"])
            except Exception: pass
        web.ltm.add_parse_tree(t, shuffle=True); trees.append(t)
    return web, trees


def train_unsupervised(corpus, train, tau, epochs):
    random.seed(SEED); np.random.seed(SEED); cobweb_set_seed(SEED)
    web = make_webster(corpus)
    trees = []
    for ep in range(epochs):
        trees = []
        for h in train:
            pt = web.parse_sentence(h["sentence"], threshold=tau, climb_count_threshold=tau,
                                    new_vocab=(ep == 0), learning=True,
                                    maturity_gate=MATURITY_GATE, gate_mode="skip")
            trees.append(pt)
    return web, trees


def parse_uns(web, s, tau):
    return web.parse_sentence(s, threshold=tau, climb_count_threshold=tau,
                              new_vocab=False, learning=False,
                              maturity_gate=MATURITY_GATE, gate_mode="skip")


def eval_model(web, test, gold, tau, recog, w2p, train_set, trees):
    tp = fp = fn = em = n = 0
    for h in test:
        s = h["sentence"]
        pt = parse_uns(web, s, tau)
        pred = _bracket_set(pt); g = gold[s]
        tp += len(g & pred); fp += len(pred - g); fn += len(g - pred)
        if g == pred and len(g) > 0: em += 1
        n += 1
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    F = 2 * P * R / max(P + R, 1e-12)
    web.learn_leaf_transitions(trees); web.learn_chunk_records(trees)
    web.ltm.chunk_pool_weight = 5.0
    made = gram = novel = 0
    if getattr(web, "sentence_root_chunks", None):
        for _ in range(N_GEN_EVAL):
            try: text, _ = web.generate_via_chunk_replay()
            except Exception: continue
            toks = text.split()
            if not toks: continue
            made += 1
            if all(t in w2p for t in toks) and recog(toks): gram += 1
            if text.strip() not in train_set: novel += 1
    return {"F1": F, "EM": em / max(n, 1),
            "gen_gram": gram / max(made, 1), "gen_novel": novel / max(made, 1)}


def render(pt, out_base):
    try:
        pt.visualize(out_base, render_png=True); return out_base + ".png"
    except Exception as e:
        try:
            pt.visualize(out_base, render_png=False); return out_base + ".html"
        except Exception as e2:
            return "FAILED: %s" % e2


# ───────────────────── nonterminal analysis ───────────────────────────

def make_category_fn(web):
    """composite → content basic-level class hash (its NONTERMINAL id)."""
    cache = {}

    def category(comp):
        ci = comp.get_content_instance()
        if not ci: return None
        leaf, _, node_path, _ = _categorize(ci, web.ltm.content_hierarchy, mode="dfs")
        if not node_path: return None
        h = node_path[-1].concept_hash()
        if h in cache: return cache[h]
        bl = node_path[-1].get_basic(200, 100, debug=False,
                                     eval_alpha=CONTENT_BL_ALPHA, use_root=True)
        cache[h] = str(bl.concept_hash()); return cache[h]
    return category


def pos_leaves(node, tokens, w2p):
    """POS tags of the primitive leaves under `node`, left→right."""
    leaves = []
    def w(n):
        if isinstance(n, PrimitiveParseNode):
            leaves.append((n.position_idx, n)); return
        for _, c in getattr(n, "children", []): w(c)
    w(node)
    leaves.sort(key=lambda x: x[0])
    out = []
    for pos, n in leaves:
        i = int(round(pos))
        tok = tokens[i] if 0 <= i < len(tokens) else None
        out.append(w2p.get(tok, "?") if tok else "?")
    return tuple(out)


def analyze_nonterminals(web, sents, tau, w2p):
    """Discover the unsupervised grammar's nonterminals (content basic-level
    categories) and measure their consistency."""
    category = make_category_fn(web)
    nt_count = Counter()
    nt_posspan = defaultdict(Counter)     # nonterminal → POS-span distribution
    nt_prod = defaultdict(Counter)        # nonterminal → production distribution
    posspan_nt = defaultdict(Counter)     # POS-span → nonterminal distribution

    def child_sig(category, ch, tokens):
        if isinstance(ch, PrimitiveParseNode):
            i = int(round(ch.position_idx))
            return ("t", w2p.get(tokens[i], "?") if 0 <= i < len(tokens) else "?")
        return ("N", category(ch))

    for h in sents:
        s = h["sentence"] if isinstance(h, dict) else h
        tokens = re.findall(r"[\w']+|[.,!?;]", s)
        pt = parse_uns(web, s, tau)

        def rec(n):
            if isinstance(n, CompositeParseNode) and not getattr(n, "is_global_root", False):
                cat = category(n)
                span = pos_leaves(n, tokens, w2p)
                nt_count[cat] += 1
                nt_posspan[cat][span] += 1
                posspan_nt[span][cat] += 1
                kids = sorted(getattr(n, "children", []),
                              key=lambda y: y[0] if y[0] is not None else 0)
                if len(kids) == 2:
                    nt_prod[cat][(child_sig(category, kids[0][1], tokens),
                                  child_sig(category, kids[1][1], tokens))] += 1
            for _, c in getattr(n, "children", []):
                rec(c)
        for _, c in pt.global_root_node.children:
            rec(c)

    # per-nonterminal stats
    rows = []
    for cat, cnt in nt_count.most_common():
        spans = nt_posspan[cat]
        top_span, top_span_n = spans.most_common(1)[0]
        repr_consistency = top_span_n / cnt          # one meaning?
        prods = nt_prod[cat]
        top_prod_n = prods.most_common(1)[0][1] if prods else 0
        expand_consistency = top_prod_n / cnt        # one expansion?
        rows.append({
            "nt": cat[:8], "count": cnt,
            "dom_span": " ".join(top_span),
            "repr_cons": repr_consistency,
            "n_spans": len(spans),
            "expand_cons": expand_consistency,
            "n_prods": len(prods),
        })

    # corpus-level consistency (usage-weighted)
    tot = sum(nt_count.values())
    repr_w = sum(r["repr_cons"] * r["count"] for r in rows) / max(tot, 1)
    expand_w = sum(r["expand_cons"] * r["count"] for r in rows) / max(tot, 1)
    assign_num = assign_den = 0
    for span, cats in posspan_nt.items():
        s_tot = sum(cats.values())
        assign_num += cats.most_common(1)[0][1]
        assign_den += s_tot
    assign_cons = assign_num / max(assign_den, 1)

    return {
        "n_nonterminals": len(nt_count),
        "rows": rows,
        "mean_repr_consistency": repr_w,
        "mean_expand_consistency": expand_w,
        "assignment_consistency": assign_cons,
    }


def main():
    gname = (sys.argv[1] if len(sys.argv) > 1 else "MED").upper()
    cfg = SETTINGS[gname]
    grammar, corpus, tau = cfg["grammar"], cfg["corpus"], cfg["tau"]
    out_dir = os.path.join(_HERE, "gen_learn", gname)
    tree_dir = os.path.join(out_dir, "trees")
    os.makedirs(tree_dir, exist_ok=True)

    recog, w2p = _build_cyk_recognizer(grammar)
    data = gen_corpus(grammar, cfg["n_train"] + 40, seed=SEED, flatten=cfg["flatten"])
    random.seed(SEED); random.shuffle(data)
    train, test = data[:cfg["n_train"]], data[cfg["n_train"]:cfg["n_train"] + 40]
    train_set = {h["sentence"].strip() for h in train}

    scratch = make_webster(corpus)
    for h in data:
        for tok in re.findall(r"[\w']+|[.,!?;]", h["sentence"]):
            scratch.ltm.add_to_vocab(tok)
    gold = {h["sentence"]: gold_brackets(scratch, h) for h in test}

    print(f"=== gen_learn_test [{gname}] — τ={tau}, n_train={len(train)}, epochs={cfg['epochs']} ===\n")
    print("training supervised reference (gold merges)…", flush=True)
    web_s, trees_s = train_supervised(corpus, train)
    sup = eval_model(web_s, test, gold, 30, recog, w2p, train_set, trees_s)
    print("training unsupervised (sentences only)…", flush=True)
    web_u, trees_u = train_unsupervised(corpus, train, tau, cfg["epochs"])
    uns = eval_model(web_u, test, gold, tau, recog, w2p, train_set, trees_u)

    print(f"\n{'metric':>16} {'SUPERVISED':>12} {'UNSUPERVISED':>14}")
    for k, lbl in [("F1", "parse F1"), ("EM", "exact-match"),
                   ("gen_gram", "gen grammatical"), ("gen_novel", "gen novel")]:
        print(f"{lbl:>16} {100*sup[k]:>11.1f}% {100*uns[k]:>13.1f}%")

    # ── nonterminal analysis (on train fold — the grammar it formed) ──
    print(f"\n--- NONTERMINAL ANALYSIS (unsupervised, {gname}) ---", flush=True)
    A = analyze_nonterminals(web_u, train, tau, w2p)
    print(f"  {A['n_nonterminals']} nonterminals discovered")
    print(f"  representation consistency (one meaning per NT):  {100*A['mean_repr_consistency']:.1f}%")
    print(f"  expansion consistency      (one production per NT): {100*A['mean_expand_consistency']:.1f}%")
    print(f"  assignment consistency     (one NT per surface POS-span): {100*A['assignment_consistency']:.1f}%")
    print(f"\n  {'NT':>9} {'uses':>5} {'dominant POS-span':>22} {'repr%':>6} {'#spans':>6} {'expand%':>7} {'#prods':>6}")
    for r in A["rows"][:20]:
        print(f"  {r['nt']:>9} {r['count']:>5d} {r['dom_span']:>22} "
              f"{100*r['repr_cons']:>5.0f} {r['n_spans']:>6d} "
              f"{100*r['expand_cons']:>6.0f} {r['n_prods']:>6d}", flush=True)

    # ── render trees ──
    print(f"\nrendering {N_SHOW_TREES} test trees (unsup vs gold) → {tree_dir}", flush=True)
    shown = [h for h in test if len(h["sentence"].split()) >= 4][:N_SHOW_TREES] or test[:N_SHOW_TREES]
    tree_notes = []
    for i, h in enumerate(shown):
        s = h["sentence"]
        render(parse_uns(web_u, s, tau), os.path.join(tree_dir, f"test{i}_unsup"))
        gt = FiniteParseTree(web_u.ltm, context_length=CONTEXT_LENGTH)
        gt.build_primitives(s, threshold="converge")
        for m in h["merges"]:
            try: gt.apply_candidate(m["left"], m["right"])
            except Exception: pass
        render(gt, os.path.join(tree_dir, f"test{i}_gold"))
        tree_notes.append((s, f"test{i}_unsup.png", f"test{i}_gold.png"))

    print(f"rendering {N_GEN_SHOW} generated trees → {tree_dir}", flush=True)
    web_u.learn_leaf_transitions(trees_u); web_u.learn_chunk_records(trees_u)
    web_u.ltm.chunk_pool_weight = 5.0
    gen_samples, seen, tries = [], set(), 0
    while len(gen_samples) < N_GEN_SHOW and tries < 300:
        tries += 1
        try: text, _ = web_u.generate_via_chunk_replay()
        except Exception: continue
        text = text.strip(); toks = text.split()
        if not toks or text in seen or len(toks) < 3: continue
        seen.add(text)
        gram = all(t in w2p for t in toks) and recog(toks)
        novel = text not in train_set
        idx = len(gen_samples)
        render(parse_uns(web_u, text, tau), os.path.join(tree_dir, f"gen{idx}"))
        gen_samples.append((text, gram, novel, f"gen{idx}.png"))

    # ── report ──
    md = os.path.join(out_dir, "results.md")
    with open(md, "w") as f:
        f.write(f"# gen_learn_test — {gname} (τ={tau}, n_train={len(train)}, epochs={cfg['epochs']})\n\n")
        f.write("## Head-to-head vs supervised reference\n\n")
        f.write("| metric | SUPERVISED | UNSUPERVISED |\n|---|--:|--:|\n")
        for k, lbl in [("F1", "parse F1"), ("EM", "exact-match"),
                       ("gen_gram", "generation grammatical"), ("gen_novel", "generation novel")]:
            f.write(f"| {lbl} | {100*sup[k]:.1f}% | {100*uns[k]:.1f}% |\n")
        f.write(f"\n## Nonterminal analysis (unsupervised)\n\n")
        f.write(f"- **{A['n_nonterminals']} nonterminals** discovered (content basic-level categories)\n")
        f.write(f"- representation consistency (one meaning per NT): **{100*A['mean_repr_consistency']:.1f}%**\n")
        f.write(f"- expansion consistency (one production per NT): **{100*A['mean_expand_consistency']:.1f}%**\n")
        f.write(f"- assignment consistency (one NT per surface POS-span): **{100*A['assignment_consistency']:.1f}%**\n\n")
        f.write("| nonterminal | uses | dominant POS-span | repr% | #spans | expand% | #prods |\n")
        f.write("|---|--:|---|--:|--:|--:|--:|\n")
        for r in A["rows"]:
            f.write(f"| {r['nt']} | {r['count']} | `{r['dom_span']}` | {100*r['repr_cons']:.0f}% "
                    f"| {r['n_spans']} | {100*r['expand_cons']:.0f}% | {r['n_prods']} |\n")
        f.write("\n## Sample generated sentences (unsupervised)\n\n")
        for text, gram, novel, png in gen_samples:
            tags = [t for t, ok in [("grammatical", gram), ("novel", novel)] if ok]
            f.write(f"- `{text}`  ({', '.join(tags) or 'in-train / off-grammar'})  → `trees/{png}`\n")
        f.write("\n## Test parse trees (unsupervised vs gold)\n\n")
        for s, u_png, g_png in tree_notes:
            f.write(f"- `{s}`  →  unsup `trees/{u_png}`  vs  gold `trees/{g_png}`\n")
    print(f"\nreport → {md}\ntrees  → {tree_dir}")


if __name__ == "__main__":
    main()
