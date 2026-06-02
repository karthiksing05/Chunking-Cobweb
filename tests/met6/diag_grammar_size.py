"""
met6 diagnostic — is the formed grammar actually SIMPLE?

Reframe: the baseline already shows determinism=1.0 (the FROZEN grammar
parses the same input the same way) — that IS consistency. The raw
"#types" = distinct context-concept labels was 150-190, but that counts
every fine-grained label, not generalized categories. The real grammar
size is the number of distinct content-tree BASIC-LEVEL chunk CLASSES
(the generalized categories) and the distinct PRODUCTIONS
(parentClass → leftClass rightClass) over them.

This probe trains W epochs at a given gate, freezes, and reports:
  - determinism      (re-parse identical → consistency)
  - coverage, recursion (self-embed, depth)
  - n_labels         (raw context-concept labels — the old "#types")
  - n_basic_classes  (distinct content basic-level classes = categories)
  - n_productions    (distinct parent→(L,R) class triples = grammar rules)

Usage:
    PYTHONHASHSEED=0 python tests/met6/diag_grammar_size.py
"""
import os, sys, random
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "src")))

from cobweb.cobweb_discrete import set_random_seed as cobweb_set_seed
import unsupervised_grammar_formation as M
from parse_mh import CompositeParseNode, PrimitiveParseNode

GATES   = [2, 10]
N_TRAIN = 50
W_EPOCHS = 5
SEED    = 13


def basic_class_of(web, comp):
    """Content basic-level class hash for a composite (generalized category)."""
    ci = comp.get_content_instance()
    if not ci:
        return None
    from parse_mh import _categorize
    leaf, _, node_path, _ = _categorize(ci, web.ltm.content_hierarchy, mode="dfs")
    if not node_path:
        return None
    bl = node_path[-1].get_basic(200, 100, debug=False, eval_alpha=10, use_root=True)
    return str(bl.concept_hash())


def child_class(web, ch):
    if isinstance(ch, PrimitiveParseNode):
        return ("w", ch.word_id)
    return ("c", basic_class_of(web, ch))


def measure(web, sents, gate):
    det = 0
    labels = set(); basic = set(); prods = set()
    depths = []; emb = 0; cov = 0
    for s in sents:
        pt1 = M.parse_no_learn(web, s, gate)
        pt2 = M.parse_no_learn(web, s, gate)
        if M._bracket_set(pt1) == M._bracket_set(pt2):
            det += 1
        kids = list(pt1.global_root_node.children)
        if M.n_frontier_roots(pt1) == 1 and kids and isinstance(kids[0][1], CompositeParseNode):
            cov += 1
        d, nc, se = M.tree_depth_and_selfembed(pt1)
        depths.append(d)
        if se: emb += 1

        def rec(n):
            if isinstance(n, CompositeParseNode) and not getattr(n, "is_global_root", False):
                lab = n.label
                if isinstance(lab, dict) and lab:
                    labels.add(next(iter(lab.keys())))
                bc = basic_class_of(web, n)
                basic.add(bc)
                kids2 = sorted(getattr(n, "children", []), key=lambda y: y[0] if y[0] is not None else 0)
                if len(kids2) == 2:
                    prods.add((bc, child_class(web, kids2[0][1]), child_class(web, kids2[1][1])))
            for _, c in getattr(n, "children", []):
                rec(c)
        for _, c in pt1.global_root_node.children:
            rec(c)
    n = len(sents)
    return {
        "gate": gate,
        "determinism": round(det / n, 2),
        "coverage": round(cov / n, 2),
        "mean_depth": round(float(np.mean(depths)), 1),
        "self_embed": round(emb / n, 2),
        "n_labels": len(labels),
        "n_basic_classes": len(basic),
        "n_productions": len(prods),
    }


def main():
    sents = M.gen_unique(N_TRAIN, seed=SEED, min_words=3)
    print(f"=== grammar-size diagnostic — {len(sents)} sentences, {W_EPOCHS} warmup epochs, then FROZEN ===\n")
    for gate in GATES:
        random.seed(SEED); np.random.seed(SEED); cobweb_set_seed(SEED)
        web = M.make_webster()
        for epoch in range(W_EPOCHS):
            for s in sents:
                web.parse_sentence(s, threshold=30, climb_count_threshold=gate,
                                   new_vocab=(epoch == 0), learning=True,
                                   maturity_gate=M.MATURITY_GATE, gate_mode=M.GATE_MODE)
        r = measure(web, sents, gate)
        print(f"gate={gate}: det={r['determinism']} cover={r['coverage']} "
              f"meanD={r['mean_depth']} selfEmb={r['self_embed']} | "
              f"n_labels={r['n_labels']} n_basic_classes={r['n_basic_classes']} "
              f"n_productions={r['n_productions']}", flush=True)


if __name__ == "__main__":
    main()
