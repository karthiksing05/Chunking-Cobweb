"""
Test: compare standard redistribute (misplaced-PU) vs BFS activation-based
redistribute_bfs on the same synthetic POS hierarchy.

Three trees are built from identical training data:
  1. baseline     – no redistribution
  2. redist_pu    – standard redistribute(n)
  3. redist_bfs   – BFS activation-based redistribute_bfs(...)

For each tree the same battery of query instances is evaluated and printed
so that structural differences can be inspected.
"""

import sys, os, copy, json, random, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from cobweb.cobweb_discrete import CobwebDiscreteTree
from viz import HTMLCobwebDrawer


# ── simulated concept IDs (POS hierarchy) ────────────────────────────────────
FUNC_WORD    = 100
CONTENT_WORD = 200
ARTICLE      = 110
NOUN         = 210
VERB         = 220
DEF_ART      = 111
INDEF_ART    = 112

THE   = 1011
A     = 1012
CAT   = 2011
DOG   = 2012
BIRD  = 2013
FISH  = 2021
MOUSE = 2022
HORSE = 2031   # UNSEEN
COW   = 2032   # UNSEEN
RUNS  = 2211
SEES  = 2212

VALUE_NAMES = {
    FUNC_WORD: "FUNC_WORD", CONTENT_WORD: "CONTENT_WORD",
    ARTICLE: "ARTICLE", NOUN: "NOUN", VERB: "VERB",
    DEF_ART: "DEF_ART", INDEF_ART: "INDEF_ART",
    THE: "the", A: "a",
    CAT: "cat", DOG: "dog", BIRD: "bird",
    FISH: "fish", MOUSE: "mouse",
    HORSE: "horse (unseen)", COW: "cow (unseen)",
    RUNS: "runs", SEES: "sees",
}


def inst(l0, l1, l2, r0, r1, r2):
    return {
        0: {l0: 1.0}, 1: {l1: 1.0}, 2: {l2: 1.0},
        3: {r0: 1.0}, 4: {r1: 1.0}, 5: {r2: 1.0},
    }


def article_def():   return (FUNC_WORD, ARTICLE, DEF_ART)
def article_indef(): return (FUNC_WORD, ARTICLE, INDEF_ART)
def noun(leaf):      return (CONTENT_WORD, NOUN, leaf)
def verb(leaf):      return (CONTENT_WORD, VERB, leaf)


DET_NOUN_BIGRAMS = [
    inst(*article_def(),   *noun(CAT)),
    inst(*article_def(),   *noun(DOG)),
    inst(*article_def(),   *noun(BIRD)),
    inst(*article_def(),   *noun(FISH)),
    inst(*article_def(),   *noun(MOUSE)),
    inst(*article_indef(), *noun(CAT)),
    inst(*article_indef(), *noun(DOG)),
    inst(*article_indef(), *noun(BIRD)),
    inst(*article_indef(), *noun(FISH)),
    inst(*article_indef(), *noun(MOUSE)),
]

NOUN_VERB_BIGRAMS = [
    inst(*noun(CAT), *verb(RUNS)),
    inst(*noun(DOG), *verb(SEES)),
]

TRAINING = DET_NOUN_BIGRAMS + NOUN_VERB_BIGRAMS


# ── helpers ───────────────────────────────────────────────────────────────────
def count_concepts(node):
    return 1 + sum(count_concepts(c) for c in node.children)


def path_to_leaf(tree, instance):
    leaf = tree.categorize(instance)
    path = []
    n = leaf
    while n is not None:
        path.append(n)
        n = n.parent
    path.reverse()
    return path


def print_scores(label, tree, instance):
    path = path_to_leaf(tree, instance)

    tree_lp      = tree.log_prob(instance, 100, False)
    tree_class_lp = tree.log_prob_class_given_instance(instance, 100, False)
    root_lp      = path[0].log_prob_instance(instance)
    leaf_lp      = path[-1].log_prob_instance(instance)

    basic_node    = path[-1].get_basic(1000, 100, False, eval_alpha=1)
    basic_lp      = basic_node.log_prob_instance(instance)
    basic_depth   = basic_node.depth()

    best_node     = path[-1].get_best(instance)
    best_lp       = best_node.log_prob_instance(instance)
    best_depth    = best_node.depth()

    print(f"\n{'='*60}")
    print(f"  Query: {label}")
    print(f"{'='*60}")
    print(f"         tree log-prob : {tree_lp:.6f}")
    print(f"   tree class log-prob : {tree_class_lp:.6f}")
    print(f"         root log-prob : {root_lp:.6f}  (count={path[0].count})")
    print(f"         leaf log-prob : {leaf_lp:.6f}  (count={path[-1].count})")
    print(f"        basic log-prob : {basic_lp:.6f}  (depth={basic_depth}, count={basic_node.count})")
    print(f"         best log-prob : {best_lp:.6f}  (depth={best_depth}, count={best_node.count})")

    print(f"\n  Full path ({len(path)} nodes):")
    for i, n in enumerate(path):
        lp = n.log_prob_instance(instance)
        markers = []
        if n is basic_node: markers.append("basic-level")
        if n is best_node:  markers.append("best")
        marker = (" ← " + ", ".join(markers)) if markers else ""
        print(f"    [{i}] depth={i}  count={n.count:5.0f}  lp={lp:.6f}{marker}")


QUERIES = [
    ('Det+Noun  "the cat"  (FREQUENT, SEEN)',
     inst(*article_def(), *noun(CAT))),
    ('Noun+Verb "cat runs" (RARE, SEEN)',
     inst(*noun(CAT), *verb(RUNS))),
    ('Det+UnseenNoun "the horse"  (UNSEEN leaf, NP pattern)',
     inst(*article_def(), *noun(HORSE))),
    ('UnseenNoun+Verb "horse runs" (UNSEEN N, SEEN V)',
     inst(*noun(HORSE), *verb(RUNS))),
    ('Det+UnseenNoun "a horse" (UNSEEN leaf, INDEF, NP pattern)',
     inst(*article_indef(), *noun(HORSE))),
]


def build_baseline(training, shuffle=False):
    tree = CobwebDiscreteTree(alpha=1e-3, weight_attr=False)
    data = list(training)
    if shuffle:
        random.shuffle(data)
    for item in data:
        tree.ifit(item)
    return tree


def save_tree_html(tree, output_path, tag):
    _val_fn = lambda vid: VALUE_NAMES.get(vid, str(vid))
    drawer = HTMLCobwebDrawer(
        attributes=["Left-D0", "Left-D1", "Left-D2",
                     "Right-D0", "Right-D1", "Right-D2"],
        id_to_value=[],
        value_to_id={},
        attr_value_fn={i: _val_fn for i in range(6)},
    )
    try:
        html_file, png_file = drawer.draw_tree(tree.root, output_path)
        print(f"\n  [{tag}] tree visualization saved to: {html_file}")
    except Exception as exc:
        html_file = output_path + ".html"
        d3_json = json.dumps(drawer._node_to_dict(tree.root))
        html_str = drawer._build_html(d3_json)
        os.makedirs(os.path.dirname(html_file), exist_ok=True)
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_str)
        print(f"\n  [{tag}] tree visualization (HTML only) saved to: {html_file}")
        print(f"    (PNG skipped: {exc})")


# ── main ──────────────────────────────────────────────────────────────────────
def test_redistribute_compare(shuffle=False):
    # ── 1. Baseline (no redistribution) ─────────────────────────────────
    baseline = build_baseline(TRAINING, shuffle)

    # We need two independent copies for the two redistribution methods.
    # Cheapest way: just rebuild from the same data with the same order.
    redist_pu  = build_baseline(TRAINING, shuffle=False)
    redist_bfs = build_baseline(TRAINING, shuffle=False)

    # ── 2. Standard redistribution (misplaced-PU) ──────────────────────
    print("\n" + "─" * 70)
    print("  Running standard redistribute (misplaced-PU), n=200 ...")
    print("─" * 70)
    redist_pu.redistribute(200)

    # ── 3. BFS activation-based redistribution ─────────────────────────
    print("\n" + "─" * 70)
    print("  Running redistribute_bfs (activation-based) ...")
    print(f"    n_probes=200, max_nodes=50, sim_threshold=0.5, max_merges=10")
    print("─" * 70)
    n_merges = redist_bfs.redistribute_bfs(
        n_probes=200, max_nodes=50, sim_threshold=0.5, max_merges=10)
    print(f"    → {n_merges} merge(s) performed")

    # ── 4. Report tree sizes ───────────────────────────────────────────
    trees = [
        ("BASELINE",   baseline),
        ("REDIST-PU",  redist_pu),
        ("REDIST-BFS", redist_bfs),
    ]

    for tag, tree in trees:
        nc = count_concepts(tree.root)
        print(f"\n  [{tag}] {nc} concepts, root.count={tree.root.count}")

    # ── 5. Score queries across all three trees ────────────────────────
    for query_label, query_inst in QUERIES:
        for tag, tree in trees:
            print_scores(f"[{tag}] {query_label}", tree, query_inst)

    # ── 6. Tree visualizations ─────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(__file__), "output", "test_redistribute_compare")
    os.makedirs(output_dir, exist_ok=True)
    save_tree_html(baseline,   os.path.join(output_dir, "redist_compare_baseline"),   "BASELINE")
    save_tree_html(redist_pu,  os.path.join(output_dir, "redist_compare_pu"),         "REDIST-PU")
    save_tree_html(redist_bfs, os.path.join(output_dir, "redist_compare_bfs"),        "REDIST-BFS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--shuffle", action="store_true",
                        help="Shuffle training data before fitting")
    args = parser.parse_args()
    test_redistribute_compare(shuffle=args.shuffle)
