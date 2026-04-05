"""
tests/speedups/test_basic_level_entropy.py

Compares get_basic_entropy() (O(height) -entropy() proxy) against the
ground-truth sampled get_basic(), with ablation studies to understand
when and why they disagree.

Tests
─────
  1. test_entropy_vs_sampled_agreement  — agreement table for both hierarchies
  2. test_alpha_ablation                — agreement rate vs eval_alpha sweep
  3. test_disagreement_case_studies     — full per-node path trace for mismatches
  4. test_timing_comparison             — single-leaf microbenchmark
  5. test_plot_comparison               — saves scatter / alpha-curve / error PNGs
"""

import os
import sys
import time
import pytest

# ----- path setup so imports work under pytest (pythonpath = src) -----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cobweb.cobweb_discrete import CobwebDiscreteTree, CobwebDiscreteNode

SAVED_STATE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "unittests",
    "hollow_learn_test_mh", "final_ltm_data"
)
SAVED_STATE_DIR = os.path.normpath(SAVED_STATE_DIR)

ABLATION_ALPHAS  = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
MAX_CASE_STUDIES = 5    # disagreement leaves shown per hierarchy
QUICK_EPMI_N     = 50   # n_samples for annotating case-study paths (not ground-truth)
MAX_NODES        = 100

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _collect_leaves(tree: CobwebDiscreteTree):
    """BFS over a CobwebDiscreteTree, return all leaf nodes."""
    if tree.root is None:
        return []
    leaves = []
    stack = [tree.root]
    while stack:
        node = stack.pop()
        if not node.children:
            leaves.append(node)
        else:
            stack.extend(node.children)
    return leaves


@pytest.fixture(scope="module")
def webster():
    """Load or train a WEBSTER model."""
    from parse_mh import WEBSTER

    if os.path.isdir(SAVED_STATE_DIR):
        print(f"\n[fixture] Loading saved WEBSTER state from {SAVED_STATE_DIR}")
        w = WEBSTER.load_state(SAVED_STATE_DIR)
        return w

    print("\n[fixture] Saved state not found — training inline model (~40 sentences).")
    from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1

    random_seed = 42
    import random
    random.seed(random_seed)

    num_sentences = 40
    document = [generate("S", TEST_GRAMMAR1) for _ in range(num_sentences)]

    w = WEBSTER(
        TEST_CORPUS1,
        context_length=3,
        threshold=5,
        content_alpha=1e-3,
        context_alpha=1e-3,
        content_bl_alpha=1e-1,
        context_bl_alpha=1.0,
        bow=False,
        chunk_context=False,
        empty_weighting=True,
        weighting="binary",
        categorization_mode="dfs",
    )

    primitives_first = 20
    for i, doc in enumerate(document):
        p_threshold = 1e9 if i < primitives_first else 5
        w.parse_sentence(doc, threshold=p_threshold, new_vocab=True, learning=True)

    return w


def _walk_path_to_root(leaf: CobwebDiscreteNode):
    """Return [leaf, ..., root] as a list."""
    path = []
    curr = leaf
    while curr is not None:
        path.append(curr)
        curr = curr.parent
    return path


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

def test_entropy_vs_sampled_agreement(webster, capsys):
    """
    For every leaf in both hierarchies compare get_basic_entropy() against
    the ground-truth sampled get_basic(200, 100).  Prints hash- and
    depth-agreement rates plus speedup.
    """
    ltm = webster.ltm
    trees = {"content": ltm.content_hierarchy, "context": ltm.context_hierarchy}

    for tree_name, tree in trees.items():
        leaves = _collect_leaves(tree)
        if not leaves:
            continue

        s_hashes, e_hashes = [], []
        s_depths,  e_depths  = [], []

        t0 = time.perf_counter()
        for leaf in leaves:
            n = leaf.get_basic(200, MAX_NODES)
            s_hashes.append(n.concept_hash())
            s_depths.append(n.depth())
        t_sampled = time.perf_counter() - t0

        t0 = time.perf_counter()
        for leaf in leaves:
            n = leaf.get_basic_entropy()
            e_hashes.append(n.concept_hash())
            e_depths.append(n.depth())
        t_entropy = time.perf_counter() - t0

        n = len(leaves)
        h_agree = sum(a == b for a, b in zip(s_hashes, e_hashes))
        d_agree = sum(a == b for a, b in zip(s_depths,  e_depths))
        speedup = t_sampled / t_entropy if t_entropy > 0 else float("inf")

        with capsys.disabled():
            print(f"\n{'=' * 72}")
            print(f"  [{tree_name}]  {n} leaves")
            print(f"  {'Method':<28}  {'Hash agree':>18}   {'Depth agree':>18}   ms/leaf")
            print(f"  {'-'*28}  {'-'*18}   {'-'*18}   -------")
            print(f"  {'sampled get_basic(200,100)':<28}  {n}/{n} (100.0%)   "
                  f"{n}/{n} (100.0%)   {t_sampled/n*1000:.3f}ms")
            print(f"  {'entropy proxy (-H)':<28}  {h_agree}/{n} ({100*h_agree/n:5.1f}%)   "
                  f"{d_agree}/{n} ({100*d_agree/n:5.1f}%)   {t_entropy/n*1000:.3f}ms")
            print(f"  Speedup: {speedup:,.0f}x")
            print(f"{'=' * 72}")

    assert True  # output-only test


# ── Test 2: alpha ablation ─────────────────────────────────────────────────────

def test_alpha_ablation(webster, capsys):
    """
    Varies eval_alpha in get_basic() across ABLATION_ALPHAS and reports the
    hash-agreement rate with get_basic_entropy() for each value.

    Rationale: entropy() uses the tree's fixed alpha (cached counts); only the
    sampled side changes with eval_alpha.  If agreement is stable across alphas,
    the disagreements are structural (not alpha-dependent).  If agreement peaks
    at a particular alpha, that alpha best aligns the sampled EPMI objective
    with the -H proxy.
    """
    ltm = webster.ltm
    trees = {"content": ltm.content_hierarchy, "context": ltm.context_hierarchy}
    sample_size = 200  # cap to keep the test fast

    with capsys.disabled():
        tree_keys = list(trees.keys())
        print(f"\n{'=' * 72}")
        print("  Alpha ablation — hash-agreement: entropy() vs get_basic(alpha)")
        print(f"  {'alpha':>10}  " + "  ".join(f"[{t}]" for t in tree_keys))
        print(f"  {'-'*10}  " + "  ".join(f"{'─'*18}" for _ in tree_keys))

    for alpha in ABLATION_ALPHAS:
        row = [f"  {alpha:>10.1e}  "]
        for tree_name in tree_keys:
            tree = trees[tree_name]
            leaves = _collect_leaves(tree)[:sample_size]
            if not leaves:
                row.append("N/A")
                continue
            agree = sum(
                leaf.get_basic(200, MAX_NODES, False, alpha).concept_hash()
                == leaf.get_basic_entropy().concept_hash()
                for leaf in leaves
            )
            pct = 100.0 * agree / len(leaves)
            row.append(f"{agree}/{len(leaves)} ({pct:5.1f}%)")

        with capsys.disabled():
            print("  ".join(row))

    with capsys.disabled():
        print(f"{'=' * 72}")


# ── Test 3: disagreement case studies ─────────────────────────────────────────

def test_disagreement_case_studies(webster, capsys):
    """
    For up to MAX_CASE_STUDIES disagreement leaves per hierarchy, print the
    full leaf-to-root path with both -H(c) and expected_pmi(n=QUICK_EPMI_N)
    at every node, and mark which ancestor each method selects.

    This directly reveals WHY entropy disagrees: the two score functions can
    peak at different ancestors.  Key quantities printed per case:
      Δ(-H):   -H at entropy's pick  −  -H at sampled's pick
      Δ(EPMI): EPMI at sampled's pick − EPMI at entropy's pick
    """
    ltm = webster.ltm
    trees = {"content": ltm.content_hierarchy, "context": ltm.context_hierarchy}

    for tree_name, tree in trees.items():
        leaves = _collect_leaves(tree)
        if not leaves:
            continue

        disagree_cases = []
        for leaf in leaves:
            s = leaf.get_basic(200, MAX_NODES)
            e = leaf.get_basic_entropy()
            if s.concept_hash() != e.concept_hash():
                disagree_cases.append((leaf, s, e))
            if len(disagree_cases) >= MAX_CASE_STUDIES:
                break

        with capsys.disabled():
            print(f"\n{'=' * 90}")
            print(f"  [{tree_name}] Disagreement case studies  "
                  f"({len(disagree_cases)} shown, EPMI annotated with n={QUICK_EPMI_N})")
            print(f"{'=' * 90}")

        for case_idx, (leaf, sampled_node, entropy_node) in enumerate(disagree_cases):
            path = _walk_path_to_root(leaf)
            sampled_hash = sampled_node.concept_hash()
            entropy_hash = entropy_node.concept_hash()

            with capsys.disabled():
                print(f"\n  Case {case_idx + 1}:  leaf={leaf.concept_hash()[:10]}  "
                      f"sampled→depth={sampled_node.depth()}  "
                      f"entropy→depth={entropy_node.depth()}")
                print(f"  {'Depth':>5}  {'Hash':>10}  {'count':>7}  "
                      f"{'-H(c)':>12}  {'EPMI(n=50)':>12}  picks")
                print(f"  {'─'*5}  {'─'*10}  {'─'*7}  {'─'*12}  {'─'*12}  ─────")

            neg_h_vals  = []
            epmi_vals   = []
            for node in path:
                h      = node.concept_hash()
                neg_h  = -node.entropy()
                epmi   = node.expected_pmi(QUICK_EPMI_N, MAX_NODES)
                neg_h_vals.append(neg_h)
                epmi_vals.append(epmi)
                picks  = ""
                if h == sampled_hash:
                    picks += " ← sampled"
                if h == entropy_hash:
                    picks += " ← entropy"
                with capsys.disabled():
                    print(f"  {node.depth():>5}  {h[:10]:>10}  {node.count:>7}  "
                          f"{neg_h:>12.4f}  {epmi:>12.4f}{picks}")

            s_idx = path.index(sampled_node)
            e_idx = path.index(entropy_node)
            delta_h    = neg_h_vals[e_idx] - neg_h_vals[s_idx]
            delta_epmi = epmi_vals[s_idx]  - epmi_vals[e_idx]
            with capsys.disabled():
                print(f"\n  Summary:")
                print(f"    sampled pick  -H={neg_h_vals[s_idx]:+.4f}  EPMI={epmi_vals[s_idx]:+.4f}")
                print(f"    entropy pick  -H={neg_h_vals[e_idx]:+.4f}  EPMI={epmi_vals[e_idx]:+.4f}")
                print(f"    Δ(-H)   = entropy_pick − sampled_pick = {delta_h:+.4f}")
                print(f"    Δ(EPMI) = sampled_pick − entropy_pick = {delta_epmi:+.4f}")
                print(f"  {'─' * 86}")

        with capsys.disabled():
            print(f"{'=' * 90}")


# ── Test 4: timing comparison ──────────────────────────────────────────────────

def test_timing_comparison(webster, capsys):
    """Single-leaf microbenchmark: 20 repetitions of each method on a deep leaf."""
    ltm  = webster.ltm
    tree = ltm.content_hierarchy
    leaves = _collect_leaves(tree)
    if not leaves:
        pytest.skip("No content leaves found")

    leaf = next((l for l in leaves if l.depth() >= 2), leaves[0])
    N    = 20

    t0 = time.perf_counter()
    for _ in range(N):
        leaf.get_basic(200, MAX_NODES)
    t_sampled = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(N):
        leaf.get_basic_entropy()
    t_entropy = time.perf_counter() - t0

    with capsys.disabled():
        print(f"\n{'=' * 58}")
        print(f"  Microbenchmark ({N} reps, leaf depth={leaf.depth()})")
        print(f"  get_basic(200,100)  : {t_sampled/N*1000:8.2f} ms/call")
        print(f"  get_basic_entropy   : {t_entropy/N*1000:8.2f} ms/call  "
              f"({t_sampled/t_entropy:.0f}x)")
        print(f"{'=' * 58}")

    assert t_entropy <= t_sampled * 10


# ── Test 5: plots ──────────────────────────────────────────────────────────────

def test_plot_comparison(webster, capsys):
    """
    Save three diagnostic figures to tests/speedups/output/:
      scatter_entropy_vs_sampled.png  — per-hierarchy depth scatter
      alpha_ablation_curve.png        — hash-agreement vs eval_alpha (line chart)
      error_distribution.png          — histogram of depth delta (entropy − sampled)
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    ltm    = webster.ltm
    trees  = {"content": ltm.content_hierarchy, "context": ltm.context_hierarchy}
    sample = 200  # cap for ablation curve

    # ── Collect per-leaf depths ────────────────────────────────────────────────
    all_data = {}
    for tree_name, tree in trees.items():
        leaves = _collect_leaves(tree)
        if not leaves:
            continue
        s = [leaf.get_basic(200, MAX_NODES).depth()    for leaf in leaves]
        e = [leaf.get_basic_entropy().depth()          for leaf in leaves]
        all_data[tree_name] = {"sampled": s, "entropy": e, "n": len(leaves)}

    # ── Plot 1: scatter ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(all_data),
                             figsize=(5 * len(all_data), 4.5), squeeze=False)
    rng = np.random.default_rng(0)
    for col, (tree_name, data) in enumerate(all_data.items()):
        ax  = axes[0][col]
        sv  = np.array(data["sampled"])
        ev  = np.array(data["entropy"])
        n   = data["n"]
        ok  = (sv == ev)
        jit = rng.uniform(-0.15, 0.15, size=(n, 2))
        ax.scatter((sv + jit[:, 0])[~ok], (ev + jit[:, 1])[~ok],
                   c="tomato",    alpha=0.5, s=14, label="mismatch")
        ax.scatter((sv + jit[:, 0])[ ok], (ev + jit[:, 1])[ ok],
                   c="steelblue", alpha=0.5, s=14, label="agree")
        lo = min(int(sv.min()), int(ev.min())) - 0.5
        hi = max(int(sv.max()), int(ev.max())) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Sampled depth"); ax.set_ylabel("Entropy depth")
        n_ok = int(ok.sum())
        ax.set_title(f"{tree_name}\n{n_ok}/{n} ({100*n_ok/n:.1f}% agree)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("get_basic_entropy vs get_basic (depth)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scatter_entropy_vs_sampled.png"), dpi=120)
    plt.close(fig)

    # ── Plot 2: alpha ablation curve ───────────────────────────────────────────
    ablation = {t: [] for t in list(trees.keys())}
    for alpha in ABLATION_ALPHAS:
        for tree_name, tree in trees.items():
            leaves = _collect_leaves(tree)[:sample]
            if not leaves:
                ablation[tree_name].append(float("nan"))
                continue
            agree = sum(
                leaf.get_basic(200, MAX_NODES, False, alpha).concept_hash()
                == leaf.get_basic_entropy().concept_hash()
                for leaf in leaves
            )
            ablation[tree_name].append(100.0 * agree / len(leaves))

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = {"content": "#1f77b4", "context": "#ff7f0e"}
    for tree_name, vals in ablation.items():
        ax.plot(range(len(ABLATION_ALPHAS)), vals,
                marker="o", label=tree_name, color=colors.get(tree_name))
    ax.set_xticks(range(len(ABLATION_ALPHAS)))
    ax.set_xticklabels([f"{a:.0e}" for a in ABLATION_ALPHAS], fontsize=8)
    ax.set_xlabel("eval_alpha (on sampled get_basic)")
    ax.set_ylabel("Hash agreement with entropy (%)")
    ax.set_ylim(0, 105)
    ax.axhline(100, color="gray", linestyle="--", lw=0.7)
    ax.legend()
    ax.set_title("Agreement vs eval_alpha  (entropy always uses cached counts)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "alpha_ablation_curve.png"), dpi=120)
    plt.close(fig)

    # ── Plot 3: error distribution ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(all_data),
                             figsize=(5 * len(all_data), 3.5), squeeze=False)
    for col, (tree_name, data) in enumerate(all_data.items()):
        ax    = axes[0][col]
        delta = np.array(data["entropy"]) - np.array(data["sampled"])
        d_min, d_max = int(delta.min()), int(delta.max())
        bins  = np.arange(d_min - 0.5, d_max + 1.5)
        ax.hist(delta, bins=bins, color="#2ca02c", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", lw=0.8)
        ax.set_xlabel("Depth delta (entropy − sampled)")
        ax.set_ylabel("Count")
        acc = float((delta == 0).mean()) * 100
        ax.set_title(f"{tree_name}\nacc={acc:.1f}%  mean_err={float(delta.mean()):+.2f}",
                     fontsize=9)
    fig.suptitle("Entropy depth prediction error", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "error_distribution.png"), dpi=120)
    plt.close(fig)

    with capsys.disabled():
        print(f"\n[plots] Saved to {output_dir}/")
        print(f"  scatter_entropy_vs_sampled.png")
        print(f"  alpha_ablation_curve.png")
        print(f"  error_distribution.png")
