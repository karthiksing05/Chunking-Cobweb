"""
tests/speedups/test_basic_level_fix.py

Benchmarks 1 fast method vs sampled get_basic(200, 100):
  inst_pmi  - argmax logP_c(x) - logP_root(x)        O(height × |x|)

Tests
─────
  1. test_method_agreement      — agreement table for both hierarchies
  2. test_alpha_ablation        — sampled agreement vs eval_alpha sweep
  3. test_disagreement_cases    — per-node path trace for mismatches
  4. test_timing_comparison     — single-leaf microbenchmark (all methods)
  5. test_plot_comparison       — saves 3 diagnostic PNGs
"""

import csv
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cobweb.cobweb_discrete import CobwebDiscreteTree, CobwebDiscreteNode

SAVED_STATE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "unittests",
    "hollow_learn_test_mh", "final_ltm_data"
))

ABLATION_ALPHAS  = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
MAX_CASE_STUDIES = 5
QUICK_EPMI_N     = 50
MAX_NODES        = 100
BENCH_REPS       = 20
OUTPUT_DIR       = os.path.join(os.path.dirname(__file__), "output")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_leaves(tree: CobwebDiscreteTree):
    if tree.root is None:
        return []
    leaves, stack = [], [tree.root]
    while stack:
        node = stack.pop()
        if not node.children:
            leaves.append(node)
        else:
            stack.extend(node.children)
    return leaves


def _walk_path_to_root(leaf: CobwebDiscreteNode):
    path, curr = [], leaf
    while curr is not None:
        path.append(curr)
        curr = curr.parent
    return path


def _csv_write(path: str, fieldnames: list, rows: list):
    """Write a list-of-dicts to a CSV file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture(scope="module")
def webster():
    from parse_mh import WEBSTER

    if os.path.isdir(SAVED_STATE_DIR):
        print(f"\n[fixture] Loading saved WEBSTER state from {SAVED_STATE_DIR}")
        return WEBSTER.load_state(SAVED_STATE_DIR)

    print("\n[fixture] Training inline model (~40 sentences).")
    from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
    import random
    random.seed(42)
    document = [generate("S", TEST_GRAMMAR1) for _ in range(40)]
    w = WEBSTER(
        TEST_CORPUS1,
        context_length=3, threshold=5,
        content_alpha=1e-3, context_alpha=1e-3,
        content_bl_alpha=1e-1, context_bl_alpha=1.0,
        bow=False, chunk_context=False,
        empty_weighting=True, weighting="binary",
        categorization_mode="dfs",
    )
    for i, doc in enumerate(document):
        p_threshold = 1e9 if i < 20 else 5
        w.parse_sentence(doc, threshold=p_threshold, new_vocab=True, learning=True)
    return w


def _method_fns(leaf):
    """Return dict of method_name -> result_node for a given leaf."""
    inst = {a: {max(v, key=v.get): 1.0} for a, v in leaf.av_count.items()} if leaf.av_count else {}
    return {
        "sampled":  lambda l=leaf: l.get_basic(200, MAX_NODES),
        "inst_pmi": lambda l=leaf: l.get_basic_instance_pmi(
                        {a: {max(v, key=v.get): 1.0} for a, v in l.av_count.items()} if l.av_count else {}),
    }


# ---------------------------------------------------------------------------
# Test 1: method agreement table
# ---------------------------------------------------------------------------

def test_method_agreement(webster, capsys):
    """
    For every leaf in both hierarchies compare all 4 fast methods against
    sampled get_basic(200, 100).  Prints hash/depth agreement % and ms/leaf.
    """
    ltm   = webster.ltm
    trees = {"content": ltm.content_hierarchy, "context": ltm.context_hierarchy}
    METHODS = ["inst_pmi"]
    csv_rows = []

    for tree_name, tree in trees.items():
        leaves = _collect_leaves(tree)
        if not leaves:
            continue
        n = len(leaves)

        # Sampled ground truth
        t0 = time.perf_counter()
        sampled = [leaf.get_basic(200, MAX_NODES) for leaf in leaves]
        t_sampled = time.perf_counter() - t0
        s_hashes = [nd.concept_hash() for nd in sampled]
        s_depths = [nd.depth()        for nd in sampled]

        rows = {}
        for method in METHODS:
            t0 = time.perf_counter()
            if method == "inst_pmi":
                results = [leaf.get_basic_instance_pmi(
                    {a: {max(v, key=v.get): 1.0} for a, v in leaf.av_count.items()} if leaf.av_count else {}
                ) for leaf in leaves]
            else:
                raise ValueError(method)
            t_method = time.perf_counter() - t0
            h_agree  = sum(r.concept_hash() == s for r, s in zip(results, s_hashes))
            d_agree  = sum(r.depth()        == s for r, s in zip(results, s_depths))
            rows[method] = (h_agree, d_agree, t_method)

        csv_rows.append({"tree": tree_name, "method": "sampled",
                         "n_leaves": n, "hash_agree": n, "hash_agree_pct": 100.0,
                         "depth_agree": n, "depth_agree_pct": 100.0,
                         "ms_per_leaf": round(t_sampled / n * 1000, 4), "speedup": 1.0})
        for method, (ha, da, tm) in rows.items():
            spd = t_sampled / tm if tm > 0 else float("inf")
            csv_rows.append({"tree": tree_name, "method": method,
                             "n_leaves": n,
                             "hash_agree": ha, "hash_agree_pct": round(100 * ha / n, 2),
                             "depth_agree": da, "depth_agree_pct": round(100 * da / n, 2),
                             "ms_per_leaf": round(tm / n * 1000, 4),
                             "speedup": round(spd, 1)})

        with capsys.disabled():
            print(f"\n{'=' * 80}")
            print(f"  [{tree_name}]  {n} leaves")
            print(f"  {'Method':<22}  {'Hash agree':>18}   {'Depth agree':>18}   ms/leaf  speedup")
            print(f"  {'-'*22}  {'-'*18}   {'-'*18}   -------  -------")
            print(f"  {'sampled get_basic(200,100)':<22}  {n}/{n} (100.0%)   "
                  f"{n}/{n} (100.0%)   {t_sampled/n*1000:.3f}ms   1.0x")
            for method, (ha, da, tm) in rows.items():
                spd = t_sampled / tm if tm > 0 else float("inf")
                print(f"  {method:<22}  {ha}/{n} ({100*ha/n:5.1f}%)   "
                      f"{da}/{n} ({100*da/n:5.1f}%)   {tm/n*1000:.3f}ms   {spd:.0f}x")
            print(f"{'=' * 80}")

    _csv_write(os.path.join(OUTPUT_DIR, "method_agreement.csv"),
               ["tree", "method", "n_leaves", "hash_agree", "hash_agree_pct",
                "depth_agree", "depth_agree_pct", "ms_per_leaf", "speedup"],
               csv_rows)
    assert True


# ---------------------------------------------------------------------------
# Test 2: alpha ablation — one table per fast method
# ---------------------------------------------------------------------------

def _fast_results(method: str, leaves: list) -> list:
    """Compute fast-method results for a list of leaves (no sampling)."""
    if method == "inst_pmi":
        return [leaf.get_basic_instance_pmi(
            {a: {max(v, key=v.get): 1.0} for a, v in leaf.av_count.items()}
            if leaf.av_count else {}
        ) for leaf in leaves]
    raise ValueError(f"Unknown method: {method}")


def test_alpha_ablation(webster, capsys):
    """
    For each fast method, sweeps eval_alpha on sampled get_basic() and
    reports hash- and depth-agreement.  A separate table per technique
    makes alpha sensitivity easy to compare per method.

    Interpretation:
      - Stable across alphas → disagreements are structural (not alpha-driven)
      - Peaks at a particular alpha → that alpha best aligns sampled with method
    """
    ltm        = webster.ltm
    trees      = {"content": ltm.content_hierarchy, "context": ltm.context_hierarchy}
    FAST_METHS = ["inst_pmi"]
    sample_cap = 200

    # Pre-compute fast results (alpha-independent) for all methods × trees
    fast_cache: dict[str, dict[str, list]] = {}   # method -> tree_name -> [nodes]
    fast_hashes: dict[str, dict[str, list]] = {}  # method -> tree_name -> [hashes]
    fast_depths: dict[str, dict[str, list]] = {}  # method -> tree_name -> [depths]
    for method in FAST_METHS:
        fast_cache[method]  = {}
        fast_hashes[method] = {}
        fast_depths[method] = {}
        for tree_name, tree in trees.items():
            leaves = _collect_leaves(tree)[:sample_cap]
            res = _fast_results(method, leaves)
            fast_cache[method][tree_name]  = (leaves, res)
            fast_hashes[method][tree_name] = [r.concept_hash() for r in res]
            fast_depths[method][tree_name] = [r.depth()        for r in res]

    SEP = "=" * 72

    for method in FAST_METHS:
        with capsys.disabled():
            print(f"\n{SEP}")
            print(f"  Alpha ablation — [{method}]  "
                  f"(sampled eval_alpha swept; fast method is fixed)")
            print(f"  {'alpha':>10}  ", end="")
            for tree_name in trees:
                print(f"  {'hash agree':>22}  {'depth agree':>22}", end="")
            print()
            print(f"  {'-'*10}  ", end="")
            for _ in trees:
                print(f"  {'-'*22}  {'-'*22}", end="")
            print()

        csv_rows = []
        for alpha in ABLATION_ALPHAS:
            row = f"  {alpha:>10.1e}  "
            for tree_name, tree in trees.items():
                leaves, _ = fast_cache[method][tree_name]
                f_hashes  = fast_hashes[method][tree_name]
                f_depths  = fast_depths[method][tree_name]
                n         = len(leaves)
                if n == 0:
                    row += "  " + "N/A".center(22) + "  " + "N/A".center(22)
                    continue
                sampled = [leaf.get_basic(200, MAX_NODES, False, alpha)
                           for leaf in leaves]
                s_hashes = [s.concept_hash() for s in sampled]
                s_depths = [s.depth()        for s in sampled]
                h_agree  = sum(f == s for f, s in zip(f_hashes, s_hashes))
                d_agree  = sum(f == s for f, s in zip(f_depths, s_depths))
                row += (f"  {h_agree}/{n} ({100*h_agree/n:5.1f}%)".ljust(24) +
                        f"  {d_agree}/{n} ({100*d_agree/n:5.1f}%)".ljust(24))
                csv_rows.append({"method": method, "alpha": alpha, "tree": tree_name,
                                 "n_leaves": n,
                                 "hash_agree": h_agree,
                                 "hash_agree_pct": round(100 * h_agree / n, 2),
                                 "depth_agree": d_agree,
                                 "depth_agree_pct": round(100 * d_agree / n, 2)})
            with capsys.disabled():
                print(row)

        with capsys.disabled():
            print(SEP)
        _csv_write(os.path.join(OUTPUT_DIR, f"alpha_ablation_{method}.csv"),
                   ["method", "alpha", "tree", "n_leaves",
                    "hash_agree", "hash_agree_pct", "depth_agree", "depth_agree_pct"],
                   csv_rows)


# ---------------------------------------------------------------------------
# Test 2b: alpha ablation graphics
# ---------------------------------------------------------------------------

def test_plot_alpha_ablation(webster, capsys):
    """
    Reads the CSVs written by test_alpha_ablation and produces:
      alpha_ablation_grid.png — 4 rows (methods) × 2 cols (hash %, depth %)
        Each panel: x = eval_alpha (log scale), y = agreement %, one line
        per hierarchy (content / context).  Horizontal dashed line at 100%.
    Saved to tests/speedups/output/alpha_ablation_grid.png
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    FAST_METHS  = ["inst_pmi"]
    METRICS     = [("hash_agree_pct",  "Hash agreement (%)"),
                   ("depth_agree_pct", "Depth agreement (%)")]
    TREE_COLORS = {"content": "#1f77b4", "context": "#ff7f0e"}

    # Build data structure: method -> tree -> metric -> list[(alpha, value)]
    data: dict = {}
    for method in FAST_METHS:
        csv_path = os.path.join(OUTPUT_DIR, f"alpha_ablation_{method}.csv")
        if not os.path.exists(csv_path):
            pytest.skip(f"Run test_alpha_ablation first (missing {csv_path})")
        rows = []
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        data[method] = {}
        for row in rows:
            tree  = row["tree"]
            alpha = float(row["alpha"])
            if tree not in data[method]:
                data[method][tree] = {m: [] for m, _ in METRICS}
            for metric, _ in METRICS:
                data[method][tree][metric].append((alpha, float(row[metric])))

    nrow, ncol = len(FAST_METHS), len(METRICS)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(6 * ncol, 3.5 * nrow), squeeze=False)

    for row_i, method in enumerate(FAST_METHS):
        for col_i, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row_i][col_i]
            method_data = data.get(method, {})
            for tree_name, tree_data in method_data.items():
                pts   = sorted(tree_data[metric])          # sorted by alpha
                alphas = [p[0] for p in pts]
                vals   = [p[1] for p in pts]
                ax.plot(alphas, vals,
                        marker="o", markersize=4, linewidth=1.6,
                        color=TREE_COLORS.get(tree_name, "gray"),
                        label=tree_name)
            ax.axhline(100, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_xscale("log")
            ax.set_xlim(min(ABLATION_ALPHAS) * 0.8, max(ABLATION_ALPHAS) * 1.2)
            ax.set_ylim(0, 108)
            ax.set_xlabel("eval_alpha (sampled get_basic)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"[{method}]  {ylabel}", fontsize=9)
            ax.set_xticks(ABLATION_ALPHAS)
            ax.set_xticklabels([f"{a:.0e}" for a in ABLATION_ALPHAS], fontsize=7)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        "Alpha ablation: agreement between fast methods and sampled get_basic\n"
        "(x = eval_alpha on the sampled side; fast-method scores are fixed)",
        fontsize=11,
    )
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "alpha_ablation_grid.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)

    with capsys.disabled():
        print(f"\n[plots] Saved alpha_ablation_grid.png to {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Test 3: disagreement case studies
# ---------------------------------------------------------------------------

def test_disagreement_cases(webster, capsys):
    """
    For up to MAX_CASE_STUDIES disagreement leaves per hierarchy, print the
    full leaf-to-root path with scores for all methods at every node.
    Marks which ancestor each method selects.
    """
    ltm   = webster.ltm
    trees = {"content": ltm.content_hierarchy, "context": ltm.context_hierarchy}
    csv_rows = []

    for tree_name, tree in trees.items():
        leaves = _collect_leaves(tree)
        if not leaves:
            continue

        # Collect disagreement cases (any fast method differs from sampled)
        disagree_cases = []
        for leaf in leaves:
            s_node   = leaf.get_basic(200, MAX_NODES)
            ip_inst  = {a: {max(v, key=v.get): 1.0} for a, v in leaf.av_count.items()} if leaf.av_count else {}
            ip_node  = leaf.get_basic_instance_pmi(ip_inst)
            picks    = {
                "sampled":  s_node,
                "inst_pmi": ip_node,
            }
            s_hash = s_node.concept_hash()
            if any(n.concept_hash() != s_hash for k, n in picks.items() if k != "sampled"):
                disagree_cases.append((leaf, picks))
            if len(disagree_cases) >= MAX_CASE_STUDIES:
                break

        with capsys.disabled():
            print(f"\n{'=' * 100}")
            print(f"  [{tree_name}] Disagreement case studies ({len(disagree_cases)} shown)")
            print(f"{'=' * 100}")

        for case_idx, (leaf, picks) in enumerate(disagree_cases):
            path = _walk_path_to_root(leaf)
            s_hash = picks["sampled"].concept_hash()

            with capsys.disabled():
                print(f"\n  Case {case_idx + 1}: leaf={leaf.concept_hash()[:10]}")
                col_fmt = f"  {'Depth':>5}  {'Hash':>10}  {'count':>7}  "
                col_fmt += f"{'iPMI':>12}  {'EPMI(n=50)':>12}  picks"
                print(col_fmt)
                print(f"  {'─'*5}  {'─'*10}  {'─'*7}  {'─'*12}  {'─'*12}  ─────")

            for node in path:
                h      = node.concept_hash()
                inst   = {a: {max(v, key=v.get): 1.0} for a, v in leaf.av_count.items()} if leaf.av_count else {}
                root   = node
                while root.parent is not None:
                    root = root.parent
                ip_sc  = node.log_prob_instance(inst, 1.0) - root.log_prob_instance(inst, 1.0) if inst else 0.0
                epmi   = node.expected_pmi(QUICK_EPMI_N, MAX_NODES)
                p_tags = []
                for method_name, pick_node in picks.items():
                    if pick_node.concept_hash() == h:
                        p_tags.append(f"←{method_name}")
                csv_rows.append({"tree": tree_name, "case": case_idx + 1,
                                 "leaf_hash": leaf.concept_hash()[:10],
                                 "depth": node.depth(), "node_hash": h[:10],
                                 "count": node.count,
                                 "inst_pmi": round(ip_sc, 6), "epmi_n50": round(epmi, 6),
                                 "picks": "|".join(p_tags)})
                with capsys.disabled():
                    print(f"  {node.depth():>5}  {h[:10]:>10}  {node.count:>7}  "
                          f"{ip_sc:>12.4f}  {epmi:>12.4f}  "
                          f"{'  '.join(p_tags)}")

            with capsys.disabled():
                print(f"  {'─' * 96}")

        with capsys.disabled():
            print(f"{'=' * 100}")

    _csv_write(os.path.join(OUTPUT_DIR, "disagreement_cases.csv"),
               ["tree", "case", "leaf_hash", "depth", "node_hash", "count",
                "inst_pmi", "epmi_n50", "picks"],
               csv_rows)


# ---------------------------------------------------------------------------
# Test 4: timing comparison
# ---------------------------------------------------------------------------

def test_timing_comparison(webster, capsys):
    """Single-leaf microbenchmark: BENCH_REPS reps of each method."""
    ltm    = webster.ltm
    tree   = ltm.content_hierarchy
    leaves = _collect_leaves(tree)
    if not leaves:
        pytest.skip("No content leaves found")

    leaf = next((l for l in leaves if l.depth() >= 2), leaves[0])
    inst = {a: {max(v, key=v.get): 1.0} for a, v in leaf.av_count.items()} if leaf.av_count else {}

    timings = {}
    for name, fn in [
        ("sampled",  lambda: leaf.get_basic(200, MAX_NODES)),
        ("inst_pmi", lambda: leaf.get_basic_instance_pmi(inst)),
    ]:
        t0 = time.perf_counter()
        for _ in range(BENCH_REPS):
            fn()
        timings[name] = time.perf_counter() - t0

    t_sampled = timings["sampled"]
    with capsys.disabled():
        print(f"\n{'=' * 62}")
        print(f"  Microbenchmark ({BENCH_REPS} reps, leaf depth={leaf.depth()})")
        print(f"  {'Method':<22}  ms/call    speedup")
        print(f"  {'-'*22}  ---------  -------")
        for name, t in timings.items():
            spd = t_sampled / t if t > 0 else float("inf")
            print(f"  {name:<22}  {t/BENCH_REPS*1000:8.2f}ms  {spd:6.0f}x")
        print(f"{'=' * 62}")

    _csv_write(os.path.join(OUTPUT_DIR, "timing_comparison.csv"),
               ["method", "ms_per_call", "speedup"],
               [{"method": name,
                 "ms_per_call": round(t / BENCH_REPS * 1000, 4),
                 "speedup": round(t_sampled / t, 1) if t > 0 else float("inf")}
                for name, t in timings.items()])

    for name in ["inst_pmi"]:
        assert timings[name] <= t_sampled * 5, f"{name} not fast enough vs sampled"


# ---------------------------------------------------------------------------
# Test 5: plots
# ---------------------------------------------------------------------------

def test_plot_comparison(webster, capsys):
    """
    Save three diagnostic figures to tests/speedups/output/:
      method_agreement_bar.png   — bar chart of hash agreement % per method
      depth_scatter_grid.png     — depth scatter (fast vs sampled) for all methods
      error_distribution.png     — histogram of depth delta per method
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    ltm    = webster.ltm
    trees  = {"content": ltm.content_hierarchy, "context": ltm.context_hierarchy}
    METHODS = ["inst_pmi"]

    # Collect data
    all_data = {}
    for tree_name, tree in trees.items():
        leaves = _collect_leaves(tree)
        if not leaves:
            continue
        sampled = [leaf.get_basic(200, MAX_NODES) for leaf in leaves]
        s_hashes = [n.concept_hash() for n in sampled]
        s_depths  = [n.depth()        for n in sampled]
        method_depths = {}
        method_hash_agree = {}
        for method in METHODS:
            if method == "inst_pmi":
                results = [leaf.get_basic_instance_pmi(
                    {a: {max(v, key=v.get): 1.0} for a, v in leaf.av_count.items()} if leaf.av_count else {}
                ) for leaf in leaves]
            else:
                raise ValueError(method)
            method_depths[method]     = [r.depth() for r in results]
            method_hash_agree[method] = sum(r.concept_hash() == s
                                            for r, s in zip(results, s_hashes)) / len(leaves) * 100
        all_data[tree_name] = {
            "s_depths": s_depths, "s_hashes": s_hashes,
            "method_depths": method_depths, "method_hash_agree": method_hash_agree,
            "n": len(leaves)
        }

    tree_names = list(all_data.keys())

    # ── Plot 1: bar chart of hash agreement ─────────────────────────────────
    fig, axes = plt.subplots(1, len(tree_names), figsize=(6 * len(tree_names), 4), squeeze=False)
    colors = ["#4878cf", "#6acc65", "#d65f5f", "#b47cc7"]
    for col, tree_name in enumerate(tree_names):
        ax   = axes[0][col]
        data = all_data[tree_name]
        bars = ax.bar(METHODS, [data["method_hash_agree"][m] for m in METHODS],
                      color=colors[:len(METHODS)], alpha=0.8, edgecolor="white")
        ax.set_ylim(0, 105)
        ax.axhline(100, color="gray", linestyle="--", lw=0.8)
        ax.set_ylabel("Hash agreement with sampled (%)")
        ax.set_title(f"{tree_name}\n(n={data['n']} leaves)", fontsize=9)
        for bar, m in zip(bars, METHODS):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{data['method_hash_agree'][m]:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Fast method hash agreement vs sampled get_basic(200,100)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "method_agreement_bar.png"), dpi=120)
    plt.close(fig)

    # ── Plot 2: depth scatter grid ───────────────────────────────────────────
    rng  = np.random.default_rng(0)
    nrow = len(tree_names)
    ncol = len(METHODS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4 * nrow), squeeze=False)
    for row, tree_name in enumerate(tree_names):
        data = all_data[tree_name]
        sv   = np.array(data["s_depths"])
        for col, method in enumerate(METHODS):
            ax = axes[row][col]
            mv = np.array(data["method_depths"][method])
            n  = len(sv)
            ok = sv == mv
            jit = rng.uniform(-0.15, 0.15, size=(n, 2))
            ax.scatter((sv + jit[:,0])[~ok], (mv + jit[:,1])[~ok],
                       c="tomato",    alpha=0.5, s=12, label="mismatch")
            ax.scatter((sv + jit[:,0])[ ok], (mv + jit[:,1])[ ok],
                       c="steelblue", alpha=0.5, s=12, label="agree")
            lo = min(sv.min(), mv.min()) - 0.5
            hi = max(sv.max(), mv.max()) + 0.5
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel("Sampled depth"); ax.set_ylabel(f"{method} depth")
            n_ok = int(ok.sum())
            ax.set_title(f"{tree_name} / {method}\n{n_ok}/{n} ({100*n_ok/n:.1f}%)", fontsize=8)
            ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Depth scatter: fast methods vs sampled", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "depth_scatter_grid.png"), dpi=120)
    plt.close(fig)

    # ── Plot 3: error distribution ───────────────────────────────────────────
    fig, axes = plt.subplots(len(tree_names), len(METHODS),
                             figsize=(4.5 * len(METHODS), 3.5 * len(tree_names)), squeeze=False)
    for row, tree_name in enumerate(tree_names):
        data = all_data[tree_name]
        sv   = np.array(data["s_depths"])
        for col, method in enumerate(METHODS):
            ax    = axes[row][col]
            delta = np.array(data["method_depths"][method]) - sv
            d_min, d_max = int(delta.min()), int(delta.max())
            bins  = np.arange(d_min - 0.5, d_max + 1.5)
            ax.hist(delta, bins=bins, color="#2ca02c", alpha=0.75, edgecolor="white")
            ax.axvline(0, color="black", linestyle="--", lw=0.8)
            ax.set_xlabel("depth delta (fast − sampled)")
            ax.set_ylabel("count")
            acc = float((delta == 0).mean()) * 100
            ax.set_title(f"{tree_name} / {method}\nacc={acc:.1f}%  mean={delta.mean():+.2f}",
                         fontsize=8)
    fig.suptitle("Depth prediction error distribution", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "error_distribution.png"), dpi=120)
    plt.close(fig)

    with capsys.disabled():
        print(f"\n[plots] Saved to {output_dir}/")
        print(f"  method_agreement_bar.png")
        print(f"  depth_scatter_grid.png")
        print(f"  error_distribution.png")


# ---------------------------------------------------------------------------
# Test 6: score curve plots — sampled EPMI vs proxy score along each path
# ---------------------------------------------------------------------------

N_CURVE_LEAVES = 5   # sample leaves per panel


def test_plot_score_curves(webster, capsys):
    """
    For a sample of leaves per (tree, method) panel, plot both the sampled
    EPMI and the method's proxy score along the full leaf-to-root path.

    Both curves are z-score normalised so their shapes are comparable on the
    same axes regardless of absolute scale.  The key visual question is:
    'Do the two curves peak at the same ancestor?'

    Layout: 2 rows (content, context) × 1 col (inst_pmi).
    Each panel overlays up to N_CURVE_LEAVES leaf paths.
    x-axis: ancestor depth (0 = root, max = leaf depth).
    Vertical dashed lines: blue = sampled argmax, red = proxy argmax.

    Saved to tests/speedups/output/score_curves_grid.png
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import numpy as np

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ltm    = webster.ltm
    trees  = {"content": ltm.content_hierarchy, "context": ltm.context_hierarchy}
    METHODS = ["inst_pmi"]
    tree_names = list(trees.keys())

    def _proxy_scores(method, path_root_first, leaf):
        """Proxy score for each node in path (root first)."""
        root = path_root_first[0]
        inst = ({a: {max(v, key=v.get): 1.0} for a, v in leaf.av_count.items()}
                if leaf.av_count else {})
        lp_root = root.log_prob_instance(inst) if inst else 0.0
        return [n.log_prob_instance(inst) - lp_root
                for n in path_root_first]

    def _znorm(arr):
        a = np.array(arr, dtype=float)
        std = a.std()
        return (a - a.mean()) / (std if std > 1e-12 else 1.0)

    nrow, ncol = len(tree_names), len(METHODS)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(5 * ncol, 4 * nrow), squeeze=False)

    for row, tree_name in enumerate(tree_names):
        tree = trees[tree_name]
        all_leaves = _collect_leaves(tree)
        # prefer deeper leaves for more interesting paths
        sample_leaves = [l for l in all_leaves if l.depth() >= 2][:N_CURVE_LEAVES]
        if not sample_leaves:
            sample_leaves = all_leaves[:N_CURVE_LEAVES]

        for col, method in enumerate(METHODS):
            ax = axes[row][col]

            for leaf in sample_leaves:
                # path root-first so x-axis increases left→right
                path = list(reversed(_walk_path_to_root(leaf)))
                depths = [n.depth() for n in path]

                epmi_raw  = [n.expected_pmi(QUICK_EPMI_N, MAX_NODES) for n in path]
                proxy_raw = _proxy_scores(method, path, leaf)

                epmi_z  = _znorm(epmi_raw)
                proxy_z = _znorm(proxy_raw)

                sampled_depth = depths[int(np.argmax(epmi_raw))]
                proxy_depth   = depths[int(np.argmax(proxy_raw))]
                agree         = sampled_depth == proxy_depth

                ax.plot(depths, epmi_z,  color="steelblue", alpha=0.5, lw=1.3)
                ax.plot(depths, proxy_z, color="tomato",    alpha=0.5, lw=1.3,
                        linestyle="--")
                ax.axvline(sampled_depth, color="steelblue", alpha=0.3,
                           lw=1.0, linestyle=":")
                ax.axvline(proxy_depth, color="tomato", linestyle=":",
                           **({"alpha": 0.3, "lw": 1.0} if agree
                              else {"alpha": 0.7, "lw": 1.5}))

            ax.set_xlabel("depth from root")
            ax.set_ylabel("z-score")
            ax.set_title(f"{tree_name} / {method}", fontsize=9)
            ax.legend(handles=[
                mlines.Line2D([], [], color="steelblue", lw=1.5,
                              label="EPMI sampled"),
                mlines.Line2D([], [], color="tomato", lw=1.5, linestyle="--",
                              label=f"{method} proxy"),
                mlines.Line2D([], [], color="steelblue", lw=1.0, linestyle=":",
                              alpha=0.5, label="sampled pick"),
                mlines.Line2D([], [], color="tomato",    lw=1.0, linestyle=":",
                              alpha=0.5, label="proxy pick"),
            ], fontsize=6, loc="upper left")

    fig.suptitle(
        f"Score curves: sampled EPMI vs proxy along leaf→root path\n"
        f"(z-normalised, ≤{N_CURVE_LEAVES} leaves/panel, "
        f"EPMI evaluated with n={QUICK_EPMI_N})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "score_curves_grid.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    with capsys.disabled():
        print(f"\n[plots] Saved score_curves_grid.png to {OUTPUT_DIR}/")
