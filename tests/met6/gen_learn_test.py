"""
met6 — gen_learn_test: validate the unsupervised settings that match the
supervised reference, with parse-tree visualizations + a results table.

This is the validation harness for the settings found by the
research loop (research_loop_match_supervised.py). For a chosen grammar it:

  1. Trains a SUPERVISED reference (gold merges) and an UNSUPERVISED model
     (sentences only, τ-driven) on the SAME corpus + config.
  2. Reports the head-to-head table: parse F1 / EM vs gold, generation
     grammaticality / novelty (CYK legality oracle).
  3. RENDERS parse trees (PNG + HTML via FiniteParseTree.visualize):
       - held-out test sentences: unsupervised parse vs gold parse
       - freshly GENERATED sentences: their unsupervised parse trees
  4. Writes a markdown results report with the table + sample generated
     sentences (marked grammatical / novel).

Best settings (from research loop):
  - SMALL: τ=5  → matches supervised (F1 ~98.6 vs 100, gen 100 vs 100)
  - MED:   τ set from the loop's best-balanced pick (see SETTINGS below)

Usage:
    PYTHONHASHSEED=0 python tests/met6/gen_learn_test.py            # SMALL
    PYTHONHASHSEED=0 python tests/met6/gen_learn_test.py MED        # MED
"""
import os, sys, re, random
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from util.cfg import (TEST_GRAMMAR_SMALL, TEST_CORPUS_SMALL,
                      TEST_GRAMMAR_MED, TEST_CORPUS_MED)
from parse_mh import FiniteParseTree
import diag_supervised_gap as G

# Per-grammar validated settings (τ from research loop). n_train/epochs
# match the probe that achieved parity.
SETTINGS = {
    "SMALL": dict(grammar=TEST_GRAMMAR_SMALL, corpus=TEST_CORPUS_SMALL,
                  tau=8, n_train=150, epochs=6, flatten=("VP",)),
    "MED":   dict(grammar=TEST_GRAMMAR_MED, corpus=TEST_CORPUS_MED,
                  tau=2, n_train=150, epochs=6, flatten=("VP",)),
}
N_SHOW_TREES = 4     # held-out test sentences to render (unsup vs gold)
N_GEN_SHOW   = 4     # generated sentences to render parse trees for
N_GEN_EVAL   = 60


def render(pt, out_base):
    """Render a parse tree to PNG (+HTML). Falls back to HTML-only if the
    headless browser is unavailable."""
    try:
        pt.visualize(out_base, render_png=True)
        return out_base + ".png"
    except Exception as e:
        try:
            pt.visualize(out_base, render_png=False)
            return out_base + ".html  (PNG render failed: %s)" % e
        except Exception as e2:
            return "FAILED: %s" % e2


def main():
    gname = (sys.argv[1] if len(sys.argv) > 1 else "SMALL").upper()
    cfg = SETTINGS[gname]
    grammar, corpus, tau = cfg["grammar"], cfg["corpus"], cfg["tau"]
    out_dir = os.path.join(_HERE, "gen_learn", gname)
    tree_dir = os.path.join(out_dir, "trees")
    os.makedirs(tree_dir, exist_ok=True)

    recog, w2p = G._build_cyk_recognizer(grammar)
    data = G.gen_corpus(grammar, cfg["n_train"] + 40, seed=G.SEED, flatten=cfg["flatten"])
    random.seed(G.SEED); random.shuffle(data)
    train, test = data[:cfg["n_train"]], data[cfg["n_train"]:cfg["n_train"] + 40]
    train_set = {h["sentence"].strip() for h in train}

    scratch = G.make_webster(corpus)
    for h in data:
        for tok in re.findall(r"[\w']+|[.,!?;]", h["sentence"]):
            scratch.ltm.add_to_vocab(tok)
    gold = {h["sentence"]: G.gold_brackets(scratch, h) for h in test}

    print(f"=== gen_learn_test [{gname}] — τ={tau}, n_train={len(train)}, epochs={cfg['epochs']} ===\n")

    # ── train both ──
    print("training supervised reference (gold merges)…", flush=True)
    web_s, trees_s = G.train_supervised(corpus, train)
    sup = G.eval_model(web_s, test, gold, 30, recog, w2p, train_set, trees_s)

    print("training unsupervised (sentences only)…", flush=True)
    web_u, trees_u = G.train_unsupervised(corpus, train, tau, cfg["epochs"])
    uns = G.eval_model(web_u, test, gold, tau, recog, w2p, train_set, trees_u)

    # ── results table ──
    print(f"\n{'metric':>16} {'SUPERVISED':>12} {'UNSUPERVISED':>14}")
    for k, lbl in [("F1", "parse F1"), ("EM", "exact-match"),
                   ("gen_gram", "gen grammatical"), ("gen_novel", "gen novel")]:
        print(f"{lbl:>16} {100*sup[k]:>11.1f}% {100*uns[k]:>13.1f}%")

    # ── render parse trees: test sentences (unsup vs gold) ──
    print(f"\nrendering {N_SHOW_TREES} test parse trees (unsup vs gold) → {tree_dir}", flush=True)
    shown = [h for h in test if len(h["sentence"].split()) >= 4][:N_SHOW_TREES] or test[:N_SHOW_TREES]
    tree_notes = []
    for i, h in enumerate(shown):
        s = h["sentence"]
        pt = web_u.parse_sentence(s, threshold=tau, climb_count_threshold=tau,
                                  new_vocab=False, learning=False,
                                  maturity_gate=G.MATURITY_GATE, gate_mode="skip")
        render(pt, os.path.join(tree_dir, f"test{i}_unsup"))
        gt = FiniteParseTree(web_u.ltm, context_length=G.CONTEXT_LENGTH)
        gt.build_primitives(s, threshold="converge")
        for m in h["merges"]:
            try: gt.apply_candidate(m["left"], m["right"])
            except Exception: pass
        render(gt, os.path.join(tree_dir, f"test{i}_gold"))
        tree_notes.append((s, f"test{i}_unsup.png", f"test{i}_gold.png"))

    # ── generate + render generated trees ──
    print(f"rendering {N_GEN_SHOW} generated parse trees → {tree_dir}", flush=True)
    web_u.learn_leaf_transitions(trees_u); web_u.learn_chunk_records(trees_u)
    web_u.ltm.chunk_pool_weight = 5.0
    gen_samples = []
    seen = set()
    tries = 0
    while len(gen_samples) < N_GEN_SHOW and tries < 200:
        tries += 1
        try: text, _ = web_u.generate_via_chunk_replay()
        except Exception: continue
        text = text.strip()
        toks = text.split()
        if not toks or text in seen or len(toks) < 3:
            continue
        seen.add(text)
        gram = all(t in w2p for t in toks) and recog(toks)
        novel = text not in train_set
        idx = len(gen_samples)
        pt = web_u.parse_sentence(text, threshold=tau, climb_count_threshold=tau,
                                  new_vocab=False, learning=False,
                                  maturity_gate=G.MATURITY_GATE, gate_mode="skip")
        render(pt, os.path.join(tree_dir, f"gen{idx}"))
        gen_samples.append((text, gram, novel, f"gen{idx}.png"))

    # ── markdown report ──
    md = os.path.join(out_dir, "results.md")
    with open(md, "w") as f:
        f.write(f"# gen_learn_test — {gname} (τ={tau}, n_train={len(train)}, epochs={cfg['epochs']})\n\n")
        f.write("## Head-to-head vs supervised reference\n\n")
        f.write("| metric | SUPERVISED | UNSUPERVISED |\n|---|--:|--:|\n")
        for k, lbl in [("F1", "parse F1"), ("EM", "exact-match"),
                       ("gen_gram", "generation grammatical"), ("gen_novel", "generation novel")]:
            f.write(f"| {lbl} | {100*sup[k]:.1f}% | {100*uns[k]:.1f}% |\n")
        f.write("\n## Sample generated sentences (unsupervised)\n\n")
        for text, gram, novel, png in gen_samples:
            tags = []
            if gram: tags.append("grammatical")
            if novel: tags.append("novel")
            f.write(f"- `{text}`  ({', '.join(tags) or 'in-train / off-grammar'})  → `trees/{png}`\n")
        f.write("\n## Test parse trees (unsupervised vs gold)\n\n")
        for s, u_png, g_png in tree_notes:
            f.write(f"- `{s}`  →  unsup `trees/{u_png}`  vs  gold `trees/{g_png}`\n")
    print(f"\nreport → {md}")
    print(f"trees  → {tree_dir}")


if __name__ == "__main__":
    main()
