"""Phase B probes for heads (``heads.py``) on the complex grammars.

``--part suspicion`` (the night's grammar of a run, refitted): the oracle
number split of Phase A, with rule classes tracked in generation. It counts
the rule classes whose left and right tables both hold singular and plural
categories ("mixed"), and the share of the residual agreement errors whose
derivation draws a mismatched (singular, plural) pair from such a class. Then
the same split symbols with head-conditioned rules (``side="learned"``): the
same counts. Also checks that the no-heads option reproduces the current
code on the unsplit symbols.

``--part gold`` (the gold-tree grammar, ``Trellis2`` on the gold trees):
``--variant`` baseline (current), composition (heads in the composition
hierarchy and grammar only), both (and in the representation hierarchy), or
oracle (heads from the target grammar's productions: a diagnostic upper
bound, both sides). Reported as ``run_complex.py`` does (no prompting: the
prompt chart does not support heads yet).

Results: ``experiments/v2/results/heads/probe/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "results", "heads", "probe")

# Heads of the target grammars' binary productions (left/right part by
# category, number stripped): a diagnostic oracle only.
ORACLE_HEADS = {("Det", "Nb"): 1, ("Adj", "Nb"): 1, ("Nb", "PP"): 0, ("Nb", "RC"): 0, ("Nb", "Gap"): 0,
                ("Vt", "NP"): 0, ("VP", "PP"): 0, ("VP", "CoVP"): 0, ("And", "VP"): 1, ("NP", "CoNP"): 0,
                ("And", "NP"): 1, ("P", "NP"): 0, ("NP", "VP"): 1, ("Rel", "VP"): 1, ("Rel", "Gap"): 1,
                ("NP", "Vt"): 1, ("Vc", "Comp"): 0, ("That", "S"): 1, ("S", "CoS"): 0, ("But", "S"): 1}


def sample_tracked(g, rng, max_len=45):
    """A whole-tree sample with the rule class and the two parts' symbols of
    every composite: (tokens, [(c, B, C)]) or None."""
    import heads as H
    stack = [int(rng.choice(g.K, p=g.S))]
    tokens, draws = [], []
    # Expand depth-first, left to right (the read's context is the words so far).
    while stack:
        sym = stack.pop()
        c = int(rng.choice(g.M, p=g.rules(g.read(tokens, len(tokens)))[sym]))
        if rng.random() < g.pk[c]:
            tokens.append(g.vocab[int(rng.choice(len(g.vocab), p=g.E[c]))])
            if len(tokens) > max_len:
                return None
            continue
        if isinstance(g, H.HeadGrammar):
            pair = g.P2[c].ravel()
            b, d = divmod(int(rng.choice(g.K * g.K, p=pair / pair.sum())), g.K)
        else:
            b, d = int(rng.choice(g.K, p=g.Lt[c])), int(rng.choice(g.K, p=g.Rt[c]))
        draws.append((c, b, d))
        stack.append(d)
        stack.append(b)
    return tokens, draws


def suspicion(args) -> dict:
    import number_oracle as NO
    import heads as H
    import complex_grammars as CG
    from trellis2.evaluation import CFG
    cfg, blind = CFG(CG.build(args.grammar)), CFG(CG.number_blind(args.grammar))
    mem, g, train = NO.refit(args.grammar, args.train, args.seed, "night")
    res = {"part": "suspicion", "grammar": args.grammar, "train": args.train, "seed": args.seed,
           "night total bits": g.info["total bits"]}
    # Exactness: no heads reproduces the current code on the same symbols.
    plain = NO.compile_with_symbols(mem, g.elem_symbol, seed=args.seed)
    nohead = H.compile_with_symbols_heads(mem, g.elem_symbol, seed=args.seed, betas=())
    res["no-heads check: factored bits (current / heads module)"] = [plain.info["bits (factored grammar)"],
                                                                     nohead.info["bits (factored grammar)"]]
    gold_cat, gold_num = NO.element_gold(mem, train, cfg)
    split, which = NO.oracle_split(g.elem_symbol, gold_num)
    # The number of each split symbol (s, p or '' for categories not split).
    num_of = {}
    for k, nn in zip(split, [None] * len(split)):
        pass
    tag = {}
    for old, new, nn in zip(g.elem_symbol, split, gold_num):
        tag.setdefault(int(new), Counter())[nn or "-"] += 1
    sym_num = {}
    for k, cnt in tag.items():
        s_, p_ = cnt.get("s", 0), cnt.get("p", 0)
        sym_num[k] = "s" if s_ > 4 * p_ and s_ > 0 else ("p" if p_ > 4 * s_ and p_ > 0 else "-")
    train_set = {tuple(e.tokens) for e in train}
    for name, grammar in (("oracle split, current rules", NO.compile_with_symbols(mem, split, seed=args.seed)),
                          ("oracle split, head-conditioned rules",
                           H.compile_with_symbols_heads(mem, split, seed=args.seed))):
        _, s_ = np.unique(split, return_inverse=True)
        # Symbols are renumbered by _compact in increasing order of the split ids.
        order = sorted(set(int(x) for x in split))
        num = np.array([sym_num[k] for k in order])
        # Mixed classes: both parts' tables put >= 5% on singular and on plural symbols.
        P2 = getattr(grammar, "P2", None)
        if P2 is None:
            P2 = grammar.Lt[:, :, None] * grammar.Rt[:, None, :]
        lt, rt = P2.sum(axis=2), P2.sum(axis=1)
        comp = grammar.pk < 0.5

        def both(row):
            return row[num == "s"].sum() >= 0.05 and row[num == "p"].sum() >= 0.05
        mixed = [c for c in range(grammar.M) if comp[c] and both(lt[c]) and both(rt[c])]
        # Probability that a mixed class draws a mismatched (s, p) / (p, s) pair.
        mismatch = {}
        for c in mixed:
            m = P2[c] / P2[c].sum()
            mismatch[c] = float(m[np.ix_(num == "s", num == "p")].sum() + m[np.ix_(num == "p", num == "s")].sum())
        rng = np.random.default_rng(args.seed)
        n_ok = n_agree = n_struct = n_agree_mixed = n_mismatch_draw = 0
        samples = 0
        while samples < args.n_gen:
            out = sample_tracked(grammar, rng)
            if out is None:
                continue
            samples += 1
            tokens, draws = out
            ok, bl = cfg.recognizes(tokens), blind.recognizes(tokens)
            n_ok += ok
            if bl and not ok:
                n_agree += 1
                bad = [d for d in draws if d[0] in mismatch and {num[d[1]], num[d[2]]} == {"s", "p"}]
                n_agree_mixed += bool(bad)
            n_struct += not bl
            n_mismatch_draw += any(d[0] in mismatch and {num[d[1]], num[d[2]]} == {"s", "p"} for d in draws)
        res[name] = {"total bits": grammar.info["total bits"], "symbols": grammar.K, "rule classes": grammar.M,
                     "composite classes": int(comp.sum()), "mixed classes": len(mixed),
                     "mismatch probability of mixed classes (mean)": float(np.mean(list(mismatch.values()))) if mismatch else 0.0,
                     "grammatical": n_ok / samples, "agreement errors only": n_agree / samples,
                     "structural errors": n_struct / samples,
                     "agreement errors with a mismatched pair from a mixed class": n_agree_mixed / max(n_agree, 1),
                     "samples drawing a mismatched pair from a mixed class": n_mismatch_draw / samples}
        if hasattr(grammar, "side_p"):
            res[name]["head beta"] = grammar.info.get("head beta")
            res[name]["classes by side (L, R, none)"] = grammar.info.get("classes by side (L, R, none)")
        print(name, json.dumps(res[name]), flush=True)
    return res


def gold(args) -> dict:
    import heads as H
    import complex_grammars as CG
    import run_complex as RC
    from trellis2.evaluation import CFG
    variant = args.variant
    import number_oracle as NO          # imported before any substitution (it wraps compile_grammar)
    if variant in ("final_single", "final_joint"):
        import splits as SP
        subs = SP.install_final(joint=variant == "final_joint", where="consolidate")
    elif variant in ("splits", "splits_heads", "splits_joint"):
        import splits as SP
        SP.CONFIG["joint"] = variant == "splits_joint"
        subs = SP.install_splits(heads=variant == "splits_heads", representation_heads=variant == "splits_heads")
    elif variant == "composition":
        subs = H.install(representation=False, composition=True, side="learned")
    elif variant == "both":
        subs = H.install(representation=True, composition=True, side="learned")
    elif variant == "control":
        subs = H.install(representation=True, composition=True, side="learned", representation_sides=False)
    elif variant == "oracle":
        subs = H.install(representation=True, composition=True, side="oracle")
    elif variant not in ("splits", "splits_heads", "splits_joint", "final_single", "final_joint"):
        subs = []
    from trellis2 import model as M
    cfg = CFG(CG.build(args.grammar))
    RC.BLIND["cfg"] = CFG(CG.number_blind(args.grammar))
    train_all, held = CG.load(args.grammar)
    order = np.random.default_rng(args.seed).permutation(len(train_all))
    train = [train_all[i] for i in order[:args.train]]
    test = held[:args.heldout]
    train_set = {tuple(e.tokens) for e in train}
    t0 = time.time()
    if args.analyses == "night":
        # The night's stored analyses (``run_complex.py``), consolidated again.
        import pickle
        from trellis2.unsupervised import UnsupervisedLearner
        with open(args.pickle or os.path.join(HERE, "results", "complex", f"{args.grammar}_{args.train}_s{args.seed}",
                                              "grammar_night.pkl"), "rb") as f:
            saved = pickle.load(f)
        learner = UnsupervisedLearner(seed=args.seed)
        for e in train:
            learner.observe(e.tokens)
        sup = learner.fit(saved["trees"])
    else:
        sup = M.Trellis2(seed=args.seed)
        for e in train:
            sup.learn(e.tokens, e.tree)
    if variant == "oracle":
        mem = sup.memory
        side = np.full(len(mem), -1, dtype=np.int64)
        labels = {}
        for sid, e in enumerate(train):
            labels[sid] = cfg.gold_labels(e.tokens, e.tree)
        strip = lambda x: x[:-2] if x.endswith(("_s", "_p")) else x   # noqa: E731
        unknown = Counter()
        for i in range(len(mem)):
            if mem.kind[i] == mem.PRIMITIVE:
                continue
            lab = labels[mem.experience_of[i]]
            key = (strip(lab[mem.span[mem.left[i]]]), strip(lab[mem.span[mem.right[i]]]))
            if key not in ORACLE_HEADS:
                unknown[key] += 1
            side[i] = ORACLE_HEADS.get(key, 0)
        mem.oracle_side = side
        print("oracle: productions without a listed head:", dict(unknown), flush=True)
    g = sup.consolidate() if args.analyses == "gold" else sup.grammar
    seconds = time.time() - t0
    res = {"part": "gold", "analyses": args.analyses, "variant": variant, "substituted": subs, "grammar": args.grammar, "train": args.train,
           "seed": args.seed, "seconds": seconds, "total_bits": g.info["total bits"], "symbols": g.K,
           "rule_classes": g.M, "chunk_types": g.info["chunk types"],
           "head beta": g.info.get("head beta"), "classes by side (L, R, none)": g.info.get("classes by side (L, R, none)"),
           "plain grammar total bits (same symbols)": g.info.get("plain grammar total bits"),
           "representation head sides (L, R) by class": g.info.get("representation head sides (L, R) by class")}
    print(f"[{variant}] {seconds:.0f}s: {g.info['total bits']:.1f} bits, K={g.K}, M={g.M}, "
          f"beta={res['head beta']}, sides={res['classes by side (L, R, none)']}", flush=True)
    for k in ("unsplit total bits", "unsplit stage B total bits", "splits accepted", "split history", "split seconds",
              "refine: baseline total bits", "refine: stage B on unsplit symbols", "refine: splits accepted",
              "refine: joint accepted", "refine: seconds"):
        if k in g.info:
            res[k] = g.info[k]
    # Purity of the final categories by grammatical number (evaluation only).
    gold_cat, gold_num = NO.element_gold(sup.memory, train, cfg)
    pur = {}
    for group, cats in NO.GROUPS.items():
        by = {}
        for e, (cg, nn) in enumerate(zip(gold_cat, gold_num)):
            if cg is not None and NO.strip(cg) in cats and nn is not None:
                by.setdefault(int(g.elem_symbol[e]), Counter())[nn] += 1
        tot = sum(sum(v.values()) for v in by.values())
        if tot:
            pur[group] = {"purity by number": sum(max(v.values()) for v in by.values()) / tot,
                          "categories": len(by)}
    res["number purity of final categories"] = pur
    res.update(RC.evaluate(sup, test, cfg, train_set, args.seed, args.n_gen))
    print(json.dumps({k: v for k, v in res.items() if k not in ("samples", "sample_examples")}, default=str), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("suspicion", "gold"), default="gold")
    ap.add_argument("--variant", choices=("baseline", "composition", "both", "control", "oracle", "splits", "splits_heads",
                                         "splits_joint", "final_single", "final_joint"), default="composition")
    ap.add_argument("--analyses", choices=("gold", "night"), default="gold")
    ap.add_argument("--pickle", default=None, help="night analyses to refit (default: results/complex's)")
    ap.add_argument("--grammar", default="agree")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--heldout", type=int, default=300)
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = suspicion(args) if args.part == "suspicion" else gold(args)
    name = (f"suspicion_{args.grammar}_{args.train}_s{args.seed}" if args.part == "suspicion"
            else f"{args.analyses}_{args.variant}_{args.grammar}_{args.train}_s{args.seed}")
    with open(os.path.join(args.out, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
