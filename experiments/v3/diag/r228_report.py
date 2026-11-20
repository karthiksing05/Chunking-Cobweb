"""Round 228: tables for the chunks-off / concepts-off controls in chess (bits per position; Chase & Simon trial-1 recall at 4,000 positions)."""
import glob, json, os
R = "experiments/v3/results/"
SETS = ["control", "family", "wing", "shift"]
SEEDS = [13, 17, 19]
out = []


def rows(j):
    return [r for r in j["by levels"] if isinstance(r.get("content level"), int) and r["copy"]]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


bits = {}
for s in SETS:
    full, leaf, off, ps, babs, brel, seeds = [], [], [], [], [], [], []
    for sd in SEEDS:
        f, o = R + f"chess_compress/r228-full-{s}_s{sd}.json", R + f"chess_compress/r228-off-{s}_s{sd}.json"
        if not (os.path.exists(f) and os.path.exists(o)):
            continue
        jf, jo = json.load(open(f)), json.load(open(o))
        full.append(min(r["bits per position"] for r in rows(jf)))
        leaf.append([r["bits per position"] for r in rows(jf) if r["content level"] == 0 and r["context level"] == 0][0])
        off.append(min(r["bits per position"] for r in jo["by levels"] if isinstance(r.get("content level"), int)))
        ps.append(jf["piece-square bits mean"])
        for tag, lst in (("bpe-abs", babs), ("bpe-rel", brel)):
            p = R + f"chess_compress/r141-{tag}_s{sd}.json"
            lst.append(json.load(open(p))["sets"][s]["best"] if os.path.exists(p) else None)
        seeds.append(sd)
    bits[s] = dict(seeds=seeds, full=mean(full), leaf=mean(leaf), off=mean(off), ps=mean(ps), babs=mean(babs), brel=mean(brel),
                   per_seed={"full": full, "leaf": leaf, "off": off})
out.append("### (a) bits per position after 4,000 positions (mean over seeds; lower is better; model cost not charged)\n")
out.append("| set | seeds | full recipe (best level) | concepts off (leaf-only level 0/0) | chunks off (max size 1) | piece-square list | BPE over squares | BPE over shapes |\n|---|---|---|---|---|---|---|---|")
for s in SETS:
    b = bits[s]
    out.append(f"| {s} | {b['seeds']} | {b['full']:.1f} | {b['leaf']:.1f} | {b['off']:.1f} | {b['ps']:.1f} | {b['babs']:.1f} | {b['brel']:.1f} |")
out.append("\nper seed (full / leaf / off): " + "; ".join(f"{s}: {bits[s]['per_seed']}" for s in SETS))
out.append("\n### (b) Chase & Simon trial-1 correct pieces at 4,000 positions (mean over seeds; 10 subjects each)\n")
out.append("| arm | seeds | K | middle | end | random middle | random end |\n|---|---|---|---|---|---|---|")
for name, pat in (("full recipe (cs73x)", "chess_cs1973_exact/cs73x_s*.json"), ("concepts off (leaf-only reads)", "chess_cs1973_exact/r228-csleaf_s*.json"),
                  ("chunks off (max size 1)", "chess_cs1973_exact/r228-csoff_s*.json")):
    js = [json.load(open(f)) for f in sorted(glob.glob(R + pat))]
    js = [j for j in js if "4000" in j["points"]]
    for K in ("K=7", "K=4"):
        vals = [mean([j["points"]["4000"]["memory"][K][t]["trial 1 correct"]["mean"] for j in js]) for t in ("middle", "end", "random middle", "random end")]
        out.append(f"| {name} | {[j['seed'] for j in js]} | {K} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    if js:
        out.append(f"|  stored chunks at 4000 (per seed) | | | {[j['points']['4000']['stored chunks'] for j in js]} | | | |")
open(R + "chess_regimes/r228_tables.md", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
