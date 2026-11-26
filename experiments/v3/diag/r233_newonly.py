"""Round 233: the arms read as forced-new generators (statistics among draws that are not training sentences).
usage: r233_newonly.py <dump dir> ...   -> per arm: share new, then among new draws: sem, syn, non, obj, nh, NCh (distinct sem with an
unseen head pair, per new draw), NC (distinct sem with an unseen trigram, per new draw)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r233_analyse as A  # noqa: E402
import glob, json, re  # noqa: E402
from collections import Counter  # noqa: E402
for d in sys.argv[1:]:
    m = re.search(r"_s(\d+)_n(\d+)", os.path.basename(d.rstrip("/")))
    seed, n = int(m.group(1)), int(m.group(2))
    seen, bi, tri, hp = A.train_sets(seed, n)
    print(f"\n## {os.path.basename(d.rstrip('/'))} (among new draws)\n| arm | new share | N new | sem | syn | non | obj | nh | NC | NCh |\n|---|---|---|---|---|---|---|---|---|---|")
    src = [({"grammar": 1}, [" ".join(x) for x in A.G.load_corpus(seed, n)["held"][:1600]])]
    for f in sorted(glob.glob(os.path.join(d, "story_s*_arm*.jsonl")), key=lambda p: int(re.search(r"arm(\d+)", p).group(1))):
        L = [json.loads(x) for x in open(f, encoding="utf-8")]
        src.append((L[0].get("arm", {}), [x["gen"] for x in L[1:] if x.get("gen")]))
    for arm, txt in src:
        gs = [t.split() for t in txt]
        new = [g for g in gs if tuple(g) not in seen]
        if not new:
            continue
        c, done = Counter(), set()
        for g in new:
            j, why = A.G.judge(g)
            c[j] += 1; c["obj"] += "object" in why
            h = A.heads(g); nh = bool(h - hp) if h is not None else False
            c["nh"] += nh
            first = tuple(g) not in done; done.add(tuple(g))
            c["NC"] += j == "sem" and bool(A.grams(g, 3) - tri) and first
            c["NCh"] += j == "sem" and nh and first
        N = len(new)
        lab = "grammar" if arm.get("grammar") else ("W " + str(arm.get("wreach")) + " " + str(arm.get("wemit"))) if arm.get("causal") else "T " + ",".join(f"{k}={v}" for k, v in arm.items() if k not in ("cond", "fill_name"))
        print(f"| {lab} | {len(new)/len(gs):.3f} | {N} | " + " | ".join(f"{c[k]/N:.3f}" for k in ("sem", "syn", "non", "obj", "nh", "NC", "NCh")) + " |")
