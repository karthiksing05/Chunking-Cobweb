"""Round 233 factorial summary (per draw, over ALL draws): emission x gate arms on the controlled grammar.
usage: r233_fact.py <dir> ...   (dirs .../<feats>_<arm>_s<seed>_n<n>, one arm per dir)
Columns: sem, non; new-correct by relation = distinct R-sem draws with an unseen head pair of that relation (sv, vo, an; an excludes
adjectives that accept every noun); P = precision: among draws with an unseen head pair (free adjectives excluded), share R-sem;
viol_vo = share of draws violating an object restriction. Bootstrap 95% interval over draws for NCh (all relations)."""
import glob, json, os, re, sys, random
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r233_analyse as A  # noqa: E402
FREE = {a for a, ts in A.G.ADJ.items() if len(ts) == len(A.G.NOUNS)}
def rel_new(g, hp):
    h = A.heads(g)
    if h is None:
        return set()
    return {x[0] for x in h - hp if not (x[0] == "an" and x[1] in FREE)}
rows = []
cache = {}
for d in sys.argv[1:]:
    b = os.path.basename(d.rstrip("/"))
    m = re.search(r"^(.*)_s(\d+)_n(\d+)$", b)
    arm, seed, n = m.group(1), int(m.group(2)), int(m.group(3))
    if (seed, n) not in cache:
        cache[(seed, n)] = A.train_sets(seed, n)
    seen, bi, tri, hp = cache[(seed, n)]
    fs = glob.glob(os.path.join(d, "story_s*_arm0.jsonl"))
    if not fs:
        continue
    L = [json.loads(x) for x in open(fs[0], encoding="utf-8")][1:]
    gs = [x["gen"].split() for x in L if x.get("gen")]
    N = len(gs)
    c, done, ev = Counter(), set(), []
    for g in gs:
        j, why = A.G.judge(g)
        c[j] += 1
        c["vviol"] += "object" in why
        r = rel_new(g, hp)
        first = tuple(g) not in done
        done.add(tuple(g))
        if r:
            c["withnew"] += 1
            c["withnew_sem"] += j == "sem"
        hit = j == "sem" and bool(r) and first
        ev.append(hit)
        for k in ("sv", "vo", "an"):
            c["N" + k] += j == "sem" and k in r and first
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(ev) for _ in range(N)) / N for _ in range(400))
    rows.append((arm, seed, N, c[ "sem"] / N, c["non"] / N, c["vviol"] / N, c["Nsv"] / N, c["Nvo"] / N, c["Nan"] / N, sum(ev) / N, bs[10], bs[389], c["withnew_sem"] / max(c["withnew"], 1), len(done) / N))
print("| arm | seed | N | sem | non | viol_vo | NCh sv | NCh vo | NCh an | NCh | 95% | P(sem|new pair) | distinct |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(rows):
    print(f"| {r[0]} | {r[1]} | {r[2]} | " + " | ".join(f"{v:.3f}" for v in r[3:10]) + f" | [{r[10]:.3f}, {r[11]:.3f}] | {r[12]:.3f} | {r[13]:.3f} |")
