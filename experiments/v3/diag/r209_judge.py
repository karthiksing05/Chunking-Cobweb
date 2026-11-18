"""Round 209: real-sentence ceiling of stronger grammaticality judges, and the family numbers under each.

Judges = (model, prompt) pairs. Prompt v1 = the original few-shot yes/no prompt of t3_judge.py (same shots, same
P(yes)/(P(yes)+P(no)) > .5 rule). Prompt strict = a longer, clearer definition with 16 shots (fragments and agreement
errors fail; odd meaning passes). The 3B column is the cached original judge (judge_cache_Qwen_Qwen2.5-3B-Instruct.json).
Sets: real held-out refs (r200-n1000-refs, r200-n4000-refs; real / scrambled / content-swapped) and the distinct new
sentences of each family's best fair setting (n=1000 seeds 13/17; n=4000 seed 13).
Usage (compute node, repo root): r209_judge.py <out dir>
"""
import json
import os
import sys
import glob
import re
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3")
sys.path.insert(0, "experiments/v3/diag")
import story_frontier as sf  # noqa: E402
import r204_fair as rf  # noqa: E402

GD = "experiments/v3/results/story_gen"
OUT = sys.argv[1]
SETS = {  # n -> seeds, [(family, temp)]
    1000: ([13, 17], [("TRELLIS edge w7", 2.0), ("TRELLIS w6", 1.6), ("LSTM", 0.8), ("GPT", 0.8)]),
    4000: ([13], [("TRELLIS edge w5", 2.0), ("LSTM", 1.0), ("GPT", 0.9)]),
}
FRONT = {1000: ["r200", "r204", "r205"], 4000: ["r200", "r204"]}


def collect():
    recs, sents, refs = [], set(), {}
    for n, (seeds, want) in SETS.items():
        rows = []
        for tag in FRONT[n]:
            p = f"{GD}/{tag}-frontier-n{n}.jsonl"
            if os.path.exists(p):
                rows += [dict(json.loads(l), src=tag) for l in open(p, encoding="utf-8")]
        rows = [r for r in rows if r["n"] == n]
        att = defaultdict(int)
        nr = {}
        for r in rows:
            nr[(r["dir"], r["file"])] = sum(1 for _ in open(f"{GD}/{r['dir']}/{r['file']}", encoding="utf-8")) - 1
            att[r["seed"]] = max(att[r["seed"]], nr[(r["dir"], r["file"])])
        pick = {}
        for r in rows:
            fam, temp = rf.family_of(r)
            if r["seed"] in seeds and (fam, round(temp, 2)) in [(f, round(t, 2)) for f, t in want]:
                k = (fam, round(temp, 2), r["seed"])
                if k not in pick or nr[(r["dir"], r["file"])] >= nr[(pick[k]["dir"], pick[k]["file"])]:
                    if k in pick:
                        print("dupe", k, pick[k]["dir"], "->", r["dir"], flush=True)
                    pick[k] = r
        print("n", n, "found", sorted(pick), flush=True)
        for (fam, temp, seed), r in sorted(pick.items()):
            T = sf.Train(sf.train_sentences(seed, n))
            lines = [json.loads(l) for l in open(f"{GD}/{r['dir']}/{r['file']}", encoding="utf-8")][1:]
            ps = json.load(open(f"{GD}/{r['dir']}/{r['file'][:-6]}.scores.json", encoding="utf-8"))
            seen = {}
            for ln, p in zip(lines, ps):
                s = tuple(ln["gen"].split())
                if s in T.whole or s in seen:
                    continue
                seen[s] = (" ".join(s), any(g not in T.tri for g in sf.grams(s, 3)))
            sents |= {v[0] for v in seen.values()}
            recs.append(dict(n=n, fam=fam, temp=temp, seed=seed, A=att[seed], sents=list(seen.values()), dir=r["dir"]))
        R = {"real": [], "word order scrambled": [], "content swapped": []}
        for f in sorted(glob.glob(f"{GD}/r200-n{n}-refs/story_s*_n{n}_refs.jsonl")):
            for ln in open(f, encoding="utf-8"):
                j = json.loads(ln)
                for k in R:
                    R[k].append(j[k])
        refs[n] = R
        for v in R.values():
            sents |= set(v)
    return recs, sorted(sents), refs


SHOTS1 = [("the dog ran to the park", "yes"), ("she gave him a big hug", "yes"),
          ("the cat sat on the mat and purred", "yes"), ("they were happy to see the sun", "yes"),
          ("the ran dog park to the", "no"), ("she him gave hug big a", "no"),
          ("the boy happy was very go", "no"), ("we went to the and store", "no")]
HEAD1 = ("Each sentence is from a children's story, written in lowercase without punctuation. "
         "Answer yes if it is a grammatical English sentence and no if it is not.\n\n")
SHOTS2 = [("the dog ran to the park", "yes"), ("they was very happy", "no"),
          ("the ball was sad and ran home", "yes"), ("she smiled in", "no"),
          ("mom said thank you spot", "yes"), ("the girl looked and nice", "no"),
          ("one day a little bird found a shiny stone", "yes"), ("you have a nice animals", "no"),
          ("how was your tree", "yes"), ("the ran dog park to the", "no"),
          ("he wanted to play with the", "no"), ("lily and ben went to the park together", "yes"),
          ("she were so excited about the new toy", "no"), ("do not be sad", "yes"),
          ("the sun is very bright and hot", "yes"), ("we went to the and store", "no")]
HEAD2 = ("Each line is one sentence from a children's story, lowercased and without punctuation. Decide whether it is "
         "a complete, well-formed English sentence. Answer yes if every word is in a grammatical place and the "
         "sentence is complete, even if its meaning is odd, silly or unlikely. Answer no if it has wrong word "
         "order, a missing or extra word, wrong agreement (such as 'they was' or 'a nice animals'), or it stops "
         "before it is finished.\n\n")
PROMPTS = {"v1": (HEAD1, SHOTS1), "strict": (HEAD2, SHOTS2)}
MODELS = {"Qwen3-8B": "Qwen/Qwen3-8B", "Llama-3.1-8B": "meta-llama/Llama-3.1-8B", "phi-4": "microsoft/phi-4"}


def run_judge(model_key, sents):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    YES, NO = tok.encode(" yes", add_special_tokens=False)[0], tok.encode(" no", add_special_tokens=False)[0]
    print(model_key, "yes/no ids", YES, NO, repr(tok.decode([YES])), repr(tok.decode([NO])), flush=True)
    res = {}
    for pk, (head, shots) in PROMPTS.items():
        prefix = head + "".join(f"Sentence: {s}\nGrammatical: {a}\n\n" for s, a in shots)
        out = []
        with torch.no_grad():
            for i in range(0, len(sents), 32):
                b = [prefix + f"Sentence: {s}\nGrammatical:" for s in sents[i:i + 32]]
                enc = tok(b, return_tensors="pt", padding=True).to("cuda")
                lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), -1)
                y, nn = lp[:, YES].exp(), lp[:, NO].exp()
                out += (y / (y + nn)).tolist()
                if (i // 32) % 100 == 0:
                    print(model_key, pk, i, len(sents), flush=True)
        res[pk] = dict(zip(sents, [round(x, 5) for x in out]))
    del model
    torch.cuda.empty_cache()
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    recs, sents, refs = collect()
    print("sentences to judge", len(sents), flush=True)
    sp = os.path.join(OUT, "r209_scores.json")
    try:
        scores = json.load(open(sp, encoding="utf-8"))
    except (OSError, ValueError):
        scores = {}
    c3 = json.load(open(f"{GD}/judge_cache_Qwen_Qwen2.5-3B-Instruct.json", encoding="utf-8"))
    miss = [s for s in sents if s not in c3]
    print("3B cache misses", len(miss), flush=True)
    scores["3B/v1"] = {s: c3[s] for s in sents if s in c3}
    for mk in MODELS:
        if f"{mk}/v1" in scores and f"{mk}/strict" in scores:
            continue
        r = run_judge(mk, sents)
        for pk in r:
            scores[f"{mk}/{pk}"] = r[pk]
        json.dump(scores, open(sp, "w", encoding="utf-8"))
    json.dump(dict(recs=recs, refs=refs), open(os.path.join(OUT, "r209_sets.json"), "w", encoding="utf-8"))
    report(recs, refs, scores)


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def report(recs, refs, scores):
    js = list(scores)
    L = []
    for n in SETS:
        R = refs[n]
        L.append(f"\n## n={n}: real-sentence ceiling and calibration (share judged grammatical, {len(R['real'])} real held-out sentences)\n")
        L.append("| judge | real (ceiling) | scrambled | content swapped | real >= .95? |")
        L.append("|---|---|---|---|---|")
        for j in js:
            v = {k: sum(scores[j][s] > .5 for s in R[k]) / len(R[k]) for k in R}
            L.append(f"| {j} | {v['real']:.3f} | {v['word order scrambled']:.3f} | {v['content swapped']:.3f} | {'YES' if v['real'] >= .95 else 'no'} |")
        # fail examples under the best judge
        best = max(js, key=lambda j: sum(scores[j][s] > .5 for s in R["real"]))
        fails = [s for s in R["real"] if scores[best][s] <= .5][:15]
        L.append(f"\nReal sentences failed by the highest-ceiling judge ({best}), first 15: " + "; ".join(fails))
        L.append(f"\n### n={n}: per family (mean over seeds {SETS[n][0]}); gram = share of distinct new judged grammatical, gram|tri = same among distinct new with an unstored trigram, norm = / real ceiling, fair = distinct gram-novel per attempted draw\n")
        L.append("| judge | family | distinct | gram | gram norm | gram\\|tri | gram\\|tri norm | fair/draw |")
        L.append("|---|---|---|---|---|---|---|---|")
        rk = {}
        for j in js:
            ceil = sum(scores[j][s] > .5 for s in R["real"]) / len(R["real"])
            for fam, temp in SETS[n][1]:
                rr = [r for r in recs if r["n"] == n and r["fam"] == fam and abs(r["temp"] - temp) < 1e-6]
                g, gt, fr, d = [], [], [], []
                for r in rr:
                    ok = [(scores[j][s] > .5, t) for s, t in r["sents"]]
                    g.append(sum(a for a, _ in ok) / len(ok))
                    tri = [a for a, t in ok if t]
                    gt.append(sum(tri) / len(tri) if tri else 0)
                    fr.append(sum(a and t for a, t in ok) / r["A"])
                    d.append(len(ok))
                m = lambda v: sum(v) / len(v)
                rk[(j, fam)] = m(fr)
                L.append(f"| {j} | {fam} t{temp} | {m(d):.0f} | {m(g):.3f} | {m(g) / ceil:.3f} | {m(gt):.3f} | {m(gt) / ceil:.3f} | {m(fr):.3f} |")
        L.append(f"\nFamily ranking by fair/draw at n={n}:\n")
        for j in js:
            order = sorted([f for _, f in rk if _ == j], key=lambda f: -rk[(j, f)])
            L.append(f"- {j}: " + " > ".join(f"{f} ({rk[(j, f)]:.3f})" for f in order))
        ks = [(fam, temp) for fam, temp in SETS[n][1]]
        L.append(f"\nRanking by gram|tri (normalised by ceiling) at n={n}:\n")
        for j in js:
            ceil = sum(scores[j][s] > .5 for s in R["real"]) / len(R["real"])
            vals = {}
            for fam, temp in ks:
                rr = [r for r in recs if r["n"] == n and r["fam"] == fam and abs(r["temp"] - temp) < 1e-6]
                z = []
                for r in rr:
                    tri = [scores[j][s] > .5 for s, t in r["sents"] if t]
                    z.append(sum(tri) / len(tri) if tri else 0)
                vals[fam] = sum(z) / len(z) / ceil
            L.append(f"- {j}: " + " > ".join(f"{f} ({v:.3f})" for f, v in sorted(vals.items(), key=lambda x: -x[1])))
    # kappa vs 3B on pooled generated + real
    L.append("\n## Agreement with the 3B v1 judge (Cohen's kappa on binary P(yes)>.5)\n")
    L.append("| judge | kappa, generated distinct new (pooled) | kappa, real+scrambled+swapped refs | agreement rate generated |")
    L.append("|---|---|---|---|")
    gen = sorted({s for r in recs for s, _ in r["sents"]})
    rs = sorted({s for n in refs for v in refs[n].values() for s in v})
    for j in js:
        if j == "3B/v1":
            continue
        a = [scores["3B/v1"][s] > .5 for s in gen]
        b = [scores[j][s] > .5 for s in gen]
        a2 = [scores["3B/v1"][s] > .5 for s in rs]
        b2 = [scores[j][s] > .5 for s in rs]
        L.append(f"| {j} | {kappa(a, b):.3f} | {kappa(a2, b2):.3f} | {sum(x == y for x, y in zip(a, b)) / len(a):.3f} |")
    L.append(f"\n(generated pooled: {len(gen)} distinct sentences; refs: {len(rs)})")
    open(os.path.join(OUT, "r209_tables.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
