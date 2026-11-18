"""Round 211 stage 1+2: collect every n=1000 dump (all families, all temperatures, seeds 13/17/19/23), then judge the distinct new
sentences with the two strict judges (Qwen3-8B strict, phi-4 strict; prompt and shots from r209_judge.py).
Per sentence flags: tri (has an unstored trigram), sub (proper contiguous span of a stored training sentence).
Judging uses ONE cached prefix (instruction + 16 shots) and only the sentence suffix per row; cross-checked against r209 scores.
Usage (repo root, compute node with GPU): r211_collect_judge.py <out dir> [collect|judge]
"""
import copy
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3")
sys.path.insert(0, "experiments/v3/diag")
import story_frontier as sf  # noqa: E402
import r204_fair as rf  # noqa: E402

GD = "experiments/v3/results/story_gen"
OUT = sys.argv[1]
OK_EXTRA = {"", "edge=exact", "lm=gpt", "lm=lstm", "ngram=3", "ngram=4", "ngram=5"}
JUDGES = {"Qwen3-8B/strict": "Qwen3-8B", "phi-4/strict": "phi-4"}


def collect():
    rows, seenf = [], set()
    for jl in sorted(glob.glob(f"{GD}/*frontier-n1000.jsonl")):
        if os.path.basename(jl).startswith("r210"):
            continue
        for l in open(jl, encoding="utf-8"):
            r = json.loads(l)
            if r["n"] != 1000 or (r["dir"], r["file"]) in seenf or r["dir"].startswith("r210"):
                continue
            arm = json.loads(r["arm"])
            extra = ",".join(f"{a}={b}" for a, b in sorted(arm.items()) if a not in ("temp", "causal", "units", "pair", "new", "cond"))
            if extra not in OK_EXTRA:
                continue
            seenf.add((r["dir"], r["file"]))
            rows.append(r)
    nr = {(r["dir"], r["file"]): sum(1 for _ in open(f"{GD}/{r['dir']}/{r['file']}", encoding="utf-8")) - 1 for r in rows}
    att = defaultdict(int)
    for r in rows:
        att[r["seed"]] = max(att[r["seed"]], nr[(r["dir"], r["file"])])
    print("attempted per seed", dict(att), flush=True)
    pick = {}
    for r in rows:
        fam, t = rf.family_of(r)
        k = (fam, round(t, 2), r["seed"])
        if k not in pick or nr[(r["dir"], r["file"])] >= nr[(pick[k]["dir"], pick[k]["file"])]:
            pick[k] = r
    cache, recs = {}, []
    for (fam, t, seed), r in sorted(pick.items()):
        if seed not in cache:
            cache[seed] = sf.Train(sf.train_sentences(seed, 1000))
        T = cache[seed]
        lines = [json.loads(l) for l in open(f"{GD}/{r['dir']}/{r['file']}", encoding="utf-8")][1:]
        seen, copies, newrows = {}, 0, 0
        for ln in lines:
            s = tuple(ln["gen"].split())
            if s in T.whole:
                copies += 1
                continue
            newrows += 1
            if s not in seen:
                seen[s] = (" ".join(s), any(g not in T.tri for g in sf.grams(s, 3)), s in T.sub)
        recs.append(dict(fam=fam, temp=t, seed=seed, A=att[seed], rows=len(lines), copyrows=copies, newrows=newrows,
                         sents=list(seen.values()), dir=r["dir"], file=r["file"]))
    return recs


def prefix_of(prompts):
    head, shots = prompts["strict"]
    return head + "".join(f"Sentence: {s}\nGrammatical: {a}\n\n" for s, a in shots)


def run_judge(model_key, sents, known):
    import torch
    import r209_judge as rj
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = rj.MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    YES, NO = tok.encode(" yes", add_special_tokens=False)[0], tok.encode(" no", add_special_tokens=False)[0]
    prefix = prefix_of(rj.PROMPTS)
    pid = tok(prefix, return_tensors="pt").input_ids.cuda()
    P = pid.shape[1]
    with torch.no_grad():
        pkv = model(input_ids=pid, use_cache=True).past_key_values
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    # tokenisation check: prefix ids + suffix ids must equal the joint tokenisation
    bad = 0
    for s in sents[:50]:
        suf = f"Sentence: {s}\nGrammatical:"
        j = tok(prefix + suf).input_ids
        c = pid[0].tolist() + tok(suf, add_special_tokens=False).input_ids
        bad += j != c
    print(model_key, "prefix tokens", P, "split-vs-joint tokenisation mismatches in 50:", bad, flush=True)
    order = sorted(range(len(sents)), key=lambda i: len(sents[i]))
    res, B = {}, 32
    with torch.no_grad():
        for b0 in range(0, len(order), B):
            idx = order[b0:b0 + B]
            ids = [tok(f"Sentence: {sents[i]}\nGrammatical:", add_special_tokens=False).input_ids for i in idx]
            L = max(map(len, ids))
            x = torch.full((len(idx), L), pad, dtype=torch.long)
            m = torch.zeros((len(idx), L), dtype=torch.long)
            for r, t in enumerate(ids):
                x[r, :len(t)] = torch.tensor(t)
                m[r, :len(t)] = 1
            x, m = x.cuda(), m.cuda()
            c = copy.deepcopy(pkv)
            c.batch_repeat_interleave(len(idx))
            am = torch.cat([torch.ones((len(idx), P), dtype=torch.long, device="cuda"), m], 1)
            lg = model(input_ids=x, attention_mask=am, past_key_values=c, use_cache=True).logits
            last = m.sum(1) - 1
            lp = torch.log_softmax(lg[torch.arange(len(idx)), last].float(), -1)
            y, nn = lp[:, YES].exp(), lp[:, NO].exp()
            for i, v in zip(idx, (y / (y + nn)).tolist()):
                res[sents[i]] = round(v, 5)
            if (b0 // B) % 200 == 0:
                print(model_key, b0, len(sents), flush=True)
    chk = [s for s in sents if s in known]
    if chk:
        dd = [abs(res[s] - known[s]) for s in chk]
        ag = sum((res[s] > .5) == (known[s] > .5) for s in chk) / len(chk)
        print(model_key, "cross-check vs r209 scores on", len(chk), "sentences: max |dp| %.4f mean %.5f decision agreement %.4f" % (max(dd), sum(dd) / len(dd), ag), flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    stage = sys.argv[2] if len(sys.argv) > 2 else "collect"
    sp = os.path.join(OUT, "r211_sets.json")
    if stage == "collect":
        recs = collect()
        json.dump(recs, open(sp, "w", encoding="utf-8"))
        print("settings", len(recs), "distinct new sentences", len({s for r in recs for s, _, _ in r["sents"]}), flush=True)
        return
    recs = json.load(open(sp, encoding="utf-8"))
    sents = sorted({s for r in recs for s, _, _ in r["sents"]})
    # extra sentences: the real held-out refs and controls, so the ceilings are measured with the same code path
    r209 = json.load(open(f"{GD}/r209/r209_scores.json", encoding="utf-8"))
    refs = []
    for f in sorted(glob.glob(f"{GD}/r200-n1000-refs/story_s*_n1000_refs.jsonl")):
        for ln in open(f, encoding="utf-8"):
            j = json.loads(ln)
            refs += [j["real"], j["word order scrambled"], j["content swapped"]]
    sents = sorted(set(sents) | set(refs))
    print("sentences", len(sents), flush=True)
    sc_path = os.path.join(OUT, "r211_scores.json")
    try:
        scores = json.load(open(sc_path, encoding="utf-8"))
    except (OSError, ValueError):
        scores = {}
    for jk, mk in JUDGES.items():
        if jk in scores and all(s in scores[jk] for s in sents):
            continue
        known = r209[jk]
        todo = [s for s in sents if s not in scores.get(jk, {})]
        # re-judge a 300-sentence overlap with r209 as a cross-check of the prefix-cache path; take the rest from the r209 cache
        chk = [s for s in todo if s in known][:300]
        rest = [s for s in todo if s not in set(chk)]
        res = run_judge(mk, sorted(set(rest) - set(known)) + chk, known)
        sc = scores.setdefault(jk, {})
        sc.update(res)
        for s in todo:
            if s not in sc:
                sc[s] = known[s]
        json.dump(scores, open(sc_path, "w", encoding="utf-8"))
    print("done", {k: len(v) for k, v in scores.items()}, flush=True)


if __name__ == "__main__":
    main()
