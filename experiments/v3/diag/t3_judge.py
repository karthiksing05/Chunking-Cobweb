"""Grammaticality judge for generated sentences (evaluation only: nothing it says
reaches the learner). A pretrained base LM is asked, few-shot, whether a
lowercased, unpunctuated children's-story sentence is grammatical English; the
score is P(yes) / (P(yes) + P(no)) at the answer, grammatical when above 0.5.

Calibration on the dump's reference file: held-out real sentences (should pass),
their words in random order (should fail), content words swapped by tag (grammar
kept, meaning broken: should mostly pass). Then every arm's generations: the
share judged grammatical, overall and among the novel ones, and the yield of
grammatical novel sentences per generation.

Usage: t3_judge.py <dump dir> [model] [out json]
"""
import glob
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

d = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-3B-Instruct"
out_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(d, "judge.json")
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda").eval()

SHOTS = [("the dog ran to the park", "yes"), ("she gave him a big hug", "yes"),
         ("the cat sat on the mat and purred", "yes"), ("they were happy to see the sun", "yes"),
         ("the ran dog park to the", "no"), ("she him gave hug big a", "no"),
         ("the boy happy was very go", "no"), ("we went to the and store", "no")]
HEAD = ("Each sentence is from a children's story, written in lowercase without punctuation. "
        "Answer yes if it is a grammatical English sentence and no if it is not.\n\n")
PREFIX = HEAD + "".join(f"Sentence: {s}\nGrammatical: {a}\n\n" for s, a in SHOTS)
YES, NO = tok.encode(" yes", add_special_tokens=False)[0], tok.encode(" no", add_special_tokens=False)[0]


@torch.no_grad()
def judge(sents, bs=32):
    out = []
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for i in range(0, len(sents), bs):
        batch = [PREFIX + f"Sentence: {s}\nGrammatical:" for s in sents[i:i + bs]]
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        logits = model(**enc).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, -1)
        y, n = lp[:, YES].exp(), lp[:, NO].exp()
        out += (y / (y + n)).tolist()
    return out


res = {"model": name, "references": {}, "arms": []}
for f in sorted(glob.glob(os.path.join(d, "*_refs.jsonl"))):
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    res["references"][os.path.basename(f)] = {
        k: round(sum(p > 0.5 for p in judge([r[k] for r in rows])) / len(rows), 3)
        for k in ("real", "word order scrambled", "content swapped")}
    print(json.dumps({"refs": os.path.basename(f), **res["references"][os.path.basename(f)]}), flush=True)
for f in sorted(glob.glob(os.path.join(d, "*_arm*.jsonl"))):
    lines = [json.loads(l) for l in open(f, encoding="utf-8")]
    arm, rows = lines[0]["arm"], lines[1:]
    ps = judge([r["gen"] for r in rows])
    ok = [p > 0.5 for p in ps]
    nov = [r["novel"] for r in rows]
    m = lambda v: round(sum(v) / max(len(v), 1), 3)
    row = {"file": os.path.basename(f), "arm": arm, "generated": len(rows), "judged grammatical": m(ok),
           "novel": m(nov), "novel judged grammatical": m([a for a, b in zip(ok, nov) if b]),
           "grammatical novel per generation": m([a and b for a, b in zip(ok, nov)]),
           "tag pattern attested": m([r["pattern"] for r in rows]),
           # where the new sentences fail: an attested tag order with the wrong words, or an unattested order
           "novel attested-pattern share": m([r["pattern"] for r in rows if r["novel"]]),
           "novel judged grammatical, pattern attested": m([a for a, r in zip(ok, rows) if r["novel"] and r["pattern"]]),
           "novel judged grammatical, pattern unattested": m([a for a, r in zip(ok, rows) if r["novel"] and not r["pattern"]]),
           "novel examples": [(r["gen"], round(p, 2)) for r, p in zip(rows, ps) if r["novel"]][:12]}
    res["arms"].append(row)
    print(json.dumps(row), flush=True)
    with open(f[:-len(".jsonl")] + ".scores.json", "w", encoding="utf-8") as g:   # each generation's judged probability
        json.dump([round(p, 4) for p in ps], g)
json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1)
