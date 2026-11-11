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
# JUDGE_PROMPT=grammar: a second judge that is told to pass sentences whose meaning is odd and to
# fail only wrong grammar, with examples of both (the first prompt's failing examples are all
# scrambled word order, so it also fails grammatical sentences that sound unlikely). Its scores,
# cache and summary carry the tag, so the first judge's files are untouched.
PROMPT = os.environ.get("JUDGE_PROMPT", "v1")
TAG = "" if PROMPT == "v1" else "." + PROMPT + "." + name.split("/")[-1]
out_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(d, f"judge{TAG}.json")
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map=os.environ.get("JUDGE_DEVICE", "cuda")).eval()

SHOTS = [("the dog ran to the park", "yes"), ("she gave him a big hug", "yes"),
         ("the cat sat on the mat and purred", "yes"), ("they were happy to see the sun", "yes"),
         ("the ran dog park to the", "no"), ("she him gave hug big a", "no"),
         ("the boy happy was very go", "no"), ("we went to the and store", "no")]
HEAD = ("Each sentence is from a children's story, written in lowercase without punctuation. "
        "Answer yes if it is a grammatical English sentence and no if it is not.\n\n")
if PROMPT == "grammar":
    SHOTS = [("the dog ran to the park", "yes"), ("the ball was sad and ran home", "yes"),
             ("the ran dog park to the", "no"), ("they was very happy", "no"),
             ("she gave him a big hug", "yes"), ("tim is a big red box", "yes"),
             ("she smiled in", "no"), ("you have a nice animals", "no"),
             ("how was your tree", "yes"), ("the girl looked and nice", "no"),
             ("mom said thank you spot", "yes"), ("we went to the and store", "no")]
    HEAD = ("Each sentence is from a children's story, written in lowercase without punctuation. "
            "Answer yes if the sentence is grammatical English, even if its meaning is odd or unlikely. "
            "Answer no only if the grammar is wrong: wrong word order, a missing or extra word, "
            "or words that do not agree.\n\n")
PREFIX = HEAD + "".join(f"Sentence: {s}\nGrammatical: {a}\n\n" for s, a in SHOTS)
YES, NO = tok.encode(" yes", add_special_tokens=False)[0], tok.encode(" no", add_special_tokens=False)[0]


# Each distinct sentence is judged once: generations repeat within an arm and across arms and
# rounds (four draws a prompt; swaps of the same stored sentence), and a pass over 1,600
# sentences takes minutes. Scores are kept in one file beside the dump directories and reused
# by later runs of the same judge model.
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(d.rstrip("/"))), "judge_cache_" + name.replace("/", "_") + ("" if PROMPT == "v1" else "_" + PROMPT) + ".json")
try:
    CACHE = json.load(open(CACHE_PATH, encoding="utf-8"))
except (OSError, ValueError):
    CACHE = {}


def save_cache():
    try:                                            # another judge may have written meanwhile: merge, then replace
        CACHE.update({k: v for k, v in json.load(open(CACHE_PATH, encoding="utf-8")).items() if k not in CACHE})
    except (OSError, ValueError):
        pass
    tmp = CACHE_PATH + f".{os.getpid()}.tmp"
    json.dump(CACHE, open(tmp, "w", encoding="utf-8"))
    os.replace(tmp, CACHE_PATH)


def judge(sents, bs=32):
    todo = sorted({s for s in sents if s not in CACHE})
    if todo:
        for s, p in zip(todo, judge_raw(todo, bs)):
            CACHE[s] = round(p, 6)
        save_cache()
    return [CACHE[s] for s in sents]


@torch.no_grad()
def judge_raw(sents, bs=32):
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
    with open(f[:-len(".jsonl")] + f".scores{TAG}.json", "w", encoding="utf-8") as g:   # each generation's judged probability
        json.dump([round(p, 4) for p in ps], g)
json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1)
