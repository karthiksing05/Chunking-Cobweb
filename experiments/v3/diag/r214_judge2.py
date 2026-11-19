"""Round 214a: a two-axis judge for TinyStories-style sentences. SYNTAX (grammatical English, meaning ignored) and
SEMANTICS (sensible in a children's story world), each scored by log-prob yes/no, P(yes)/(P(yes)+P(no)), as r209.

Compute node, repo root, env chunks, HF_HOME=/coc/flash10/ksingara3/home_cache/huggingface HF_HUB_OFFLINE=1.

  r214_judge2.py build <out dir>               build the control sets (C1..C4) -> <out>/controls.json (NLTK tagger)
  r214_judge2.py eval <out dir> [--models M,..] score every control sentence with every prompt variant -> scores.json
  r214_judge2.py report <out dir>              confusions on dev/test halves, chosen prompt/threshold -> report.md
  r214_judge2.py score <sentences.txt> <out.tsv> [--models Qwen3-8B,phi-4] [--thr thresholds.json]
        one sentence per line; TSV columns: sentence, then per model <m>_syn_p <m>_sem_p <m>_syn_pass <m>_sem_pass,
        and syn_p / sem_p / syn_pass / sem_pass averaged (p) or AND-ed over (pass) the models.
"""
import copy
import math
import json
import os
import random
import sys
from collections import Counter, defaultdict

MODELS = {"Qwen3-8B": "Qwen/Qwen3-8B", "phi-4": "microsoft/phi-4"}
SEED = 214
N_CLASS = 300

# ---------------------------------------------------------------- prompts
# Each variant = (head, shots, field name). A shot is (sentence, "yes"/"no").
SYN_HEAD = {
    "a": ("Each line is one sentence from a children's story, lowercased and without punctuation. Decide ONLY whether "
          "it is grammatical English: every word in a grammatical place, correct agreement and word forms, and a "
          "complete sentence. IGNORE MEANING COMPLETELY. A sentence that is silly, impossible or nonsensical is still "
          "grammatical if its structure is right (for example 'the cup ate the happy sky'). Invented words count as "
          "grammatical if they sit in a grammatical frame. Answer no for wrong word order, a missing or extra word, "
          "wrong agreement such as 'they was', or an unfinished sentence.\n\n"),
}
SYN_SHOTS = [
    ("the dog ran to the park", "yes"),
    ("the ran dog park to the", "no"),
    ("the cup ate the happy sky", "yes"),
    ("the slithy toves did gyre and gimble in the wabe", "yes"),
    ("they was very happy", "no"),
    ("a purple idea slept under the loud tree", "yes"),
    ("the happy sky ate cup the", "no"),
    ("the girl looked and nice", "no"),
    ("my pencil sang the quiet river", "yes"),
    ("he wanted to play with the", "no"),
    ("one day a little bird found a shiny stone", "yes"),
    ("the mimsy borogoves were all the wabe", "no"),
    ("she were so excited about the new toy", "no"),
    ("the moon ate a sleepy chair and the blue door ran", "yes"),
    ("you have a nice animals", "no"),
    ("the frabjous bandersnatch jumped over the cold milk", "yes"),
    ("mom said thank you spot", "yes"),
    ("happy the to went she store", "no"),
    ("the rock was very hungry", "yes"),
    ("we went to the and store", "no"),
]
SEM_HEAD = {
    "a": ("Each line is one sentence from a children's story, lowercased and without punctuation. Decide whether the "
          "sentence MAKES SENSE: does it describe something sensible that could happen in a children's story world? "
          "Talking animals, toys that move, and magic are fine if the sentence is a coherent event or description. "
          "Answer no if the words do not fit together into a sensible meaning: an object doing something it cannot "
          "do, a mismatched subject, verb and object, a jumble of unrelated words, or words in a scrambled order.\n\n"),
}
SEM_SHOTS = [
    ("the dog ran to the park", "yes"),
    ("the cup ate the happy sky", "no"),
    ("she gave her mom a big hug", "yes"),
    ("the ran dog park to the", "no"),
    ("the bird sang a song in the tree", "yes"),
    ("the happy rock jumped over the sad milk", "no"),
    ("the little rabbit said hello to the sun", "yes"),
    ("the happy sky ate cup the", "no"),
    ("they was very happy to see the cat", "yes"),
    ("a purple idea slept under the loud tree", "no"),
    ("tom opened the box and found a toy", "yes"),
    ("the door sang a quiet bed", "no"),
    ("the girls is playing in the garden", "yes"),
    ("my pencil drank the kind river", "no"),
    ("the old man was sad because his dog was lost", "yes"),
    ("the slithy toves did gyre and gimble in the wabe", "no"),
    ("the flower wanted to grow tall", "yes"),
    ("lily the because ball happy run", "no"),
    ("he looked at the big red ball", "yes"),
    ("the tree ate the hot clouds and cried", "no"),
]
# variants per axis: name -> (head, shots, field). v1 = first prompts; later variants added by iteration.
SYN_HEAD_B = ("Each line is one sentence in the style of a children's story, lowercased and without punctuation. Your only "
              "job is to check SYNTAX: could the words, in this order, be a complete, well-formed English sentence? "
              "Think of the grammatical categories (noun, verb, adjective, article, preposition) and whether they are "
              "arranged and inflected correctly. Do NOT consider meaning at all: a sentence about a cup eating the sky is "
              "perfectly grammatical, and so is one made of invented words in a grammatical frame. Answer no only when "
              "the structure is broken: scrambled order, a missing or extra word, wrong agreement or verb form, or an "
              "unfinished sentence.\n\n")
SYN_SHOTS_B = [
    ("the dog ran to the park", "yes"),
    ("the car ate a happy tree", "yes"),
    ("the car ate happy a tree", "no"),
    ("the sky was very sad and the rock ran home", "yes"),
    ("the sky were very sad", "no"),
    ("tim thanked the old water", "yes"),
    ("tim thanked the", "no"),
    ("the slithy toves did gyre and gimble in the wabe", "yes"),
    ("the toves slithy gimble the in did wabe", "no"),
    ("a little house flew to the ball", "yes"),
    ("a little house flew to ball the", "no"),
    ("she gave the mud a big cold song", "yes"),
    ("they was very happy to see the cat", "no"),
    ("one day a purple idea slept under the loud tree", "yes"),
    ("the girl looked and nice", "no"),
    ("the fish decided to eat the green door", "yes"),
    ("the fish decided to the green eat door", "no"),
    ("the frabjous bandersnatch jumped over the cold milk", "yes"),
    ("he wanted to play with the", "no"),
    ("my pencil sang the quiet river", "yes"),
    ("the moon ate a sleepy chair and the blue door ran", "yes"),
    ("we went to the and store", "no"),
    ("the rock was very hungry", "yes"),
    ("you have a nice animals", "no"),
]
SEM_HEAD_B = ("Each line is one sentence from a children's story, lowercased and without punctuation. Decide whether the "
              "sentence MAKES SENSE: read it as a description of something that could happen or exist in a children's "
              "story world (talking animals, moving toys and magic are fine when the sentence is a coherent event or "
              "description). Answer no when the meaning does not hold together: the subject cannot do that action to "
              "that object, an object has a property it cannot have, or the words are an unrelated jumble. Small "
              "grammar slips do not matter here, only whether the meaning is sensible.\n\n")
SEM_SHOTS_B = [
    ("the dog ran to the park", "yes"),
    ("the car ate a happy tree", "no"),
    ("she gave her mom a big hug", "yes"),
    ("the rock was very hungry and ran home", "no"),
    ("the bird sang a song in the tree", "yes"),
    ("tim thanked the old water", "no"),
    ("the little rabbit said hello to the sun", "yes"),
    ("a little house flew to the ball", "no"),
    ("they was very happy to see the cat", "yes"),
    ("she gave the mud a big cold song", "no"),
    ("tom opened the box and found a toy", "yes"),
    ("the fish decided to eat the green door", "no"),
    ("the girls is playing in the garden", "yes"),
    ("the sky was very sad and the rock ran home", "no"),
    ("the old man was sad because his dog was lost", "yes"),
    ("the toves did gyre and gimble in the wabe", "no"),
    ("the flower wanted to grow tall", "yes"),
    ("the ran dog park to the", "no"),
    ("he looked at the big red ball", "yes"),
    ("the moon ate a sleepy chair and the blue door ran", "no"),
    ("the boy was sad because he lost his toy", "yes"),
    ("the happy tree flew a quiet cake", "no"),
    ("mom made a cake for the party", "yes"),
    ("a sun the because red went box", "no"),
]
SYN_HEAD_C = ("Each line is one sentence in the style of a children's story, lowercased and without punctuation. You are a "
              "syntax checker, not a reader. Imagine every noun, verb and adjective replaced by an invented word: does "
              "the sentence frame still work as a complete, well-formed English sentence (articles before nouns, "
              "adjectives before nouns, a verb in the right form for its subject, prepositions with objects, nothing "
              "missing, nothing in a scrambled order)? Meaning is irrelevant. A sentence in which a table eats rain, a "
              "rock smiles, or a cup is a car is fully grammatical, and so is a sentence of invented words in a "
              "grammatical frame. Answer yes whenever the frame is well-formed, however absurd the content. Answer no "
              "only when the frame itself is broken.\n\n")
SYN_SHOTS_C = [
    ("the dog ran to the park", "yes"),
    ("the happy table ate a loud rain", "yes"),
    ("the old door the laughed and slept", "no"),
    ("he smiled at a proud rock", "yes"),
    ("smiled at he a proud rock", "no"),
    ("the old door laughed and the red ball slept", "yes"),
    ("they was happy to see the sun", "no"),
    ("she thanked the small cloud", "yes"),
    ("a sad cup flew to the sun", "yes"),
    ("sad a cup flew sun the to", "no"),
    ("the glorp wibbled the snarf", "yes"),
    ("the snarf the glorp wibbled", "no"),
    ("the slithy toves did gyre and gimble in the wabe", "yes"),
    ("we found loud the blue tree", "no"),
    ("we found the loud blue tree", "yes"),
    ("my house saw surprised a stone", "no"),
    ("my house saw a surprised stone", "yes"),
    ("the girls is playing in the park", "no"),
    ("then the little moon opened the big wind", "yes"),
    ("then little the moon opened big the wind", "no"),
    ("tim was a hungry window", "yes"),
    ("tim a hungry was window", "no"),
    ("the cow jumped over a nice song", "yes"),
    ("she were so excited about the new toy", "no"),
    ("the girl picked the sleepy rain", "yes"),
    ("he wanted to play with the", "no"),
    ("a purple idea slept under the loud tree", "yes"),
    ("the girl looked and nice", "no"),
    ("one day a little bird found a shiny stone", "yes"),
    ("you have a nice animals", "no"),
    ("the moon ate a sleepy chair and the blue door ran", "yes"),
    ("we went to the and store", "no"),
]
SEM_SHOTS_C = SEM_SHOTS_B + [
    ("the box laughed at a proud tree", "no"),
    ("the boy laughed at a funny dog", "yes"),
    ("the old door opened the happy ball", "no"),
    ("the cat found a toy under the bed", "yes"),
    ("the house was surprised and ran to the park", "no"),
    ("the girl was proud of her new red hat", "yes"),
]
VARIANTS = {"syn": {"v1": (SYN_HEAD["a"], SYN_SHOTS, "Grammatical"),
                    "v2": (SYN_HEAD_B, SYN_SHOTS_B, "Grammatical"),
                    "v3": (SYN_HEAD_C, SYN_SHOTS_C, "Grammatical"),
                    "v4": (SYN_HEAD_B, SYN_SHOTS_C, "Grammatical"),
                    "v5": (SYN_HEAD_C, SYN_SHOTS_B, "Grammatical")},
            "sem": {"v1": (SEM_HEAD["a"], SEM_SHOTS, "Makes sense"),
                    "v2": (SEM_HEAD_B, SEM_SHOTS_B, "Makes sense"),
                    "v3": (SEM_HEAD_B, SEM_SHOTS_C, "Makes sense")}}


def build_prefix(axis, variant):
    head, shots, field = VARIANTS[axis][variant]
    return head + "".join(f"Sentence: {s}\n{field}: {a}\n\n" for s, a in shots), field

# ---------------------------------------------------------------- controls
FUNC = set("""is was were are am be been being has have had do does did will would can could should may might must
not very so too then just also there here now all up out down off to of in on at for with from by as that this these those
it its he she they we you i me him her them us my your his their our one two some any no yes more most much many
said say says""".split())
EXCLUDE = set("anna ben bob dad lucy max mia mom mommy sam sara spot sue tim tom lily don't everyone heard i'm kind lot lots something sorry thank idea time day home share fun ground over pretty long other".split())  # names and tagger errors
CONTENT_TAGS = {"NN", "NNS", "VB", "VBD", "JJ"}  # adverbs are left alone: their position is too free to swap blindly
VERB_FRAME = {w: "INTR" for w in "came flew laughed lived played ran smiled walked went go run play".split()}
VERB_FRAME.update({w: "TR" for w in "found liked loved opened picked saw took thanked made eat find see take make".split()})
ANIM = set("bear bird boy cat dog fish friend girl man animals friends people".split())
EMO = set("happy sad proud surprised careful nice".split())
# C2 swaps flip these classes (animate <-> inanimate noun, emotion <-> physical adjective) so the result is anomalous, not sensible by chance
FLIP = {"was": "were", "were": "was", "is": "are", "are": "is", "has": "have", "have": "has"}


def swap_cands(pl, w):
    """Same-POS replacements for w; nouns flip animacy and adjectives flip emotion/physical so the result is anomalous."""
    base = [x for x in pl if x != w]
    if any(x in ANIM for x in pl):
        flip = [x for x in base if (x in ANIM) != (w in ANIM)]
    elif any(x in EMO for x in pl):
        flip = [x for x in base if (x in EMO) != (w in EMO)]
    else:
        flip = base
    return flip or base



def tag_sentence(words):
    import nltk
    return [t for _, t in nltk.pos_tag(words)]


def norm_tag(t):
    return {"VBP": "VB", "VB": "VB"}.get(t, t)


def build(out):
    import nltk
    sys.path.insert(0, "src")
    sys.path.insert(0, "experiments/v3")
    import story_frontier as sf
    sf.train_sentences(13, 10)  # populates the story stream and the 250-word vocabulary
    vocab, stories = sf._STREAM["vocab"], list(sf._STREAM["stories"])
    random.Random(13).shuffle(stories)  # the first 1500 shuffled stories are the held-out ones of story_gen
    held = stories[:1500]
    keep = lambda s: 3 <= len(s) <= 8 and all(w in vocab for w in s)
    sents, seen = [], set()
    for st in held:
        for s in st:
            if keep(s) and tuple(s) not in seen:
                seen.add(tuple(s))
                sents.append(s)
    rng = random.Random(SEED)
    rng.shuffle(sents)
    print("held-out kept distinct sentences", len(sents), flush=True)
    # word -> majority tag over the held-out sentences (in-context tagging)
    tags = [tag_sentence(s) for s in sents]
    cnt = defaultdict(Counter)
    for s, t in zip(sents, tags):
        for w, x in zip(s, t):
            cnt[w][norm_tag(x)] += 1
    maj = {w: c.most_common(1)[0][0] for w, c in cnt.items()}
    pool, pkey = defaultdict(list), {}
    for w, x in maj.items():
        # a word joins a pool only when its majority tag is content, it is not a function word, and the tag is >= 60% of uses
        if w not in FUNC and w not in EXCLUDE and x in CONTENT_TAGS and cnt[w][x] / sum(cnt[w].values()) >= .6 and cnt[w][x] >= 3:
            if x in ("VB", "VBD"):  # verbs swap only inside their argument frame, so the sentence stays grammatical
                if w not in VERB_FRAME:
                    continue
                k = f"{x}-{VERB_FRAME[w]}"
            else:
                k = x
            pool[k].append(w)
            pkey[w] = k
    pool = {k: sorted(v) for k, v in pool.items() if len(v) >= 4}
    pkey = {w: k for w, k in pkey.items() if k in pool}
    print("pools", {k: len(v) for k, v in pool.items()}, flush=True)

    def slots(s, t):
        out = []
        for i, (w, x) in enumerate(zip(s, t)):
            if w not in pkey or maj.get(w) != norm_tag(x):
                continue
            if pkey[w].split("-")[0] in ("VB", "VBD"):
                nxt = s[i + 1] if i + 1 < len(s) else None
                if nxt == "to" or (pkey[w].endswith("TR") and not pkey[w].endswith("INTR") and nxt is None):
                    continue  # a verb that governs "to" or lacks its object would break the frame when swapped
            out.append(i)
        return out

    base = []  # sentences with at least 2 swappable content slots, to make C2 full a real swap
    for s, t in zip(sents, tags):
        sl_ = slots(s, t)
        if len(sl_) >= 2 and any(pkey[s[i]] != "JJ" for i in sl_):  # a noun or verb must be swappable, else only adjectives change
            base.append((s, t))
    print("base candidates with >=2 swappable slots", len(base), flush=True)
    base = base[:N_CLASS]
    sets = defaultdict(list)
    r = random.Random(SEED + 1)
    bi = Counter()  # corpus bigram counts over every TinyStories sentence (padded); the swap picker prefers unattested bigrams
    for st in stories:
        for x in st:
            xs = ["<s>"] + x + ["</s>"]
            bi.update(zip(xs, xs[1:]))

    def unatt(ws, idx):
        xs = ["<s>"] + ws + ["</s>"]
        return sum(bi[(xs[i], xs[i + 1])] == 0 for j in idx for i in (j, j + 1))  # bigrams touching slot j (shifted by <s>)

    def swapped(s, idxs, need):
        """Random same-POS swaps at idxs; best of 60 draws by the number of unattested bigrams touching a swapped slot."""
        best, bs_ = None, -1
        for _ in range(60):
            w = list(s)
            for i in idxs:
                w[i] = r.choice(swap_cands(pool[pkey[s[i]]], s[i]))
            for j in range(len(w) - 1):  # keep a/an agreement with the new next word
                if w[j] in ("a", "an"):
                    w[j] = "an" if w[j + 1][0] in "aeiou" else "a"
            u = unatt(w, idxs)
            if u > bs_:
                best, bs_ = w, u
            if u >= need:
                break
        return best
    for s, t in base:
        sl = slots(s, t)
        sets["C1"].append(" ".join(s))
        full = swapped(s, sl, len(sl))
        sets["C2full"].append(" ".join(full))
        one = swapped(s, [r.choice(sl)], 1)
        sets["C2one"].append(" ".join(one))
        sc = list(s)
        for _ in range(100):
            r.shuffle(sc)
            if sc != list(s):
                break
        sets["C3scr"].append(" ".join(sc))
        cand = [i for i in range(len(s) - 1) if norm_tag(t[i]) != norm_tag(t[i + 1]) and s[i] != s[i + 1]]
        if cand:
            i = r.choice(cand)
            ad = list(s)
            ad[i], ad[i + 1] = ad[i + 1], ad[i]
        else:
            ad = list(s)
        sets["C3adj"].append(" ".join(ad))
    # base index kept for pairing; C3adj entries equal to the source (no different-POS neighbours) are dropped later
    sets["base_idx"] = list(range(len(base)))
    # C4: aux agreement errors from all held-out sentences (not only base), subjects i/you excluded
    c4 = []
    for s in sents:
        if any(w in FLIP for w in s) and not ({"i", "you", "and"} & set(s)):
            j = [i for i, w in enumerate(s) if w in FLIP][0]
            e = list(s)
            e[j] = FLIP[s[j]]
            c4.append((" ".join(s), " ".join(e)))
        if len(c4) >= N_CLASS:
            break
    sets["C4"] = [e for _, e in c4]
    sets["C4_src"] = [o for o, _ in c4]
    # fixed dev/test split: alternate by index within each class
    os.makedirs(out, exist_ok=True)
    json.dump(dict(sets), open(os.path.join(out, "controls.json"), "w", encoding="utf-8"), indent=0)
    for k, v in sets.items():
        print(k, len(v), v[:3])


# ---------------------------------------------------------------- scoring
def load(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model


def p_yes(tok, model, prefix, field, sents, bs=32):
    import torch
    Y, N = tok.encode(" yes", add_special_tokens=False)[0], tok.encode(" no", add_special_tokens=False)[0]
    out = []
    # the few-shot prefix is run once and its KV cache reused (tokenised separately; prefix ends in a newline, so the
    # split is clean); suffixes are right-padded and read at their last real token
    pid = tok(prefix, return_tensors="pt", add_special_tokens=True).input_ids.to("cuda")
    with torch.no_grad():
        pre = model(input_ids=pid, use_cache=True).past_key_values
        L0 = pid.shape[1]
        for i in range(0, len(sents), bs):
            sub = sents[i:i + bs]
            ids = [tok(f"Sentence: {s}\n{field}:", add_special_tokens=False).input_ids for s in sub]
            m = max(len(x) for x in ids)
            inp = torch.tensor([x + [tok.pad_token_id] * (m - len(x)) for x in ids], device="cuda")
            sm = torch.tensor([[1] * len(x) + [0] * (m - len(x)) for x in ids], device="cuda")
            cache = copy.deepcopy(pre)
            cache.batch_repeat_interleave(len(sub))
            am = torch.cat([torch.ones(len(sub), L0, dtype=sm.dtype, device="cuda"), sm], 1)
            lg = model(input_ids=inp, attention_mask=am, past_key_values=cache, use_cache=True).logits
            last = torch.tensor([len(x) - 1 for x in ids], device="cuda")
            lp = torch.log_softmax(lg[torch.arange(len(sub)), last].float(), -1)
            y, n = lp[:, Y].exp(), lp[:, N].exp()
            out += (y / (y + n)).tolist()
    return [round(x, 5) for x in out]


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def do_eval(out):
    C = json.load(open(os.path.join(out, "controls.json"), encoding="utf-8"))
    allsent = sorted({s for k, v in C.items() if k not in ("base_idx",) for s in v})
    sp = os.path.join(out, "scores.json")
    try:
        scores = json.load(open(sp, encoding="utf-8"))
    except (OSError, ValueError):
        scores = {}
    for mk in arg("--models", ",".join(MODELS)).split(","):
        tok, model = load(MODELS[mk])
        for axis in ("syn", "sem"):
            for v in VARIANTS[axis]:
                key = f"{mk}/{v}/{axis}"
                if key in scores:
                    continue
                prefix, field = build_prefix(axis, v)
                scores[key] = dict(zip(allsent, p_yes(tok, model, prefix, field, allsent)))
                print("scored", key, flush=True)
                json.dump(scores, open(sp, "w", encoding="utf-8"))
        del model
        import torch
        torch.cuda.empty_cache()


def do_score():
    inp, outp = sys.argv[2], sys.argv[3]
    sents = [l.rstrip("\n") for l in open(inp, encoding="utf-8") if l.strip()]
    cfg = json.load(open(arg("--thr", os.path.join(os.path.dirname(os.path.abspath(__file__)), "r214_judge2_thresholds.json")),
                         encoding="utf-8"))
    models = arg("--models", ",".join(cfg["models"])).split(",")
    cols, res = ["sentence"], {}
    for mk in models:
        tok, model = load(MODELS[mk])
        for axis in ("syn", "sem"):
            c = cfg["judges"][mk][axis]
            prefix, field = build_prefix(axis, c["variant"])
            res[(mk, axis)] = (p_yes(tok, model, prefix, field, sents), c["thr"])
        del model
        import torch
        torch.cuda.empty_cache()
    for axis in ("syn", "sem"):
        for mk in models:
            cols += [f"{mk}_{axis}_p", f"{mk}_{axis}_pass"]
        cols += [f"{axis}_p", f"{axis}_pass"]
    with open(outp, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for i, s in enumerate(sents):
            row = [s]
            for axis in ("syn", "sem"):
                ps = []
                pf = []
                for mk in models:
                    p, thr = res[(mk, axis)]
                    row += [f"{p[i]:.4f}", str(int(p[i] > thr))]
                    ps.append(p[i])
                    pf.append(p[i] > thr)
                if len(models) == len(cfg["models"]):  # ensemble: mean logit of the models, thr chosen on dev
                    z = sum(math.log(min(max(x, 1e-5), 1 - 1e-5) / (1 - min(max(x, 1e-5), 1 - 1e-5))) for x in ps) / len(ps)
                    row += [f"{1 / (1 + math.exp(-z)):.4f}", str(int(z > cfg["ensemble_logit_thr"][axis]))]
                else:  # a single model: its own flag
                    row += [f"{ps[0]:.4f}", str(int(pf[0]))]
            f.write("\t".join(row) + "\n")
    print("wrote", outp, len(sents))


def halves(n):
    return {"dev": [i for i in range(n) if i % 2 == 0], "test": [i for i in range(n) if i % 2 == 1]}


def rate(scores, sents, thr):
    return sum(scores[s] > thr for s in sents) / len(sents)


THR_GRID = [.01, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
CL = ["C1", "C2full", "C2one", "C3scr", "C3adj", "C4"]


def classes(C):
    cls = {k: C[k] for k in CL}
    keep = [i for i, (s, o) in enumerate(zip(C["C3adj"], C["C1"])) if s != o]  # drop adjacent-swap items equal to the source
    cls["C3adj"] = [C["C3adj"][i] for i in keep]
    return cls


def do_report(out):
    C = json.load(open(os.path.join(out, "controls.json"), encoding="utf-8"))
    S = json.load(open(os.path.join(out, "scores.json"), encoding="utf-8"))
    cls = classes(C)
    H = {k: halves(len(v)) for k, v in cls.items()}
    sub = lambda k, h: [cls[k][i] for i in H[k][h]]
    rt = lambda key, k, h, t: rate(S[key], sub(k, h), t)

    def margin(axis, key, h, t):
        r = {k: rt(key, k, h, t) for k in CL}
        if axis == "syn":
            return min(r["C1"] - .90, r["C2full"] - .80, .10 - max(r["C3scr"], r["C3adj"]))
        return min(r["C1"] - .85, .20 - r["C2full"])

    chosen, L = {}, []
    for mk in MODELS:
        for axis in ("syn", "sem"):
            opts = [(margin(axis, k, "dev", t), k.split("/")[1], t) for k in S if k.startswith(mk + "/") and k.endswith("/" + axis)
                    for t in THR_GRID]
            best = max(opts)
            chosen[(mk, axis)] = best
            L.append(f"\n### {mk} {axis}: all (variant, thr) by dev margin (top 6)")
            for o in sorted(opts, reverse=True)[:6]:
                L.append(f"- {o[1]} thr={o[2]} dev margin {o[0]:+.3f}")
            L.append(f"chosen on dev: {best[1]} thr={best[2]}")
            for h in ("dev", "test"):
                key = f"{mk}/{best[1]}/{axis}"
                L.append(f"{h}: " + " ".join(f"{k}={rt(key, k, h, best[2]):.3f}" for k in CL) + f"  margin {margin(axis, key, h, best[2]):+.3f}")
    L.append("\n## Per-variant at thr .5 (dev | test)")
    for key in sorted(S):
        L.append(f"{key}: " + " ".join(f"{k} {rt(key, k, 'dev', .5):.2f}|{rt(key, k, 'test', .5):.2f}" for k in CL))
    # 2x2 confusions on the test half (and all) with the chosen judges; rows = classes
    for hname in ("test", "all"):
        L.append(f"\n## 2x2 confusion ({hname} half): syn-pass/sem-pass counts per class (SY=syntax pass, SM=semantics pass)")
        for mk in MODELS:
            ks, km = chosen[(mk, "syn")], chosen[(mk, "sem")]
            L.append(f"\n{mk} (syn {ks[1]} thr {ks[2]}, sem {km[1]} thr {km[2]})\n")
            L.append("| class | n | SY+SM+ | SY+SM- | SY-SM+ | SY-SM- | syn pass | sem pass |")
            L.append("|---|---|---|---|---|---|---|---|")
            for k in CL:
                ss = cls[k] if hname == "all" else sub(k, "test")
                a = [(S[f"{mk}/{ks[1]}/syn"][s] > ks[2], S[f"{mk}/{km[1]}/sem"][s] > km[2]) for s in ss]
                c = Counter(a)
                L.append(f"| {k} | {len(ss)} | {c[(True, True)]} | {c[(True, False)]} | {c[(False, True)]} | {c[(False, False)]} | "
                         f"{sum(x for x, _ in a) / len(a):.3f} | {sum(y for _, y in a) / len(a):.3f} |")
    # ensemble: mean logit of the two models' dev-chosen variants, threshold chosen on dev (logit scale)
    import math
    lg = lambda p: math.log(min(max(p, 1e-5), 1 - 1e-5) / (1 - min(max(p, 1e-5), 1 - 1e-5)))
    ens, ens_thr = {}, {}
    allsents = sorted({s for k in CL for s in cls[k]})
    for axis in ("syn", "sem"):
        ens[axis] = {s: sum(lg(S[f"{mk}/{chosen[(mk, axis)][1]}/{axis}"][s]) for mk in MODELS) / len(MODELS) for s in allsents}
        grid = [x / 4 for x in range(-40, 25)]
        S[f"ens/x/{axis}"] = ens[axis]
        best = max((margin(axis, f"ens/x/{axis}", "dev", t), t) for t in grid)
        ens_thr[axis] = best[1]
        L.append(f"\n### ensemble {axis} (mean logit of {', '.join(f'{mk}:{chosen[(mk, axis)][1]}' for mk in MODELS)}); chosen on dev thr(logit)={best[1]}")
        for h in ("dev", "test"):
            L.append(f"{h}: " + " ".join(f"{k}={rt(f'ens/x/{axis}', k, h, best[1]):.3f}" for k in CL)
                     + f"  margin {margin(axis, f'ens/x/{axis}', h, best[1]):+.3f}")
    L.append("\n## Ensemble judge 2x2 confusion, test half and all")
    for hname in ("test", "all"):
        L.append(f"\n({hname})\n")
        L.append("| class | n | SY+SM+ | SY+SM- | SY-SM+ | SY-SM- | syn pass | sem pass |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k in CL:
            ss = cls[k] if hname == "all" else sub(k, "test")
            a = [(ens["syn"][s] > ens_thr["syn"], ens["sem"][s] > ens_thr["sem"]) for s in ss]
            c = Counter(a)
            L.append(f"| {k} | {len(ss)} | {c[(True, True)]} | {c[(True, False)]} | {c[(False, True)]} | {c[(False, False)]} | "
                     f"{sum(x for x, _ in a) / len(a):.3f} | {sum(y for _, y in a) / len(a):.3f} |")
    # both models together (AND of the two judges per axis)
    L.append("\n## Both models (AND per axis), test half")
    L.append("| class | n | SY+SM+ | SY+SM- | SY-SM+ | SY-SM- | syn pass | sem pass |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in CL:
        ss = sub(k, "test")
        a = []
        for s in ss:
            sy = all(S[f"{mk}/{chosen[(mk, 'syn')][1]}/syn"][s] > chosen[(mk, "syn")][2] for mk in MODELS)
            sm = all(S[f"{mk}/{chosen[(mk, 'sem')][1]}/sem"][s] > chosen[(mk, "sem")][2] for mk in MODELS)
            a.append((sy, sm))
        c = Counter(a)
        L.append(f"| {k} | {len(ss)} | {c[(True, True)]} | {c[(True, False)]} | {c[(False, True)]} | {c[(False, False)]} | "
                 f"{sum(x for x, _ in a) / len(a):.3f} | {sum(y for _, y in a) / len(a):.3f} |")
    # five examples per class (first five of the test half) with the chosen judges
    L.append("\n## Examples (first five of the test half; P(yes) syn / sem per model)")
    for k in CL:
        L.append(f"\n{k}")
        for s in sub(k, "test")[:5]:
            L.append(f"- {s} :: " + "; ".join(
                f"{mk} {S[f'{mk}/{chosen[(mk, 'syn')][1]}/syn'][s]:.2f}/{S[f'{mk}/{chosen[(mk, 'sem')][1]}/sem'][s]:.2f}" for mk in MODELS))
    thr = dict(ensemble_logit_thr=ens_thr, models=list(MODELS), judges={mk: {ax: dict(variant=chosen[(mk, ax)][1], thr=chosen[(mk, ax)][2])
                                                for ax in ("syn", "sem")} for mk in MODELS})
    json.dump(thr, open("experiments/v3/diag/r214_judge2_thresholds.json", "w", encoding="utf-8"), indent=1)
    open(os.path.join(out, "report_tables.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2])
    elif cmd == "eval":
        do_eval(sys.argv[2])
    elif cmd == "report":
        do_report(sys.argv[2])
    elif cmd == "score":
        do_score()
