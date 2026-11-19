"""Round 214 Phase 0' word-class map (no generation). Reads the R214_WORDMAP dump of r214_story.py: for each nb_bits b, the learned
name of every vocabulary word (truncated to b bits). Groups the words by name and reports, token-weighted over the training stream:
the number of classes, classes with >= 3 words, slot purity (majority spaCy UPOS per class, tagged in sentence context), and the
occurrence-weighted within-class entropy of the word (bits). Gate: some b with purity >= .8 and >= 8 classes of >= 3 words.
Usage (env cobint): r214_wordmap.py <wordmap.json> <out.md> [<out.json>]"""
import json
import math
import sys
from collections import Counter, defaultdict

import spacy
from spacy.tokens import Doc


def main(path, out_md, out_json=None):
    D = json.load(open(path, encoding="utf-8"))
    nlp = spacy.load("en_core_web_sm")
    occ = []
    for s, d in zip(D["sentences"], nlp.pipe((Doc(nlp.vocab, words=s) for s in D["sentences"]), batch_size=64)):
        occ += [(w, t.pos_) for w, t in zip(s, d)]
    N = len(occ)
    wc = Counter(w for w, _ in occ)
    res, L = {}, ["| nb_bits b | classes | classes with >= 3 words | largest class (words) | POS purity | within-class entropy (bits) | gate |",
                  "|---|---|---|---|---|---|---|"]
    for b, nm in D["names"].items():
        cls = defaultdict(list)
        for w in wc:
            cls[nm[w]].append(w)
        pur = ent = 0.0
        pos = defaultdict(Counter)
        for w, p in occ:
            pos[nm[w]][p] += 1
        for c, ws in cls.items():
            tot = sum(wc[w] for w in ws)
            pur += max(pos[c].values())
            ent += tot * -sum(wc[w] / tot * math.log2(wc[w] / tot) for w in ws)
        big = sum(len(ws) >= 3 for ws in cls.values())
        r = dict(classes=len(cls), big=big, largest=max(len(v) for v in cls.values()), purity=pur / N, entropy=ent / N)
        r["pass"] = r["purity"] >= .8 and big >= 8
        res[b] = r
        L.append("| %s | %d | %d | %d | %.3f | %.2f | %s |" % (b, r["classes"], big, r["largest"], r["purity"], r["entropy"], "pass" if r["pass"] else "no"))
    head = "seed %s, n=%s, %d word types, %d tokens, %s reservoir types\n\n" % (D["seed"], D["n"], len(wc), N, D["reservoir types"])
    ex = []
    best = sorted((b for b in D["names"] if res[b]["big"] >= 2), key=lambda b: -res[b]["purity"])[:3]
    for b in best:
        if True:
            cls = defaultdict(list)
            for w in wc:
                cls[D["names"][b][w]].append(w)
            top = sorted(cls.values(), key=lambda v: -sum(wc[w] for w in v))[:5]
            ex.append("level %s, five biggest classes by tokens: " % b + " | ".join(" ".join(sorted(v, key=lambda w: -wc[w])[:8]) for v in top))
    out = head + "\n".join(L) + "\n\n" + "\n".join(ex) + "\n\npassing b: %s\n" % ([b for b, r in res.items() if r["pass"]] or "none")
    open(out_md, "w", encoding="utf-8").write(out)
    if out_json:
        json.dump(res, open(out_json, "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
