"""Round 234: six-seed means and paired 95% t intervals of recall by sentence type and construction (r234_diag, r234_ccmdiag at 4,000)."""
import json, statistics as st
D = "experiments/v3/results/story_gen/r233/unseen_g2/diag/"
S = [13, 17, 19, 23, 29, 31]
rows = {a: {s: json.load(open(D + f"diag_s{s}_{a}.json"))["4000"] for s in S} for a in ["full", "nochunk"]}
rows["ccm"] = {s: json.load(open(D + f"ccmdiag_s{s}.json"))["4000"] for s in S}
rows["rb"] = {s: json.load(open(D + f"ccmdiag_s{s}.json"))["rb"] for s in S}
val = lambda r, c, k: r["by_sentence" if c == "s" else "by_kind"][k][0]
for c, keys in [("s", ["simple", "complex"]), ("k", ["subjNP", "subjNP+mod", "PP-with", "NP-in+with", "RC", "VP-rc", "PP-at", "VP", "NP-in", "PP-verb"])]:
    for k in keys:
        print(f"{k:12s}", " ".join(f"{a}={st.mean(val(rows[a][s], c, k) for s in S):.3f}" for a in rows),
              "n=", rows["full"][13]["by_sentence" if c == "s" else "by_kind"][k][1])
for k in ["simple", "complex"]:
    for a, b in [("full", "ccm"), ("full", "nochunk"), ("full", "rb"), ("ccm", "rb"), ("nochunk", "rb")]:
        d = [val(rows[a][s], "s", k) - val(rows[b][s], "s", k) for s in S]
        m, h = st.mean(d), 2.571 * st.stdev(d) / 6 ** .5
        print(f"{k} {a}-{b}: {m:+.3f} [{m-h:+.3f}, {m+h:+.3f}]")
