import re, statistics as st
from scipy import stats
src = "/coc/flash10/ksingara3/Chunking-Cobweb/experiments/v3/results/story_gen/r224/report_body.md"
lines = open(src).read().split("\n")
blocks = {}; cur = None; order = [250, 500, 1000, 4000]; idx = -1
for ln in lines:
    m = re.match(r"- seed (\d+): (.*)", ln)
    if m and int(m.group(1)) == 13:
        idx += 1; cur = order[idx]; blocks[cur] = {}
    if m and cur is not None:
        s = int(m.group(1)); parts = m.group(2).split(" -> ")
        vals = [[float(x) for x in p.split("/")] for p in parts]
        blocks[cur][s] = vals  # [old, syn, mean][T,L,G]
def rep(seeds, label):
    print("==", label, seeds)
    for n in (250, 500, 1000):
        for ci, cname in ((1, "syntax"), (2, "meaningful")):
            T = [blocks[n][s][ci][0] for s in seeds]; L = [blocks[n][s][ci][1] for s in seeds]; G = [blocks[n][s][ci][2] for s in seeds]
            out = f"n={n} {cname}: T {st.mean(T):.3f} L {st.mean(L):.3f} G {st.mean(G):.3f}"
            for nm, X in (("L", L), ("G", G)):
                d = [a - b for a, b in zip(T, X)]; k = len(d)
                m = st.mean(d); se = st.stdev(d) / k ** .5; tc = stats.t.ppf(.975, k - 1)
                out += f" | T-{nm} {m:+.3f} [{m - tc * se:+.3f},{m + tc * se:+.3f}] {sum(x > 0 for x in d)}/{k}"
            print(out)
rep([13, 17, 19, 23], "selection")
rep([29, 31, 37, 41], "confirmation")
rep([13, 17, 19, 23, 29, 31, 37, 41], "all")
