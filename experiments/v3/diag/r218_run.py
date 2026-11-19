"""Round 218: one (seed, n, group) job. group "eval": train the full-IO learner (= r215's) once and parse every control sentence under every
eval-only cell (windowing at evaluation only); group "learn": for each win_learn cell train a learner whose learn() uses the windowed parse,
then parse with the same knobs. Per sentence: chosen tree, logZ, logw, stats (factors, triples, windows, passes, seconds). Per cell: chunk reuse.
Usage (repo root, compute node, env trellis):  r218_run.py <seed> <n> <eval|learn> [cell,cell,...]
"""
import json
import os
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3/diag")
import r215_common as C  # noqa: E402
from r218_cells import CELLS  # noqa: E402
from trellis2.trellis3_r218 import Trellis3R218  # noqa: E402

OUT = os.path.join(C.OUT, "r218_raw")


def lines_of(seed):
    S = C.load_sets(seed)
    lines = []
    for ss in C.all_sentences(S).values():
        for l in ss:
            if l not in lines:
                lines.append(l)
    xl = [l for l in C.load_sets(13)["C1"] if l not in lines]
    return lines, xl


def train(seed, n, kw):
    tr, _, _, vocab = C.stream(seed, n)
    L = Trellis3R218(seed=seed, **dict(C.CFG, window=C.WINDOW[n]), **kw)
    t0 = time.time()
    for s, fol in tr:
        L.learn(s, follows=fol)
    L._quiet = True
    return L, time.time() - t0


def stored_chunks(L):
    """token tuples of stored composites: any weight, and weight >= .5"""
    anyw, heavy = set(), set()
    for e in L.records:
        if e.token is None and e.leaf is not None:
            t = L._ident(e)
            anyw.add(t)
            if e.weight >= 0.5:
                heavy.add(t)
    return anyw, heavy


def run_cell(L, name, cell, lines, xl):
    L.win_W = cell["kw"].get("win_W")
    L.win_stride = cell["kw"].get("win_stride")
    L.io_depth_D = cell["kw"].get("io_depth_D")
    L.win_learn = cell["kw"].get("win_learn", False)
    L.io_depth_mode = cell["kw"].get("io_depth_mode", "freeze")
    L.search, L.beam = cell.get("search", "chart"), cell.get("beam", 8)
    anyw, heavy = stored_chunks(L)
    res = {}
    t0 = time.time()
    for l in lines + xl:
        w = l.split()
        r = L.parse(w)
        st = r.get("stats") or {}
        if "logw" not in r:
            r["logw"] = sum(r["comps"][c].gain for c in r["chosen"]) * 0.6931471805599453 / L.tau
        res[l] = {"len": len(w), "chosen": [list(c) for c in r["chosen"]], "logZ": r["logZ"] if "logZ" in r else None, "logw": r["logw"],
                  "factors": st.get("factors"), "triples": st.get("triples"), "windows": st.get("windows"), "passes": st.get("passes"),
                  "seconds": st.get("seconds"), "forced": st.get("forced")}
        if name in ("greedy",):                       # beam parse: cost = composites it evaluated; logZ is the beam's
            res[l]["factors"] = len(r["comps"])
        ch = [(i, j) for i, k, j in r["chosen"] if (i, j) != (0, len(w))]
        res[l]["chunks"] = len(ch)
        res[l]["reuse_any"] = sum(tuple(w[i:j]) in anyw for i, j in ch)
        res[l]["reuse_w50"] = sum(tuple(w[i:j]) in heavy for i, j in ch)
    L.win_W = L.win_stride = L.io_depth_D = None
    L.win_learn = False
    L.search = "chart"
    L.io_depth_mode = "freeze"
    return res, time.time() - t0


def main(seed, n, group, only):
    lines, xl = lines_of(seed)
    os.makedirs(OUT, exist_ok=True)
    names = [c for c, v in CELLS.items() if v["group"] == group and (not only or c in only)]
    L = None
    for name in names:
        cell = CELLS[name]
        if group == "eval":
            if L is None:
                L, ls = train(seed, n, {})
                print(json.dumps({"seed": seed, "n": n, "learn s": round(ls), "records": len(L.records)}), flush=True)
            LL, learn_s = L, None
        else:
            LL, learn_s = train(seed, n, cell["kw"])
            print(json.dumps({"cell": name, "learn s": round(learn_s), "records": len(LL.records)}), flush=True)
        res, ps = run_cell(LL, name, cell, lines, xl)
        if name == "full":
            old = json.load(open(os.path.join(C.RAW, "trellis_s%d_n%d.json" % (seed, n)), encoding="utf-8"))["res"]
            print(json.dumps({"r215 logZ mismatches": sum(abs(res[l]["logZ"] - old[l]["logZ"]) > 0 for l in old if l in res),
                              "compared": sum(l in res for l in old)}), flush=True)
        print(json.dumps({"cell": name, "parse s": round(ps), "sentences": len(res)}), flush=True)
        json.dump({"seed": seed, "n": n, "cell": name, "kw": cell["kw"], "search": cell.get("search"), "learn_s": learn_s,
                   "records": len(LL.records), "lines": lines, "xlines": xl, "res": res},
                  open(os.path.join(OUT, "%s_s%d_n%d.json" % (name, seed, n)), "w", encoding="utf-8"))
        if group == "learn":
            del LL


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], set(sys.argv[4].split(",")) if len(sys.argv) > 4 else set())
