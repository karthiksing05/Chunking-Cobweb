"""Tables from experiments/v2/results/chess_hier/{diag,runs}/*.json.

    python summarize_chess_hier.py diag     # Step 1 tables (markdown to stdout)
    python summarize_chess_hier.py runs     # Step 2 tables
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "chess_hier")


def steps(d):
    if not d:
        return "–"
    s = " / ".join(f"{x['bits']:+.0f}" for x in d["steps"])
    return s + ("" if d["complete"] else " (stops)")


def diag():
    for path in sorted(glob.glob(os.path.join(RES, "diag", "*.json"))):
        r = json.load(open(path))
        N = r["N"]
        print(f"\n### {r['setting']}\n")
        print(f"read alone {r['flat bits'] / N:.3f} bits/pos; greedy chunk search {r['greedy bits'] / N:.3f} "
              f"({len(r['greedy moves'])} moves: {', '.join(r['greedy moves'][:6])})\n")
        print("| size | configuration | class | occ. | (a) global steps, flat | (a) total | (a) total, greedy start "
              "| (b) whole-config steps, flat | (b) total | (b) total, greedy start | (b) parts read/top/rows |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in r["configurations"]:
            a, ag = c.get("(a) step by step, global [flat]"), c.get("(a) step by step, global [greedy]")
            b, bg = c.get("(b) whole configuration [flat]"), c.get("(b) whole configuration [greedy]")
            parts = {k: sum(x[k] for x in b["steps"]) for k in ("read", "top", "rows")} if b["steps"] else None
            pt = f"{parts['read']:+.0f} / {parts['top']:+.0f} / {parts['rows']:+.0f}" if parts else "–"
            print(f"| {c['size']} | `{c['configuration']}` | {', '.join(c['classes']) or '–'} | {c['occurrences']} "
                  f"| {steps(a)} | {a['total']:+.0f} | {ag['total']:+.0f}{'' if ag['complete'] else '*'} "
                  f"| {steps(b)} | {b['total']:+.0f} | {bg['total']:+.0f}{'' if bg['complete'] else '*'} | {pt} |")
        print("\n(c) oracle categories (flat start): variants applied as whole configurations, then merged level by level\n")
        print("| configuration | variants | chunks of all variants | merges by level (bits) | total | either side: merge |")
        print("|---|---|---|---|---|---|")
        for c in r["configurations"]:
            v = c.get("(c) variants as one category [flat]")
            e = c.get("(c) either side as one category [flat]")
            if not v and not e:
                continue
            lv = " / ".join(f"{x['bits']:+.0f}" for x in v["merges by level"]) if v else "–"
            print(f"| `{c['configuration']}` | {len(v['variants']) if v else '–'} | "
                  f"{v['chunks of all variants (b)']:+.0f} | {lv} | {v['total']:+.0f} | "
                  f"{(str(round(e['merge'])) + ' (top ' + str(round(e['top'])) + ', rows ' + str(round(e['rows'])) + ')') if e else '–'} |"
                  if v else
                  f"| `{c['configuration']}` | – | – | – | – | {e['merge']:+.0f} (top {e['top']:+.0f}, rows {e['rows']:+.0f}) |")


def runs():
    rows = []
    for path in sorted(glob.glob(os.path.join(RES, "runs", "*.json"))):
        r = json.load(open(path))
        rows.append(r)
    print("| setting | kind | method | read alone (train) | search (train) | TRELLIS total | held read | held grammar "
          "| held search | gain held (grammar) | chunk labels | merges | levels | pass every check | gen chunks in held-out | seconds |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        t, h = r["train bits/pos"], r["held-out bits/pos"]
        lab = r.get("search chunk labels", r.get("chunk types (grammar)"))
        g = r["generation"]
        print(f"| e{r['elo']} p{r['ply']} N{r['N']} a{r['alpha']:g} | {r['kind']} | {r['method']} | {t['read alone']:.3f} "
              f"| {t['after the search']:.3f} | {t['TRELLIS v2 total']:.3f} | {h['read alone']:.3f} "
              f"| {h['TRELLIS v2 grammar']:.3f} | {h['search code']:.3f} | {r['chunk gain, held out (read alone - grammar)']:+.3f} "
              f"| {lab} | {r.get('merges', 0)} | {r['top-level chunk levels (depth: count)']} "
              f"| {g['checks'].get('passes every check', 0):.3f} | {g['generated chunks found in a held-out position']:.3f} "
              f"| {r['seconds']:.0f} |")


if __name__ == "__main__":
    {"diag": diag, "runs": runs}[sys.argv[1]]()
