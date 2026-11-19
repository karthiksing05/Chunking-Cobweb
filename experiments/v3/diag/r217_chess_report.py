"""Round 217: tables and examples from r217_chess_gen_<tag>.json (purity per arm, 5 examples per arm) -> r217_tables_<tag>.md.
Usage (compute node): r217_chess_report.py <tag>"""
import json
import os
import sys

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "chess_regimes")
tag = sys.argv[1]
res = json.load(open(os.path.join(OUT, f"r217_chess_gen_{tag}.json"), encoding="utf-8"))
lines = ["| arm | n | R-sem | R-legal | R-illegal | = real move |", "|---|---|---|---|---|---|"]
ex = []
for arm, l in res["arms"].items():
    n = len(l)
    f = lambda k: round(sum(1 for x in l if x["regime"] == k) / n, 3)  # noqa: E731
    lines.append(f"| {arm} | {n} | {f('R-sem')} | {f('R-legal')} | {f('R-illegal')} | "
                 f"{round(sum(1 for x in l if x['move'] == x['real']) / n, 3)} |")
    ex.append(f"\n**{arm}**\n")
    for x in l[:5]:
        b = chess.Board(x["fen"])
        mv = chess.Move.from_uci(x["move"])
        san = b.san(mv) if mv in b.legal_moves else "(illegal)"
        ex.append(f"- `{x['fen']}`  move {x['move']} {san}: {x['regime']}" + (f" (loss {x['drop']})" if "drop" in x else "")
                  + f"; real game move {x['real']}")
with open(os.path.join(OUT, f"r217_tables_{tag}.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n\n## Examples\n" + "\n".join(ex) + "\n")
print("\n".join(lines))
