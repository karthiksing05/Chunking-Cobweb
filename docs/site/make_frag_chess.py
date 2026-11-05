"""Write docs/site/frag_chess.html: chess chunks as inline SVG boards, read from the memory behind
the current coding results (experiments/v3/results/chess_inspect/cur_{rel,abs}_s13.json)."""
import json
from collections import Counter
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "experiments/v3/results/chess_inspect"
rel = json.load(open(R / "cur_rel_s13.json", encoding="utf-8"))
abs_ = json.load(open(R / "cur_abs_s13.json", encoding="utf-8"))
GLYPH = {"K": "♚", "Q": "♛", "R": "♜", "B": "♝", "N": "♞", "P": "♟"}
NAME = {"K": "king", "Q": "queen", "R": "rook", "B": "bishop", "N": "knight", "P": "pawn"}
S = 20


def xy(sq):
    return (ord(sq[0]) - 97) * S, (8 - int(sq[1])) * S


def label(pieces):
    return ", ".join(f"{'white' if t[0] == 'w' else 'black'} {NAME[t[1]]} {sq}" for sq, t in sorted(pieces))


def board(pieces, rings=None, cls=""):
    """pieces: [(square, token)]; rings: {square: chunk index} outlines squares and numbers them."""
    out = [f'<svg class="cb {cls}" viewBox="-1 -1 {8 * S + 2} {8 * S + 2}" role="img" aria-label="{escape(label(pieces))}"><title>{escape(label(pieces))}</title>']
    for f in range(8):
        for r in range(8):
            out.append(f'<rect x="{f * S}" y="{r * S}" width="{S}" height="{S}" class="{"cbd" if (f + r) % 2 else "cbl"}"/>')
    for sq, t in pieces:
        x, y = xy(sq)
        if not rings:
            out.append(f'<rect x="{x}" y="{y}" width="{S}" height="{S}" class="cbon"/>')
        if rings and sq in rings:
            k = rings[sq]
            out.append(f'<rect x="{x + 1}" y="{y + 1}" width="{S - 2}" height="{S - 2}" rx="2" fill="none" stroke="var(--cb{k + 1})" stroke-width="2.2"/>')
        out.append(f'<text x="{x + S / 2}" y="{y + 16}" class="cbp {"cbw" if t[0] == "w" else "cbk"}">{GLYPH[t[1]]}︎</text>')
        if rings and sq in rings:
            out.append(f'<rect x="{x + 0.8}" y="{y + 0.8}" width="7.4" height="7.4" rx="1.5" fill="var(--cb{rings[sq] + 1})"/><text x="{x + 4.5}" y="{y + 6.7}" class="cbn">{rings[sq] + 1}</text>')
    out.append("</svg>")
    return "".join(out)


def merged(c):
    """A concept's instances with those differing only in joining relation counted together."""
    n = Counter()
    for pcs, k in c["instances"]:
        n[tuple(map(tuple, pcs))] += k
    return n.most_common()


def cell(pieces, cap):
    return f'<figure class="cbc">{board(pieces)}<figcaption>{cap}</figcaption></figure>'


def names(pieces):
    return " ".join(sq for sq, _ in pieces)


inv = rel["inventory"]
inva = abs_["inventory"]
# concepts shown for their transfer: hand-picked by rank from the lists, instances unedited
PICKS = [("2", 0, "Black pawns on a falling diagonal: the same two-piece shape anywhere on the board"),
         ("3", 0, "Three-pawn black chains, in the same direction on different files"),
         ("3", 2, "The fianchetto pawn wedge, on both wings"),
         ("4", 2, "The castled king's shelter, on both wings")]
h = ['<p>Read from the memory behind the coding results on this page: 4,000 positions learned, seed 13, chunks described by offsets, up to four pieces. '
     'No labels were used. A concept is the parent of a chunk\'s content leaf; n counts the chunks stored under it. Which concepts appear in the first group is our choice; their instances are not edited. '
     'On the small boards a chunk is drawn alone, on highlighted squares, with the rest of the position left out.</p>',
     '<h3>One concept, different squares</h3>',
     '<p>Each row is one content concept with its most frequent placements. These chunks were never told that a shape on another file is the same shape.</p>',
     '<div class="cb-rows">']
for size, rank, what in PICKS:
    c = inv[size]["top"][rank]
    h.append(f'<div class="cb-row"><p class="cb-what"><b>{escape(what)}.</b> n = {c["count"]:,}.</p><div class="cb-grid">')
    for pcs, k in merged(c)[:5]:
        h.append(cell(pcs, f'{names(pcs)}<br>seen {k}'))
    h.append("</div></div>")
h.append("</div>")
top2, top2a = inv["2"], inva["2"]
one = sum(r["distinct"] == 1 for r in inva["2"]["top"])
h.append(f'<p>Adding squares to the description loses this. Two-piece chunks then fall into {top2a["concepts"]} concepts against {top2["concepts"]} by offsets, and {one} of the 12 largest hold a single placement: '
         f'{names(inva["2"]["top"][0]["example"])} alone has {inva["2"]["top"][0]["count"]} chunks, where the offset concept that contains it holds {inv["2"]["top"][0]["count"]:,} across {inv["2"]["top"][0]["distinct"]} placements and joining relations. '
         'This is the difference behind the transfer test on this page.</p>')
h.append('<h3>The largest concepts at each size</h3><div class="cb-rows">')
for size in ("2", "3", "4"):
    v = inv[size]
    h.append(f'<div class="cb-row"><p class="cb-what"><b>{size} pieces.</b> {v["committed"]:,} chunks in {v["concepts"]:,} concepts; the five largest, each by its commonest placement.</p><div class="cb-grid">')
    for c in v["top"][:5]:
        pcs, _ = merged(c)[0]
        h.append(cell(pcs, f'{names(pcs)}<br>n = {c["count"]:,}'))
    h.append("</div></div>")
h.append("</div>")
# the first held-out position, not chosen for looks
p = rel["nested"][0]
chunks = [t for t in p["top"] if "size" in t]
rings = {sq: i for i, t in enumerate(chunks) for sq, _ in t["pieces"]}
fen = [(sq, t) for sq, t in p["fen"].items()]
h.append('<h3>One held-out position, read into chunks</h3>')
h.append(f'<p>The first of the control test positions, as the parser analyses it: {len(chunks)} chunks hold {len(rings)} of its {len(fen)} pieces. Numbers on the squares match the list. This is the analysis the parser chose, not a recall trial.</p>')
h.append(f'<div class="cb-pos">{board(fen, rings, "cb-big")}<ol class="cb-list">')
for i, t in enumerate(chunks):
    h.append(f'<li><span class="cb-sw" style="background:var(--cb{i + 1})"></span>{escape(label(t["pieces"]))}</li>')
h.append("</ol></div>")
h.append('<p class="small">Chunks are capped at four pieces here, so whole formations appear only in parts. Long-range and mixed-colour chunks also occur, such as white pawns a2 and b2 with a knight on c3 and a black rook on b8, and we have not measured how many are meaningful. Figures from earlier configurations, with chunks of up to eight pieces, are in the collapsed studies at the end of this page.</p>')
css = """<style>
:root { --cbl: #efe6d6; --cbd: #c9b596; --cbw: #fbfaf6; --cbk: #1d1b18; --cbh: #eab45a;
  --cb1: #2a6fc0; --cb2: #d9622b; --cb3: #1a9068; --cb4: #8a4fc7; --cb5: #b5304f; --cb6: #8a6d00; --cb7: #0e7f9e; --cb8: #55606b; }
.cb { display: block; width: 100%; height: auto; max-width: 170px; border-radius: 3px; }
.cb-big { max-width: 380px; flex: 1 1 280px; }
.cbl { fill: var(--cbl); } .cbd { fill: var(--cbd); } .cbon { fill: var(--cbh); }
.cbp { font: 18px "DejaVu Sans", "Segoe UI Symbol", "Apple Symbols", "Noto Sans Symbols 2", sans-serif; text-anchor: middle; paint-order: stroke; stroke-width: 1.5; }
.cbw { fill: var(--cbw); stroke: var(--cbk); } .cbk { fill: var(--cbk); stroke: var(--cbk); stroke-width: 0; }
.cbn { font: 700 6px var(--font-mono, monospace); fill: #fff; text-anchor: middle; }
.cb-rows { display: grid; gap: 20px; margin: 10px 0 18px; }
.cb-what { margin: 0 0 8px; }
.cb-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(132px, 1fr)); gap: 12px; }
.cbc { margin: 0; min-width: 0; }
.cbc figcaption { font-size: 12.5px; line-height: 1.35; color: var(--muted); margin-top: 4px; font-variant-numeric: tabular-nums; }
.cb-pos { display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; margin: 10px 0 14px; }
.cb-list { flex: 1 1 260px; min-width: 0; margin: 0; padding-left: 24px; font-size: 14px; line-height: 1.5; }
.cb-list li { margin-bottom: 5px; }
.cb-sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
</style>"""
(ROOT / "docs/site/frag_chess.html").write_text(css + "\n" + "\n".join(h) + "\n", encoding="utf-8")
print("frag_chess bytes", (ROOT / "docs/site/frag_chess.html").stat().st_size, "chunks", len(chunks), "pieces", len(rings), "of", len(fen))
