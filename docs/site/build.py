"""Assemble trellis.html from skeleton.html and line ranges of the published page (orig.html)."""
import re, sys
from pathlib import Path

D = Path(__file__).resolve().parent
orig = (D / "orig.html").read_text(encoding="utf-8").split("\n")
L = lambda a, b: "\n".join(orig[a - 1:b])
sk = (D / "skeleton.html").read_text(encoding="utf-8")

assert orig[694].startswith('<script type="application/json" id="trellis-data">'), orig[694][:80]
assert orig[696].strip() == "<script>" and orig[1134].strip() == "</script>", (orig[696], orig[1134])
data = orig[694]
script = L(697, 1135)

def sub(text, old, new, n=1):
    c = text.count(old)
    if c != n:
        sys.exit(f"expected {n} of {old[:70]!r}, found {c}")
    return text.replace(old, new)

script = sub(script, ".observe(document.querySelector('.page'));", ".observe(document.getElementById('main'));")
script = sub(script, "if (!$('generate').contains(document.activeElement)", "if (!document.querySelector('[data-page=\"generate\"]').contains(document.activeElement)")
script = sub(script, "$('tleaves').innerHTML = '';", "$('tmatrix').innerHTML = '';")
script = sub(script, "r: Math.min(8, rh / 2 - 3),", "r: Math.max(1.5, Math.min(8, rh / 2 - 3)),")
script = sub(script, "renderPicker();\nrenderAll();", "window.__trellisRender = renderAll;\nrenderPicker();\nrenderAll();")

# explorers regenerated with the current recipe (logs/v3_x_submit.sh, assemble_explorers.py)
regen = D / "trellis_data.json"
if regen.exists():
    data = '<script type="application/json" id="trellis-data">' + regen.read_text(encoding="utf-8").replace("</", "<\\/") + "</script>"
    script = re.sub(r"const TAGS = \[.*?\]\];", "const TAGS = [['expansion', 'Expansion', 'var(--s1)'], ['io', 'Inside-outside generation', 'var(--s3)']];", script, count=1, flags=re.S)
    script = re.sub(r"const RUNS = \[.*?\]\];", "const RUNS = [['leaf', 'Leaf'], ['backoff0.25', 'Back-off β 0.25'], ['backoff1', 'Back-off β 1'], ['backoff1-kind', 'Back-off β 1 + kind, words at leaf'], ['backoff4-kind', 'Back-off β 4 + kind, words at leaf'], ['level3-kind', 'Three levels up + kind, words at leaf'], ['best', 'Best surroundings score']];", script, count=1, flags=re.S)
    script = re.sub(r"const CH_RUNS = \{.*?\};", "const CH_RUNS = { s13: 'Current recipe, expansion, seed 13', s17: 'Current recipe, expansion, seed 17' };", script, count=1, flags=re.S)

out = re.sub(r"@@L (\d+)-(\d+)@@", lambda m: L(int(m.group(1)), int(m.group(2))), sk)
out = sub(out, "@@DATA@@", data)
out = sub(out, "@@SCRIPT@@", script)

# measured fragments, filled in as their rounds finish
io = (D / "frag_io.html").read_text(encoding="utf-8").replace("@@IOJSON@@", (D / "../../experiments/v3/results/explorers/io_trace.json").read_text(encoding="utf-8").replace("</", "<\\/"))
out = sub(out, "@@IOEXP@@", io)
for key, name in (("@@R94@@", "frag_r94.html"), ("@@R95@@", "frag_r95.html"), ("@@CHUNKS@@", "frag_chunks.html"), ("@@UNIFIED@@", "frag_unified.html"), ("@@CHESSCHUNKS@@", "frag_chess.html"), ("@@WINDOW@@", "frag_window.html"), ("@@COPYING@@", "frag_copying.html")):
    f = D / name
    out = sub(out, key, f.read_text(encoding="utf-8") if f.exists() else '<p class="small">Running; numbers will appear here when the round finishes.</p>')


# prose around the regenerated explorers
out = sub(out, "<b>Leaf</b> is always grammatical, and novel once leaves hold several instances.", "<b>Leaf</b> is always grammatical and, under the current recipe, almost never new (0.00–0.08 at 320).")
out = sub(out, "<b>Best surroundings score</b> fails: the best-fitting concept sits high in the tree, where kinds mix.", "<b>Best surroundings score</b> fails everywhere except <code>small</code>: the best-fitting concept sits high in the tree, where kinds mix.")
out = sub(out, "<b>Back-off with the content-kind check</b> stays near leaf grammaticality and gives more new sentences on most runs.", "<b>Back-off with the content-kind check</b> stays near leaf grammaticality but gives few new sentences (0.00–0.24). <b>Three levels up with the kind check</b> is the best trade: 95–100% grammatical and 0.26–0.41 new on five of the six grammars (0.04 on <code>small</code>). <b>Back-off without the check</b> is the most novel and loses grammar on the larger grammars (59–76%, 95% on <code>term_low</code>).")
out = sub(out, "Each type keeps a weighted reservoir sample of 8 context leaves where it stood.", "Each type keeps a weighted reservoir sample of 32 context leaves where it stood.")
out = sub(out, "Its name is the bag of concepts on those leaves' current paths.", "Its name is the bag of the parent concepts of those leaves, read from their current paths.")
# principles that generation has moved past
out = sub(out, "No rules are read off the trees. Parsing and generation read the trees directly, so the grammar is whatever the two hierarchies currently hold and changes as they do.",
          "No grammar is kept. Parsing reads the trees directly, and generation recomputes a grammar over concepts from them, so the grammar is whatever the two hierarchies currently hold and changes as they do.")
out = sub(out, "<b>Generate from the trees alone</b><p>Sample a whole experience and expand it top-down. Each part draws where it stood, then a committed instance from the concept there, in proportion to the counts. Generation reads no records or pointers and never predicts a word from the words before it.</p><span class=\"ev\">Since 23 October</span>",
          "<b>Generate by inside-outside</b><p>Read a grammar over concepts from the trees and stored records, then draw each word from what the whole sentence around it allows. The previous sentence directs the draw.</p><span class=\"ev\">Since 1 November</span>")

# stale text and diagram details
out = sub(out, "Old memories stay current by relabelling in place.",
          "Old memories stay current by relabelling in place, and records that no longer sort where they sit are moved.")
out = sub(out, "cc = the chunk's own content</text>", "cc = own content path, donated;</text>")
out = sub(out, "concept path, donated</text>", "lc / rc (neighbour chunks): on</text>")
out = sub(out, "no threshold gates a chunk</text>", "only posteriors under 0.001 are dropped</text>")
out = sub(out, "every composite stored in both trees with its posterior as weight.</figcaption>",
          "every composite stored in both trees with its posterior as weight (dropped below 0.001).</figcaption>")
# heat cells: white text on dark green is fine in both themes; keep. Figure wrappers keep class fig.
assert "@@" not in out, out[out.index("@@") - 50: out.index("@@") + 50]
(D / "trellis.html").write_text(out, encoding="utf-8")

# sanity checks
pages = set(re.findall(r'data-page="([a-z0-9-]+)"', out))
hrefs = set(re.findall(r'href="#([a-z0-9-]+)"', out))
print("pages", len(pages), "missing targets", sorted(hrefs - pages))
ids = set(re.findall(r'\bid="([^"]+)"', out))
used = set(re.findall(r"\$\('([^']+)'\)", script))
print("ids referenced but absent", sorted(used - ids))
print("bytes", len(out.encode("utf-8")))
