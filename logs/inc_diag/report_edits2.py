import sys

path = sys.argv[1]
s = open(path, encoding="utf-8").read()


def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:90])
    s = s.replace(old, new)


def rep_between(start, end, new):
    global s
    assert s.count(start) == 1, start[:90]
    i = s.index(start)
    j = s.index(end, i) + len(end)
    s = s[:i] + new + s[j:]


# Mechanisms: the joint split is now shared code.
rep_between("""<p><b>Measured, but not yet in the shared code.</b>""", """and set aside.</p>""",
            """<p><b>The joint split, a second coordinated move.</b> Since 13 October the shared code also refines the grammar chosen at the end of the night with the <em>joint split</em> (<code>src/trellis2/splits.py</code>): it proposes splitting one category in two by the company its elements keep, carries the split through the categories that combine with it, scores the whole coordinated split once, and keeps it only if the full code shrinks. It never changes the earlier rounds. On 18 runs across the synthetic grammars, the paper's corpora, simple English and characters it never lengthened the code; it is being made the default. Heads in both hierarchies (<code>experiments/v2/heads.py</code>) were built and tested exactly, and set aside.</p>""")

# Iterations: new rows.
rep("""<td class="what">Chunks pay and grow with data. A read in context halves the gap to a transformer but leaves almost no chunk.</td></tr>
  </table></div>""",
    """<td class="what">Chunks pay and grow with data. A read in context halves the gap to a transformer but leaves almost no chunk.</td></tr>
    <tr><td class="when">12–13 Oct</td><td>Chess: attack and defence as relations for chunks; a context made of the chunks already placed.</td><td class="what">Neither adds a chunk: tactical relations lump unrelated placements together, and chunk-level context finds the chunks piece-level context finds.</td></tr>
    <tr><td class="when">13 Oct</td><td>The joint split moved into the shared code as the night's last step.</td><td class="what">Never longer on 18 runs; grammatical samples on <em>agree</em> 51% → 75% and 61% → 69% with no knowledge of number.</td></tr>
    <tr><td class="when">Next</td><td>Closing the loop: the categories and composites a night accepts become where the next night starts.</td><td class="what">Running. The test is whether nights refine what earlier nights found instead of repeating the search.</td></tr>
  </table></div>""")

# Tried and set aside: chess additions.
rep("""<tr><td>The chess read's own answers in context, as the default</td><td class="what">The best memory of a board, but its chunks stop carrying structure.</td></tr>""",
    """<tr><td>The chess read's own answers in context, as the default</td><td class="what">The best memory of a board, but its chunks stop carrying structure.</td></tr>
    <tr><td>Attack and defence as relations for chess chunks</td><td class="what">Reach the search and are scored exactly, but none pays: the best costs 115–170 bits where a specific spatial chunk saves 1,451.</td></tr>
    <tr><td>A chess context made of the chunks already placed</td><td class="what">Finds the same chunks as a context of single pieces, with the same code.</td></tr>""")

# Agreement: the shared joint split.
rep("""so the joint split is not accepted.</p>""",
    """so the joint split is not accepted.</p>
    <p><b>In the shared code.</b> Moved into the night as its last step and run on top of the joint chunk move, the joint split raises <em>agree</em> from 51% to 75% grammatical (seed 13) and from 61% to 69% (seed 17), and <em>relative</em> from 30% to 34% (seed 17), never lengthening the code. The gain on <em>agree</em> is smaller than the experimental version's 88–89%; whether a search cap or the different starting grammar is the cause is being checked.</p>""")

# Characters: the joint split there.
rep("""its code is almost entirely about position.</p>""",
    """its code is almost entirely about position.</p>
    <p>The joint split accepts three category splits on characters learned from sequences (62,414 → 62,282 bits; well formed 92.9% → 93.9%). Whether splits make the chunk types more specific, the categories' remaining weakness, is being measured.</p>""")

# Chess: results of the two chunk-carrying tests.
rep_between("""<p>With labels in context, the chunks are fianchettos""", """</p>""",
            """<p>With labels in context, the chunks are fianchettos for both sides and pawn chains up to three levels deep (<code>[[bP NW1 [bP E1 bB]] WNW bP]</code>), and random positions still get none. Since chess is about structure carried by chunks, the read in context stays an option, and two ways to let chunks carry more were tested.</p>
    <p><b>Attack and defence as relations.</b> The domain supplies "attacks" and "defends" beside direction and distance, as characters supply their operators. The search scores them exactly (121 tactical candidate types at the start), and none pays: the best "defends" chunk costs 170 bits and the best "attacks" chunk 115, while the specific pawn pair <code>[wP NW1 wP]</code> saves 1,451. A tactical relation lumps together many unrelated placements (a pawn defends a pawn along either diagonal, anywhere), so the chunk is hard to predict where it appears and still has to say which geometry each instance has.</p>
    <p><b>A context made of chunks.</b> Coding each label given the categories of the elements already placed nearby, with a chunk counted as one element, finds the same chunks as a context of single pieces (search code 50.47 against 50.46 bits on 1800+ positions). In chess the useful structure lies in specific spatial configurations on shared openings, and neither change adds to it. Whether chunks earn their place under a fixed storage budget, the question Chase and Simon actually asked, is the next test.</p>""")

# Where it stands: next iteration and beyond day and night.
rep("""a chess night with labels in context takes 3–13 hours.</li>
    </ul>""",
    """a chess night with labels in context takes 3–13 hours.</li>
    </ul>
    <h3>Next: closing the loop</h3>
    <p>Three open problems share a root: what the code discovers at night is used once and lost. The joint split finds grammatical number after the night, but the representation hierarchy that forms the next night's categories never sees it; each night's consolidation starts from blank chunk context, and its search largely from word classes. Long sentences (even a grammar read off the gold trees generates them 5% grammatically), agreement inside relative clauses, and the generic categories of characters all need categories to refine across nights.</p>
    <p>The next iteration therefore carries a night's verdicts into the next one: its final, split categories become the first round's labels and a starting partition of the search, and its accepted composites are proposed first and scored again. Every carried-over change is still kept only if the full code shrinks. This extends <em>coherence comes from what the descriptions can see</em>: the grammar's own findings become something the descriptions see. It is being built and tested on several nights over the same data, on nightly learning in growing batches, and on characters.</p>
    <h3>Beyond the day-and-night cycle</h3>
    <p>The night exists because some moves are only worth making when scored over all the experiences at once: a joined forest, a coordinated split, a composite that pays only as a whole. It need not be a separate, full rebuild. These are proposals, none tested yet:</p>
    <ul>
      <li><b>Keep the hierarchies.</b> Cobweb is already incremental, and the grammar's tables are counts that add up. Each day's experiences can update the hierarchies and the grammar as they arrive, instead of being replayed into fresh trees each night; rebuilding is a large share of a night's cost.</li>
      <li><b>Consolidate locally.</b> Re-search only the categories and chunk types new experiences touch, scoring each move by the count rows it changes, as the search already does for its own moves.</li>
      <li><b>Sleep when the code says so.</b> Consolidate when recent experiences cost noticeably more bits under the current grammar than under a quick local refit, so consolidation runs in small, frequent pieces where surprise accumulates.</li>
      <li><b>Closing the loop comes first.</b> Short, frequent consolidations only work if each one starts from what the last one found, which is exactly what carrying a night's verdicts forward provides.</li>
    </ul>""")

rep("""Last updated 12 October 2026.""", """Last updated 13 October 2026.""")

open(path, "w", encoding="utf-8").write(s)
print("ok", len(s))
