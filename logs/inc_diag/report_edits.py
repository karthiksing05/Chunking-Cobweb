import sys

path = sys.argv[1]
s = open(path, encoding="utf-8").read()


def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:80])
    s = s.replace(old, new)


def rep_between(start, end, new, keep_end=True):
    """Replace from `start` (inclusive) to the first `end` after it."""
    global s
    assert s.count(start) == 1, start[:80]
    i = s.index(start)
    j = s.index(end, i)
    s = s[:i] + new + (s[j:] if keep_end else s[j + len(end):])


# --------------------------------------------------------------- CSS
rep(""".dek { font-family: var(--font-display); font-size: clamp(18px, 2.4vw, 21px); color: var(--ink2); margin-top: 12px; line-height: 1.4; }
</style>""", """.dek { font-family: var(--font-display); font-size: clamp(18px, 2.4vw, 21px); color: var(--ink2); margin-top: 12px; line-height: 1.4; }
.toc { display: flex; flex-wrap: wrap; gap: 6px 20px; margin-top: 28px; font-size: 15px; }
.toc a { text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.principles { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 22px 34px; margin-top: 24px; }
.principles > div { min-width: 0; border-top: 1px solid var(--rule); padding-top: 12px; }
.principles b { display: block; font-family: var(--font-display); font-size: 18px; font-weight: 650; line-height: 1.3; }
.principles p { margin-top: 6px; font-size: 16px; color: var(--ink2); }
.principles .ev { display: block; margin-top: 8px; font-family: var(--font-mono); font-size: 11.5px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); }
ol.steps { padding-left: 1.5em; margin: 12px 0 0; }
ol.steps li { margin-top: 8px; }
td.when { font-family: var(--font-mono); font-size: 12.5px; color: var(--muted); white-space: nowrap; }
.tagline { font-family: var(--font-mono); font-size: 11.5px; letter-spacing: 0.05em; text-transform: uppercase; font-weight: 600; color: var(--chunk); }
</style>""")

# --------------------------------------------------------------- header
rep("""This report shows what the same machinery learns, parses and generates on three kinds of data it was not designed around, and how its generations became coherent in each: by letting the two hierarchies see what they had been missing.</p>""",
    """This report gathers the principles the system holds to, its mechanisms as they stand, and its iterations from v1 to now, with what each one showed and what was set aside. It then shows what the same machinery learns, parses and generates on simple English, on synthetic grammars with agreement and long sentences, on Chinese characters and on chess positions.</p>""")

rep("""Learned from the sequences alone, 53% are well formed.</span></div>""",
    """Learned from the sequences alone, with the joint chunk move, 88–93% are well formed and the structures come close to the gold ones.</span></div>""")

PRINCIPLES = """<div><b>Chess</b><span>Pieces joined along a queen's rays and a knight's jumps. A read that counts what is already on the board makes every generated position pass every check. Coding each chunk's label given what was read nearby makes chunks pay and grow with data (fianchettos, pawn chains), though a transformer still stores positions about 15 bits cheaper.</span></div>
  <div><b>Agreement and long sentences</b><span>On new synthetic grammars, a joint split move finds singular and plural from sentences alone: 88–89% of samples are grammatical (46–50% before). Sentences of 10–30 words form whole trees when learned night by night, but few samples are grammatical yet.</span></div>
</div>

<nav class="toc" aria-label="Contents">
  <a href="#principles">Principles</a>
  <a href="#mechanisms">Mechanisms</a>
  <a href="#v1-v2">From v1 to v2</a>
  <a href="#iterations">Iterations</a>
  <a href="#english">Simple English</a>
  <a href="#complex">Agreement and long sentences</a>
  <a href="#characters">Characters</a>
  <a href="#chess">Chess</a>
  <a href="#next">Where it stands</a>
</nav>

<section id="principles">
  <div class="tag">Principles</div>
  <h2>What the system holds to</h2>
  <div class="col">
    <p>These are the commitments kept since v1 and the lessons the iterations added. Each is stated without reference to any domain: none of them mentions words, components or pieces.</p>
  </div>
  <div class="principles">
    <div><b>Two descriptions of every element</b><p>How it behaves (its representation) and what it is made of (its composition), each organized by Cobweb into a hierarchy that holds primitives and composites alike. Categories and chunk types are cuts through these two trees, and there are no others.</p><span class="ev">From the paper's postulates</span></div>
    <div><b>One model does everything</b><p>One factored probabilistic grammar, read off the two cuts, parses, generates, completes prompts and prices the data. Generation samples the same distribution the parser and the code use, with no pools, filters or fallbacks.</p><span class="ev">v2.0</span></div>
    <div><b>Description length decides</b><p>Every choice v1 left to a threshold is made by one code for the whole corpus: which concepts are categories, which chunks exist, how much of what was just read a choice may see, which analysis is right, and which of a night's candidates is kept.</p><span class="ev">v2.0–v2.1</span></div>
    <div><b>A domain says only what an experience is</b><p>Its elements and the relations that join two parts, its context window, and its reading order. Everything else is shared code, and a test checks in each domain that the grammar draws every experience as often as its code says it should.</p><span class="ev">v2.4</span></div>
    <div><b>Coherence comes from what the descriptions can see</b><p>When generations went wrong as a whole, the cause was something neither hierarchy nor the top-level code could see. The fix was to let them see it and let description length decide what to keep.</p><span class="ev">v2.5</span></div>
    <div><b>Every move is scored exactly</b><p>Search moves, splits and joins are scored from the count tables they change and accepted only if the full code shrinks. Tests check each score against a recomputation from scratch (131 tests).</p><span class="ev">v2.1, kept since</span></div>
    <div><b>When no single step pays, move several at once</b><p>In every domain some structure pays as a whole but costs bits one step at a time: grammatical number, whole components of characters, larger chess configurations. A coordinated move, proposed whole and scored once, gets past it.</p><span class="ev">Learned 9–11 October 2026</span></div>
    <div><b>Choose at the end of the night</b><p>Candidate grammars are compared by the code each reaches after the whole night. Compared partway, the joint chunk move lost on characters; compared at the end, it wins by 1,500–2,300 bits.</p><span class="ev">Learned 11 October 2026</span></div>
    <div><b>Chunks carry the structure</b><p>A read that sees enough can explain a domain without chunks. In chess, reading each square in the light of its neighbours shortens the code by 6–11 bits per position and leaves almost no chunk. Such a read stays an option. The account of a domain that counts is the one its chunks carry.</p><span class="ev">Decided 12 October 2026</span></div>
    <div><b>Mechanisms are domain-general</b><p>Knowledge of a domain (grammatical number, the IDS components, chess openings) enters only as an oracle that diagnoses and evaluates, never as a mechanism. Every change is checked on the other domains before it is adopted.</p><span class="ev">Decided 9 October 2026</span></div>
  </div>
</section>

<section id="mechanisms">
  <div class="tag">Mechanisms</div>"""

rep_between("""<div><b>Chess</b><span>""", """<div class="tag">The framework</div>""", PRINCIPLES, keep_end=False)

# --------------------------------------------------------------- domain table
rep("""<td>square by square, counting each kind of piece</td></tr>""",
    """<td>square by square, counting each kind of piece; each element's label given what was read nearby</td></tr>""")
rep("""<tr><td>structure search</td><td>chunk and merge from word classes</td><td>none needed: the structure is given</td><td>chunk moves under the read's code</td></tr>""",
    """<tr><td>structure search</td><td>chunk, merge and joint chunk moves from word classes</td><td>none when the structure is given; from sequences, the sentence search</td><td>chunk and merge moves under the read's code</td></tr>""")

NIGHT = """<figcaption>The framework at a glance. Blue: how elements behave. Orange: what they are made of.</figcaption>
  </figure>
  <h3>The night, step by step</h3>
  <div class="col">
    <p>By day each experience is parsed with the current grammar and stored with its analysis. At night the stored analyses are searched, consolidated and parsed again. This is the night as the shared code runs it now, for any domain that learns from experiences alone:</p>
    <ol class="steps">
      <li><b>Starting points.</b> Word classes from the merge path of Brown clustering, read as description length; after the first night, also the stored analyses.</li>
      <li><b>Structure search.</b> From each start, chunk and merge moves under the plain code, a beam of four and Stolcke's lookahead. Each start is searched twice: without and with the <em>joint chunk move</em>, which applies a whole multi-level composite to all its occurrences at once.</li>
      <li><b>Consolidation.</b> The three best results of each kind are replayed into fresh hierarchies, each also with its forests joined into wholes. Cuts and model merging give each one's grammar.</li>
      <li><b>Re-analysis.</b> Viterbi parses under the new grammar (hard EM), kept while the full code shrinks.</li>
      <li><b>Re-analysis without the read's context</b>, where the categories must carry the structure (up to ten steps).</li>
      <li><b>Sampled re-analysis.</b> Analyses drawn from the posterior at falling temperatures (stochastic EM).</li>
      <li><b>The choice.</b> The best candidate of each kind goes through steps 4–6, and the full code at the end of the night picks one.</li>
      <li><b>Rewriting.</b> The stored analyses are rewritten in the new grammar's categories, ready for the next day.</li>
    </ol>
    <p><b>Measured, but not yet in the shared code.</b> A final-round <em>joint split</em> (<code>experiments/v2/splits.py</code>) refines the night's last grammar by splitting several categories at once. It finds grammatical number from sentences alone and is never worse than baseline on the paper's corpora, but it has not been moved into <code>src/</code>. Heads in both hierarchies (<code>experiments/v2/heads.py</code>) were built and tested exactly, and set aside.</p>
  </div>"""
rep("""<figcaption>The framework at a glance. Blue: how elements behave. Orange: what they are made of.</figcaption>
  </figure>""", NIGHT)

# --------------------------------------------------------------- iterations
ITER = """That search is the one new mechanism.</p>
  </div>
</section>

<section id="iterations">
  <div class="tag">Iterations</div>
  <h2>From v1 to now, and what each step showed</h2>
  <div class="col">
    <p>Each stage below changed one thing and was kept because the code got shorter or the generations more coherent, usually both. The stages from 8 October on were run on the cluster with every earlier experiment rerun under the full night; those reruns reproduced the earlier results within a few points.</p>
  </div>
  <div class="tablewrap"><table class="wide">
    <tr><th>Stage</th><th>What changed</th><th>What it showed</th></tr>
    <tr><td class="when">v1 · the paper</td><td>Greedy bottom-up parsing with a recognition threshold; generation from a recombination pool; learning from given parse trees; six synthetic grammars.</td><td class="what">The postulates work as a system. Thresholds decide everything, and one-off chunks accumulate because nothing weighs a new symbol against the whole grammar.</td></tr>
    <tr><td class="when">v2.0</td><td>Representation and composition instances in two hierarchies; cuts by description length and model merging; one grammar; inside–outside parsing and minimum-risk trees; generation from the grammar.</td><td class="what">One normalized model parses, generates and prices the paper's corpora.</td></tr>
    <tr><td class="when">v2.1</td><td>Learning from sentences alone: a message length as the objective, partial analyses (forests), word classes, and a beam search over chunk and merge moves scored exactly.</td><td class="what">Complete analyses on the paper's corpora, with a code within 0.2% of the gold-tree grammar's.</td></tr>
    <tr><td class="when">v2.2</td><td>Learning by day and by night: perceive with the current grammar, consolidate at night from the stored analyses or from a restart.</td><td class="what">After about 80 sentences the stored analyses win most nights.</td></tr>
    <tr><td class="when">v2.3</td><td>Beyond the paper's corpora: Penn Treebank tags and Chinese characters; a compiled Cobweb.</td><td class="what">Real text prefers forests of base phrases; long nights became affordable.</td></tr>
    <tr><td class="when">v2.4</td><td>Each domain brings its context window and typed relations: the chess star with direction and distance, IDS operators as relations, simple English from TinyStories.</td><td class="what">The same code runs on three kinds of data.</td></tr>
    <tr><td class="when">v2.5</td><td>Coherence from what the descriptions can see: whole sentences and pieces coded apart, a part's slot in its description, a board read that counts, each rule choice given the words before it.</td><td class="what">English samples derived whole 50% → 90% real; components in attested slots 52% → 84%; legal boards 4% → 100%.</td></tr>
    <tr><td class="when">v2.6 · 7–8 Oct</td><td>The night joins forests, re-analyses without the read's context and by sampling; prompts are completed from a scaffolded parse.</td><td class="what">Every English sentence one tree; characters from sequences 16% → 53% well formed.</td></tr>
    <tr><td class="when">8 Oct</td><td>Every experiment rerun on the cluster; storage tests against compressors and neural codes in chess and characters.</td><td class="what">TRELLIS has the shortest two-part code of the methods tried for characters; a transformer stores chess positions about 15 bits cheaper.</td></tr>
    <tr><td class="when">9 Oct</td><td>Synthetic grammars with agreement, centre embedding and long sentences; heads in both hierarchies; split moves; the joint split.</td><td class="what">Grammatical number is learned from sentences alone once a split moves several categories at once (88–89% grammatical samples).</td></tr>
    <tr><td class="when">10 Oct</td><td>Long sentences: do embedded clauses share the sentence's category; a shallower spine; learning night by night in growing batches.</td><td class="what">Whole trees and a shorter code on sentences of 10–30 words, but only 3–7% of samples are grammatical.</td></tr>
    <tr><td class="when">11 Oct</td><td>The joint chunk move in the shared search, with the night's choice made by the full code at its end (now the default).</td><td class="what">Characters from sequences reach near-gold structure: 88–93% well formed, bracket omission 6–12%.</td></tr>
    <tr><td class="when">11–12 Oct</td><td>Chess: merge moves; each element's label coded given what was read nearby (now the default); the read's own answers coded in context (an option).</td><td class="what">Chunks pay and grow with data. A read in context halves the gap to a transformer but leaves almost no chunk.</td></tr>
  </table></div>
  <h3>Tried and set aside</h3>
  <div class="tablewrap"><table class="wide">
    <tr><th>What was tried</th><th>Why it was set aside</th></tr>
    <tr><td>An attach move in the search</td><td class="what">Longer code (MED 6,539 → 8,469 bits).</td></tr>
    <tr><td>A Markov code over a forest's pieces</td><td class="what">Shorter code, but no more coherent samples.</td></tr>
    <tr><td>The sentence as the parent of a forest's pieces</td><td class="what">Joins most forests, but codes longer and generates less coherently.</td></tr>
    <tr><td>Characters described by their left heads</td><td class="what">Codes longer and recovers less structure.</td></tr>
    <tr><td>Cobweb's information-theoretic tree in place of category utility</td><td class="what">Trees kept, but categories generate worse and the code is 2–7% longer.</td></tr>
    <tr><td>Heads in the composition hierarchy (a dependent drawn given its head)</td><td class="what">Never chosen by description length: the categories that reach it carry no number.</td></tr>
    <tr><td>Heads in the representation hierarchy</td><td class="what">Help structure on one grammar (number ignored, 58% → 93% grammatical) but cost characters two points of well-formedness.</td></tr>
    <tr><td>Splits in every consolidation round</td><td class="what">The rounds diverge: one corpus codes 0.7% longer, and characters lose up to 9.5 points of well-formedness at a shorter code.</td></tr>
    <tr><td>The joint chunk move as a final-round refinement</td><td class="what">Changes almost nothing; inside the search it works.</td></tr>
    <tr><td>Choosing the night's candidate before re-analysis</td><td class="what">Lost the characters gain: the move's candidate pulls ahead only during re-analysis.</td></tr>
    <tr><td>A spine of depth 0</td><td class="what">Clauses share categories, but the code is longer on both seeds.</td></tr>
    <tr><td>Merge moves in the chess search</td><td class="what">A few real categories over chunks, but the total code is unchanged.</td></tr>
    <tr><td>The chess read's own answers in context, as the default</td><td class="what">The best memory of a board, but its chunks stop carrying structure.</td></tr>
  </table></div>
</section>"""
rep("""That search is the one new mechanism.</p>
  </div>
</section>""", ITER)

# --------------------------------------------------------------- coherence
rep("""31.9 → 30.3 per character, 81.9 → 73.3 per position.</p>""",
    """31.9 → 30.3 per character, 81.9 → 73.3 per position.</p>
    <p>Two later problems were of a different kind. Grammatical number in the synthetic grammars, and whole components in characters learned from sequences, were within what the descriptions could see. What blocked them was the search, which could not reach them one step at a time. Moves that change several things at once, scored once, reached both (<a href="#complex">agreement</a>, <a href="#characters">characters</a>).</p>""")

# --------------------------------------------------------------- English: joint chunk move
rep("""    <h3>What is new is rarer, and less coherent</h3>""",
    """    <h3>With the joint chunk move</h3>
    <p>Rerun on the cluster with the full night, the 2,500-sentence grammar codes held-out sentences in 12.0 bits. With the joint chunk move, which the night now tries and keeps when the full code prefers it, that falls to 11.7 bits, and 73% of samples of the training length are real instead of 65%. Fewer samples are new (31% instead of 40%): the same trade of novelty for realness as every change at generation time.</p>
    <h3>What is new is rarer, and less coherent</h3>""")

# --------------------------------------------------------------- complex grammars
COMPLEX = """</script></div>
</section>

<section id="complex">
  <div class="tag">Domain 1b · Synthetic grammars</div>
  <h2>Agreement, centre embedding and long sentences</h2>
  <div class="col">
    <p>The paper's six grammars have no agreement, no embedding of one clause inside another, and short sentences. Three new grammars add what real syntax has, each written so that grammaticality can be checked exactly. Each comes with 2,000 training sentences and 500 held out.</p>
    <ul>
      <li><b>agree</b> (6–15 words): determiners and verbs agree in number with their nouns; coordinated subjects are plural; prepositional phrases and coordination.</li>
      <li><b>relative</b> (8–20 words): adds subject relatives and object relatives, which embed one clause inside another (<em>the dog the cat chases runs</em>); 27% of sentences contain one.</li>
      <li><b>deep</b> (10–30 words): adds reduced relatives, sentential complements and coordinated sentences.</li>
    </ul>
    <p>The grammars are oracles only: number, clause types and gold trees are used to evaluate, never as a mechanism.</p>
    <h3>Centre embedding is learned; number at first was not</h3>
    <p>From sentences alone, 27% of the <em>relative</em> grammar's samples contain an object relative, as 27% of its training sentences do, some nested three deep. Number agreement failed: 92–99% of samples were grammatical with number ignored, but half or more failed on agreement alone. No cut through the representation hierarchy separated singular from plural; every node above the words mixed them.</p>
    <p>Splitting every mixed category by the true number shortens the code by 1,600–2,200 bits on <em>agree</em>. Splitting any one category alone lengthens it by 190 to 2,550 bits. So a search that splits one category at a time can never reach number, however well it proposes. Heads in both hierarchies did not help either. A <b>joint split</b> did: it proposes a split of one category, carries it through the categories that combine with it, and scores the whole coordinated split once.</p>
  </div>
  <div class="tablewrap"><table>
    <tr><th><em>agree</em>, 2,000 sentences, full night (seeds 13 / 17)</th><th class="num">Before</th><th class="num">With the joint split</th></tr>
    <tr><td>samples grammatical</td><td class="num">46% / 50%</td><td class="num good">89% / 88%</td></tr>
    <tr><td>samples that fail on agreement alone</td><td class="num">54% / 50%</td><td class="num good">10% / 12%</td></tr>
    <tr><td>training code (bits)</td><td class="num">59,038 / 58,924</td><td class="num good">56,873 / 57,297</td></tr>
    <tr><td>held-out bits per sentence</td><td class="num">28.8 / 28.8</td><td class="num good">27.5 / 27.6</td></tr>
    <tr><td>categories pure in number: nouns / verbs / determiners (seed 13)</td><td class="num">67 / 57 / 68%</td><td class="num good">98 / 92 / 92%</td></tr>
  </table></div>
  <div class="col">
    <p>The split it found first separated <em>a, this, the</em> from <em>these, some, many</em> across 17 categories at once; later ones separated singular from plural nouns. Samples are as new as before (about 100%). The remaining errors come from analyses that join the verb to the subject before the predicate is complete: only 46% of the night's analyses split a sentence into subject and predicate. On <em>relative</em> that share is 32%, and there even the true number split costs bits, so the joint split is not accepted.</p>
    <h3>Long sentences: whole trees, few grammatical samples</h3>
    <p>On <em>deep</em> a single night never forms a whole tree: it leaves forests of about six pieces. Two findings explain part of it. Even from the gold trees, the grammar's samples are grammatical only 4.8% of the time (11.5% with number ignored), so its categories cannot carry what long sentences need. And an embedded clause never shares a category with a whole sentence (0–0.6% of them do), because a whole sentence is described by a position no embedded clause has. A shallower spine, one level of ancestors instead of two, shortens the night's code on both seeds and doubles the gold-tree grammar's grammaticality. Learning night by night in growing batches (250, 500, 1,000, 2,000 sentences) gives whole trees:</p>
  </div>
  <div class="tablewrap"><table>
    <tr><th><em>deep</em>, 2,000 sentences (seeds 13 / 17)</th><th class="num">Whole trees</th><th class="num">Code (bits)</th><th class="num">Held-out bits</th><th class="num">Grammatical</th></tr>
    <tr><td>one night, spine of depth 2</td><td class="num">0% / 0%</td><td class="num">163,567 / 164,455</td><td class="num">76.2 / 76.4</td><td class="num">5.6% / 5.0%</td></tr>
    <tr><td>one night, spine of depth 1</td><td class="num">0% / 0%</td><td class="num">159,963 / 158,920</td><td class="num">75.4 / 75.4</td><td class="num">6.8% / 5.0%</td></tr>
    <tr><td>night by night, shortest sentences first</td><td class="num good">100% / 0%</td><td class="num good">155,978 / 159,435</td><td class="num good">73.9 / 75.7</td><td class="num">7.0% / 5.5%</td></tr>
    <tr><td>night by night, random order</td><td class="num good">100% / 100%</td><td class="num">159,076 / 157,205</td><td class="num">74.7 / 74.3</td><td class="num">4.1% / 3.2%</td></tr>
  </table></div>
  <div class="col">
    <p>The schedule, not the order, gives the whole trees, and with them a shorter held-out code than any single night. What the trees encode is still too coarse: samples are grammatical 3–7% of the time, below a 4-gram model's 18–19% at the same novelty. Sentence structure that generates long sentences correctly is the main open problem on this testbed.</p>
  </div>
</section>

<section id="characters">"""
rep("""</script></div>
</section>

<section id="characters">""", COMPLEX)

# --------------------------------------------------------------- characters
CHARS = """<div class="note"><b>From the sequences alone, near the gold structures.</b> The night's search finds each operator with its first part (<span class="cjk">[⿰ 讠]</span>) but stopped short of joining that chunk to the second part: one join at a time, a whole component costs bits, and only the complete restructuring pays. The joint chunk move proposes whole multi-level composites and scores each once, and with the night's choice made at its end it wins on both seeds:
      <div class="tablewrap"><table>
        <tr><th>2,000 characters from sequences (seeds 13 / 17)</th><th class="num">Before</th><th class="num">Joint chunk move</th></tr>
        <tr><td>well formed</td><td class="num">56.5% / 44.5%</td><td class="num good">92.9% / 88.5%</td></tr>
        <tr><td>every component in an attested slot</td><td class="num">42.2% / 34.7%</td><td class="num good">67.3% / 69.1%</td></tr>
        <tr><td>bracket omission (= commission)</td><td class="num">27.5% / 43.9%</td><td class="num good">11.9% / 6.3%</td></tr>
        <tr><td>training code (bits)</td><td class="num">63,971 / 64,479</td><td class="num good">62,414 / 62,148</td></tr>
        <tr><td>held-out bits per character</td><td class="num">25.6 / 26.3</td><td class="num good">24.8 / 24.8</td></tr>
      </table></div>
      The seed-13 code is now 0.6% above the gold structures' (62,037 bits). The categories stay generic: about 45% of chunk types are real IDS components, and few are specific reused units.</div>
    <h3>Characters as storage</h3>
    <p>How many bits does each method need to store a character, paying for its own model? Over three seeds (2,000 characters learned, 500 held out), the two TRELLIS modes that read token sequences give the shortest total code: 31.4 bits per character (operators as tokens) and 32.2 (sequences alone), against 34.7–34.9 for token n-grams, 42.8 for xz and 39.1 for the best recurrent network even with its weights entropy-coded. On held-out characters the largest network is best (23.1 bits), then TRELLIS (25.1 and 26.3), then a token trigram (29.7).</p>
    <p>The chunks carry real structure. When only which components go together is shuffled, keeping every component in its kind of slot, the sequence modes lose 8–10 bits per character, about as much as the network. About 40% of their chunk types are real IDS components, against 7–15% on shuffled characters. The model with operators as relations loses under a bit: its code is almost entirely about position.</p>"""
rep_between("""<div class="note"><b>From the sequences alone, about half of the way to the gold structures.</b>""",
            """the search remains the limit.</div>""", CHARS, keep_end=False)

# --------------------------------------------------------------- chess
CHESS = """<figcaption>Positions generated from the grammar whose read counts material, blocked pawn pairs outlined. Every one passes every check.</figcaption>
  </figure>
  <h3>Chess as storage</h3>
  <div class="col">
    <p>Chase and Simon's experts recalled real positions far better than random ones because they stored them as chunks. The storage cost of TRELLIS is its description length: the grammar's model bits plus each position's code, which arithmetic coding turns back into the position exactly. On 1500+ games after move 15, against compressors and against neural codes that pay for their parameters:</p>
  </div>
  <div class="tablewrap"><table>
    <tr><th>Bits per position (1500+, ply 30)</th><th class="num">1,000</th><th class="num">4,000</th><th class="num">16,000</th><th class="num">32,000</th></tr>
    <tr><td>64 squares, 13 states each</td><td class="num">236.8</td><td class="num">236.8</td><td class="num">236.8</td><td class="num">236.8</td></tr>
    <tr><td>xz, the whole corpus</td><td class="num">142.3</td><td class="num">129.5</td><td class="num">120.3</td><td class="num">116.5</td></tr>
    <tr><td>TRELLIS, model + data (the counted read)</td><td class="num">86.7</td><td class="num">79.1</td><td class="num">76.3</td><td class="num">75.8</td></tr>
    <tr><td>transformer over the squares, online code</td><td class="num good">82.2</td><td class="num good">72.3</td><td class="num good">65.3</td><td class="num good">62.8</td></tr>
    <tr><td>held out: TRELLIS / transformer</td><td class="num">77.4 / 71.1</td><td class="num">76.1 / 65.2</td><td class="num">75.4 / 60.0</td><td class="num">74.3 / 57.0</td></tr>
  </table></div>
  <figure>
    <img src="chess_storage_tradeoff.png" alt="Eight panels, real and random positions at 1,000, 4,000, 16,000 and 32,000 positions: total bits per position against transmitted parameters for transformers and autoencoders under 8-bit, 16-bit and online codes, with TRELLIS, the read alone, a per-square model and the best byte compressor as horizontal lines. The transformer's online code lies below TRELLIS at every size on real positions.">
    <figcaption>Total code of the training positions against the parameters a network must transmit. TRELLIS (black line) sits below every compressor; a small transformer's online code (blue dots) is below it at every size, and the best network size grows with the data.</figcaption>
  </figure>
  <h3>Making chunks pay</h3>
  <div class="col">
    <p>With the counted read, chunks were worth 0.1–0.4 bits per position at any size, ply, rating or opening family, and nothing on random positions. With the counts removed, chunks pay 2–4 bits and grow to 65–80 types, but the code is 8 bits worse. Forced as whole configurations, castled kings, fianchettos and pawn chains do not pay either: announcing a castled king at its anchor costs 3.2 bits, more than the 2.2 bits of reading it saves. The binding cost was the announcement, not the search.</p>
    <p>Coding each element's label given what the read has already placed nearby (the backward half of the star, its depth chosen by description length) makes chunks pay where positions share openings, and the gain grows with data. Coding the read's own answers the same way is the best memory of a board, but it explains the local structure without chunks:</p>
  </div>
  <div class="tablewrap"><table>
    <tr><th>Held-out</th><th class="num">counted read</th><th class="num">+ labels in context (default)</th><th class="num">+ read in context (option)</th></tr>
    <tr><td>1800+, ply 16, 8,000 positions: bits per position</td><td class="num">50.4</td><td class="num">49.1</td><td class="num">38.3</td></tr>
    <tr><td>&nbsp;&nbsp;chunk gain (bits)</td><td class="num">0.24</td><td class="num good">1.59</td><td class="num">0.20</td></tr>
    <tr><td>&nbsp;&nbsp;chunk types</td><td class="num">3</td><td class="num good">9</td><td class="num">2</td></tr>
    <tr><td>1500+, ply 30, 16,000 positions: bits per position</td><td class="num">75.4</td><td class="num">75.1</td><td class="num">67.0</td></tr>
    <tr><td>&nbsp;&nbsp;gap to the transformer (60.0)</td><td class="num">15.4</td><td class="num">15.1</td><td class="num good">7.0</td></tr>
  </table></div>
  <div class="col">
    <p>With labels in context, the chunks are fianchettos for both sides and pawn chains up to three levels deep (<code>[[bP NW1 [bP E1 bB]] WNW bP]</code>), and random positions still get none. Since chess is about structure carried by chunks, the read in context stays an option, and the work now asks what lets chunks carry more: a context made of the chunks already placed, so that a castled king next door is one informative unit, and attack and defence as relations, so that chunks can join tactically related pieces at a distance.</p>
  </div>"""
rep("""<figcaption>Positions generated from the grammar whose read counts material, blocked pawn pairs outlined. Every one passes every check.</figcaption>
  </figure>""", CHESS)

rep("""<li><b>Attack and defence are relations.</b>""",
    """<li><b>Attack and defence are relations.</b> <span class="tagline">Being tested</span>""")

# --------------------------------------------------------------- where it stands
NEXT = """<div class="col" id="next-body">
    <h3>What carries over</h3>
    <ul>
      <li><b>One framework, every domain.</b> A domain supplies its elements and relations, its context window and its reading order; everything else is shared. In each, the representation hierarchy forms categories of things that behave alike (positions of components, kinds of piece, word classes) and the composition hierarchy forms chunk types (parts of characters, fianchettos and pawn chains, phrases).</li>
      <li><b>Coherence comes from what the descriptions can see,</b> and from a search that can reach what the code prefers. Where single steps cannot pay, coordinated moves scored once have reached grammatical number and the structure of characters.</li>
      <li><b>What the read has just seen</b> conditions each choice: the words before an element in a sentence, the pieces already placed near a square on a board. Description length decides how much to use.</li>
      <li><b>One grammar still parses, generates and prices the data</b> in every domain, every score is exact, and 131 tests check it.</li>
    </ul>
    <h3>What doesn't yet</h3>
    <ul>
      <li><b>Long sentences.</b> Whole trees now form night by night, but only 3–7% of samples are grammatical, and even a grammar read off the gold trees manages 5%. Embedded clauses get categories of their own instead of sharing the sentence's.</li>
      <li><b>Agreement inside relative clauses.</b> The joint split is not accepted on <em>relative</em>, where too few analyses separate subject from predicate.</li>
      <li><b>Generic categories.</b> Characters learned from sequences reach near-gold structure, but their chunk types are rarely the database's components (top-50 recall 2–6%).</li>
      <li><b>Chess chunks carry little of a board.</b> With labels in context they pay 1.3–1.6 bits on shared openings; a transformer is still about 15 bits ahead of the default learner.</li>
      <li><b>Inventive only at a cost.</b> Every change at generation time trades new samples for coherent ones along one curve; about one English sample in a hundred is both new and real, as for a word trigram.</li>
      <li><b>Night cost.</b> Reading the grammar off the hierarchies takes 39–59% of a night, because each candidate cut recounts every record; porting the search, read-out and charts to C++ would speed a night up 3–8 times at most, and scoring cuts by the rows they change is the larger lever. The joint chunk move doubles the search, and a chess night with labels in context takes 3–13 hours.</li>
    </ul>
  </div>
</section>

"""
rep_between("""<div class="col" id="next-body">""", """<footer>""", NEXT, keep_end=True)

rep("""<footer>Code and the full tables: <code>docs/FRAMEWORK.md</code> and <code>docs/V2_DESIGN.md</code> in the ChunkingCobweb repository, branch <code>inside-outside</code>.</footer>""",
    """<footer>Code and the full tables: <code>docs/FRAMEWORK.md</code>, <code>docs/V2_DESIGN.md</code> and a <code>summary.md</code> in each directory of <code>experiments/v2/results/</code>, in the Chunking-Cobweb repository, branch <code>inside-outside</code>. Last updated 12 October 2026.</footer>""")

open(path, "w", encoding="utf-8").write(s)
print("ok", len(s))
