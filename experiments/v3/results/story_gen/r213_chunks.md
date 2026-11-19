# Round 213: which chunks fall out of TRELLIS learning, how consistent they are, and what the non-right-branching trees look like (TinyStories; chess in a linked file)

Chess counterpart: [`../chess_inspect/r213_chess_chunks.md`](../chess_inspect/r213_chess_chunks.md).
Raw numbers: `r213/stats.txt` (language statistics), `r213/cases.txt` (full case studies, 14 sentences, 5 chunk biographies), `r213/candidates.txt` (all 262 non-right-branching, non-left-branching trees of seed 13), dumps `r213/lang_s{13,17,19,23}.json`.
Scripts: `experiments/v3/diag/r213_lang.py` (learn, dump), `r213_lang_analyze.py` (statistics), `r213_lang_cases.py` (case studies, biographies). Nothing in `src/` changed.

## Setup

* Learner: the r204/r205 edge recipe's memory (`logs/v3_r204_grid.sh`: bands naming, CU trees, `store: posterior`, `restructure: sample`, `type_names` off, `word_cc` off, `repair_path`, `repair_use 700`, `exact_alone: content`, `story_facet`, window 7). Window and the edge-exact generation read only matter for generation; the parser reads the same learned memory (default `L.parse`, the chosen binary tree).
* Seeds 13, 17, 19, 23; each learns its own n = 1000 stream (kept sentences of 3-8 words, 250-word vocabulary, story order, `follows` as in `t3_story.py`); checkpoints after 250, 500 and 1000 sentences. Streams are not shared across seeds (different story subsets). Sentences equal to one of the held-out sentences are skipped (as in `t3_parse_consistency.py`), so nothing the parser sees at test time was learned.
* "Chunk" = a word string of 2+ words (not the whole sentence) that is a bracket of the tree `learn()` returns for a training sentence; its count is the number of training sentences in which it was one. The counts are small (seed 13: 496 distinct strings and 1733 bracket occurrences by sentence 1000) because most 3-8 word sentences produce a few brackets. The 8,000 or so "composite strings" in memory (e.g. 7590 for seed 13) are the stored candidate composites, most never chosen in a learn tree.
* Held-out set: the shared `experiments/v3/results/silver/heldout_s13.json`, 400 sentences. Reference = the NLTK shallow chunker of `story_parse_eval.py` (NP, PP, VP, AP, VG spans of 2+ words, whole sentence excluded). The reference is flat chunks, not a full tree.
* Not rerun, cited: the earlier parse evaluation for the same recipe family on another 300-sentence held-out set (`story_parse_eval/r203_parse.md`, `r198_gold_ceiling.md`). Where my numbers and those differ, the sets differ.
* Baselines for chunk inventories, built on the same prefix of each seed's stream: (a) BPE merges (`story_parse_eval.learn_bpe`), budget = the number of distinct strings TRELLIS bracketed at that checkpoint, ranked by merge order; (b) raw frequency: the top contiguous 2-8-word n-grams by count; (c) the same, length-matched to the TRELLIS top-k length histogram; (d) random: k strings drawn uniformly from the n-grams (2-8 words) that occur at least twice in that seed's prefix (pool 350-402 strings at 250, 770-855 at 500, 1618-1681 at 1000), 100 draws.

## 1. Which chunks fall out

Most-used chunks, pooled over the four seeds at 1000 (count = training sentences in which the string was a bracket, summed over seeds):

the bird 84; tim was 66; they were 59; was very 56; he wanted 56; he was 53; one day 52; it was 51; they played 50; the dog 50; the cat 48; every day 47; the park 45; the little 41; lily was 40; to play 39; tim and 37; she was 35; he wanted to 35; his mom 34; played together 34; they all 34; they played together 33; sue was 33; lily and 33; so happy 32; to do 32; the ball 32; tom and 32; and had fun 29; her mom 29; in the park 29; tim was very 28; had fun 28; the man 28; his friends 28; all day 28; what to do 27; they had 27; thank you 27.

Per seed, top 25 (counts in brackets):

* seed 13: he wanted (22); the park (19); tim was (19); the dog (15); the bird (14); lily and (13); the cat (13); they were (13); he was (12); played together (12); they played (12); in the park (11); tom and (11); it is (10); said yes (10); she was (10); so happy (10); the little (10); to do (10); to play (10); and ben (9); he wanted to (9); it was (9); one day (9); the boy (9).
* seed 17: the bird (28); they were (19); was very (19); they played together (17); every day (16); he wanted to (16); the dog (16); they played (16); he wanted (15); he was (15); tim was (15); it was (14); one day (14); she was (12); the man (12); tim and (12); played together (11); they all (11); do you want (10); had fun (10); his mom (10); lily was (10); the cat (10); the little (10); to play (10).
* seed 19: was very (26); the bird (22); so happy (18); had an (17); one day (17); the cat (16); tim and (16); every day (15); he was (15); it was (15); they played (15); tim was (15); together every day (13); was so happy (13); his friends (12); to do (12); very happy (12); he did not (11); he wanted (11); and had fun (10); had fun (10); he did (10); he had (10); her mom (10); thank you (10).
* seed 23: the bird (20); lily was (18); they were (17); tim was (17); the little (14); it was (13); the park (13); to play (13); one day (12); sue was (12); to help (12); he was (11); the dog (11); his mom (10); the tree (10); tom and (10); thank you (9); the ball (9); the cat (9); every day (8); he wanted (8); her mom (8); in the park (8); lily and (8); lily was very (8).

### What kinds of chunks (read off the lists above, not assumed)

* **Short.** In the pooled top 200 (counts summed over the four seeds), 141 are 2 words, 49 are 3 words and 10 are 4 words; none longer. No sentence-length chunk, no "once upon a time" (the held-out and 3-8 word training sentences hold almost no such opener; `upon` is outside the 250-word vocabulary).
* **Determiner (or possessive) + noun**, the commonest kind: the bird, the dog, the cat, the park, the ball, the man, the tree, the boy, his mom, her mom, his friends, the little (a determiner + adjective that is left open for the noun).
* **Subject + auxiliary or verb**, the next commonest and in a flat reference not a phrase: tim was, lily was, sue was, he was, she was, they were, it was, they played, they all, he wanted, tim was very, lily was very, they had, he did.
* **Name + "and"**: lily and, tom and, tim and, and ben, sue and. These are the left half of a coordinated subject.
* **Fixed adverbials and openers**: one day, every day, all day, together every day, the one real fixed expression at the start of a story. Also thank you, had fun, and had fun, what to do, so happy, very happy, played together, they played together.
* **to + verb, verb + to**: to play, to do, to help, he wanted to.
* **Prepositional phrases**: in the park (8 to 11 per seed, 29 pooled) and the park; no other preposition+noun phrase reaches the top 50.

So the chunk inventory at 1000 sentences is the frequent word pairs of the corpus, mostly determiner+noun, subject+be/verb and fixed time words. It is a tidier list than it sounds: it is also very close to the list a frequency counter gives (section 2).

### The content concepts that group them

Seed 13, 1000 sentences, the concept each of the 30 commonest chunks falls under (nearest concept = parent of its content leaf; wider = deepest concept on its path holding 4 or more member strings). Concept ids are internal. Members shown are other strings under the same concept.

| chunk (count) | nearest concept, members | wider concept, members |
|---|---|---|
| he wanted (22) | she wanted | tom wanted; she wanted; he liked |
| the park (19) | their room; the house; back home | same |
| tim was (19) | one day; they said; they found; she put; he walked | same |
| the dog (15) | her mom; new toy; it up; his friends; his mom | same |
| the bird (14) | happy too; the animals; was inside; with tim; the tree | same |
| lily and (13) | anna and; sara and | anna and; sue and; sara and; mia and; she had |
| the cat (13) | max lived; once there; she lived; now tim | same |
| they were (13) | the little; they said | tom and; the ball; tom thought; they found; she played |
| he was (12) | in the; but they | in the; lily was; they were; sue was; the bird |
| they played (12) | one day; tim was; sam came; they asked; they went | same |
| tom and (11) | she played; he said; and his; they thought | same |
| he wanted to (9) | he liked to; they like to; tim liked to | same |
| and ben (9) | and sue; and tom | and sam; and sue; and tom; and lucy; and lily |
| one day (9) | at the; he knew; inside the; as he; the little | same |
| in the park (11) | with his friends; all day long | in the sun; with his friends; his friends outside; her friend tom |

Read plainly: a few concepts are clean. The "name + and" chunks (lily and, and ben) fall with anna and, sue and, sara and, mia and, and tom, and lucy; "he wanted" falls with she wanted, tom wanted, he liked; "he wanted to" falls with he liked to, they like to, tim liked to. Most of the others are not categories a linguist would draw. "the bird" shares a concept with "happy too; the animals; was inside; with tim; the tree"; "tim was" with "one day; they said; they found; she put". The concept follows a shared role in the contexts, not a part of speech, and the concept a given chunk falls under changes along the stream and differs by seed (section 5, biographies). The two-word subject+verb strings land with other two-word strings, whatever their category.

## 2. Consistency, with baselines

Mean over the 6 seed pairs. "Jaccard" compares the top-k chunk strings of two seeds. Every row is on the same stream prefix.

| checkpoint | top-k | TRELLIS | raw frequency | frequency, length-matched | BPE merges | random (same pool) |
|---|---|---|---|---|---|---|
| 250 | 50 | .153 | .258 | .241 | .224 | .024 |
| 250 | 200 | .194 | .233 | .229 | .150 | .099 |
| 500 | 50 | .244 | .391 | .391 | .346 | .013 |
| 500 | 200 | .223 | .366 | .363 | .263 | .057 |
| 1000 | 50 | .332 | .510 | .490 | .431 | .008 |
| 1000 | 200 | .298 | .454 | .438 | .362 | .031 |

* TRELLIS chunk inventories overlap across seeds far more than chance (about .30 to .33 at 1000 against .01 to .03), and the overlap grows with data (.15 to .33 for top-50).
* **They overlap less than the baselines that need no learning.** At 1000, raw n-gram frequency has Jaccard .510 (top-50) and .454 (top-200) against TRELLIS's .332 and .298; BPE .431 and .362. Length matching changes frequency by about .02. Chunking adds no cross-seed agreement beyond what the frequent n-grams give; it loses some.
* Within a seed, TRELLIS's top-50 and the frequency top-50 overlap .352 (top-200: .341); TRELLIS and BPE .325 (.335). So the TRELLIS list is related to frequency but not the same list.
* Length profile of the top 200 (seed 13): TRELLIS 2/3/4 words 146/40/14; frequency 2/3/4/5 words 144/47/7/2. Not different in kind.
* Count-weighted overlap (sum of min over sum of max of raw counts over the union): all bracketed TRELLIS strings .196 / .243 / .284 at 250 / 500 / 1000; all n-grams with their raw counts .136 / .193 / .245 (this one is dominated by the long tail of rare n-grams). Restricted to top-200: TRELLIS .178 / .230 / .305, length-matched frequency .246 / .375 / .464.

**Chunks every seed finds.** 54 strings are in the top-200 of all four seeds at 1000 (random expectation: about 0.4). 14 are in the top-50 of all four: and had fun; every day; he wanted; he was; it was; one day; the bird; the cat; the dog; the little; they all; they played; they were; tim was. Others in all four top-200s: the park; lily was; to play; tim and; she was; he wanted to; played together; his mom; lily and; sue was; they played together; to do; the ball; tom and; in the park; her mom; tim was very; had fun; all day. Among strings bracketed at least 3 times in one seed, 66 are bracketed at least 3 times in all four seeds, 55 in three, 126 in two, 312 in one only.

**Along the stream** (within a seed, top-k Jaccard of an earlier checkpoint with the final 1000):

| top-k | checkpoint | TRELLIS | frequency | frequency, length-matched | BPE |
|---|---|---|---|---|---|
| 50 | 250 | .350 | .483 | .465 | .413 |
| 50 | 500 | .551 | .691 | .668 | .569 |
| 200 | 250 | .436 | .460 | .447 | .319 |
| 200 | 500 | .511 | .616 | .588 | .489 |

Spearman rank correlation of a chunk's count at 250 (500) with its count at 1000, over the final top-50: TRELLIS .488 (.677), frequency .664 (.800). All of the final top-50 TRELLIS chunks are already brackets at 500 (1.00; .91 at 250). TRELLIS's ranking is less stable than frequency along the stream.

### Parse consistency (the saved held-out parses, 400 sentences)

The first four rows use the story_parse_eval definitions (183 repeated 2-4 word strings; "consistency" = share of occurrences bracketed; "decisive" = share bracketed the majority way). Gold is the NLTK phrase set.

| | consistency | decisive | cross-seed bracket F1 | stability vs previous checkpoint | F1 vs NLTK | brackets/sentence |
|---|---|---|---|---|---|---|
| NLTK reference | .130 | .989 | - | - | 1 | - |
| right-branching | .000 | 1.000 | 1.000 (identical) | - | .488 | - |
| left-branching | .354 | .940 | 1.000 | - | .074 | - |
| TRELLIS 250 | .306 | .841 | .652 | - | .311 | 3.27 |
| TRELLIS 500 | .292 | .850 | .701 | .849 | .327 | 3.27 |
| TRELLIS 1000 | .285 | .861 | .724 | .891 | .341 | 3.27 |

* Cross-seed bracket F1 (same held-out sentences, independent learners, mean sentence F1 over the 6 pairs): .652, .701, .724 at 250, 500, 1000. Per pair at 1000: 13-17 .728, 13-19 .719, 13-23 .805, 17-19 .642, 17-23 .745, 19-23 .703. Baselines already measured (other 300-sentence held-out set, `r203_parse.md`): random binary .343, BPE x1 .682, TRELLIS default chosen .742, parts .756. The right- and left-branching parses agree across seeds trivially (F1 1), so agreement alone does not show structure.
* Phrase consistency .285 at 1000 is above the reference's .130: TRELLIS brackets repeated strings more often than the reference does (over-bracketing), and brackets one string the same way 86% of the time against the reference's 99%.
* "Bracketed somewhere" consistency (`t3_parse_agree.py`: of 2-4 word strings the parser brackets at least once, the share of occurrences bracketed): seed 13 .509 (153 strings), 17 .510 (151), 19 .505 (144), 23 .537 (140). Right-branching .610 (84 strings), left-branching .802 (81).
* F1 against the NLTK reference: TRELLIS .320, .327, .386, .331 for seeds 13, 17, 19, 23 at 1000; **right-branching .488 beats every seed**, as in the r203 table (.466 vs .37 there). Right-branching earns its F1 from the reference's end-anchored phrases (VP, PP at the end of a sentence).
* Existing measurements for the same recipe family, not rerun: consistency .324 / .335, decisive .868 / .857 (default / parts, chosen tree, 1000 sentences) against gold .116 / .982 on the 300-sentence set; BPE x1 consistency .443, decisive .942 (`r203_parse.md`, `r198_gold_ceiling.md`).

## 3. Trees that are not right-branching

Binary chosen tree, whole-sentence bracket excluded, shared held-out set. "Differs from right-branching" means the set of inner brackets differs from {(i, n): 1 <= i <= n-2}; "neither" means it equals neither the right- nor the left-branching tree.

| checkpoint | differs from right-branching | equals left-branching | neither |
|---|---|---|---|
| 250 | .848 | .184 | .664 |
| 500 | .837 | .163 | .674 |
| 1000 | **.822** | .159 | .663 |

At 1000 per seed: seed 13 .830 (332 of 400 sentences) differs, .175 left-branching, .655 neither (262); seed 17 .907 (363), .185, .723 (289); seed 19 .735 (294), .050, .685 (274); seed 23 .815 (326), .228, .588 (235). 263 of 400 sentences are non-right-branching in all four seeds.

Score of the non-right-branching parses against the NLTK reference (F1 on the sentences concerned, 1000 sentences learned):

| seed | n differing | TRELLIS | right-branching | left-branching | on the "neither" set: n, TRELLIS / RB / LB |
|---|---|---|---|---|---|
| 13 | 332 | .283 | .471 | .077 | 262: .310 / .435 / .079 |
| 17 | 363 | .309 | .476 | .075 | 289: .333 / .442 / .067 |
| 19 | 294 | .324 | .446 | .081 | 274: .330 / .436 / .081 |
| 23 | 326 | .293 | .474 | .077 | 235: .326 / .423 / .070 |

So where TRELLIS departs from right-branching it scores .28 to .32 F1 against the reference, about .15 below right-branching on the same sentences and far above left-branching. The deviations are real structure in some sentences, but the reference prefers the right-branching tree. (For orientation, right-branching scores .59 to .82 on the sentences where TRELLIS happens to produce it.)

### Showcase (chosen)

These are picked by reading `candidates.txt` (seed 13, 1000 sentences, sentence index in the held-out set) for a non-right-branching structure that matches linguistic intuition, one construction each. They are not typical: the table above gives the typical case. Tree = TRELLIS's binary chosen tree; reference = NLTK phrases.

| # | construction | TRELLIS (seed 13) | NLTK reference |
|---|---|---|---|
| 366 | subject NP, then VP | `[[the tree] [[had no] friends]]` | `[the tree] [had [no friends]]` |
| 202 | subject NP, then VP | `[[the bird] [[looked at] spot]]` | `[the bird] [looked [at spot]]` |
| 178 | fixed opener "one day", PP "to the park" | `[[[[one day] tom] went] [to [the park]]]` | `[one day] [[tom went] [to [the park]]]` |
| 299 | fixed opener, object NP | `[[[one day] max] [saw [a door]]]` | `[one day] [[max saw] [a door]]` |
| 361 | subject NP, adjective+noun "big park", PP | `[[[this rock] lived] [in [a [big park]]]]` | `[this rock] [lived [in [a big park]]]` |
| 205 | "all the" + noun, then AP | `[[[[all the] animals] were] [very happy]]` | `[all the animals] [were [very happy]]` |
| 328 | NP "a fun day" + PP "for everyone" | `[[it was] [[a [fun day]] [for everyone]]]` | `it [was [a fun day] [for everyone]]` |
| 301 | subject NP, PP "after the ball" | `[[[the cat] ran] [after [the ball]]]` | `[the cat] [ran [after [the ball]]]` |
| 147 | PP "in the park", "amy loved" | `[[[amy loved] to] [play [in [the park]]]]` | `[amy loved] [[to play] [in [the park]]]` |

Honest reading of the showcase: in every one the NP and the PP are bracketed as the reference has them, but the binary tree attaches the subject NP to the verb first ("[[the cat] ran]") where the reference leaves subject and verb phrase apart, so even these show F1 of .50 to .75 against the reference (per sentence, bracket F1) and not 1.0. "once upon a time" cannot be shown: no held-out sentence contains it.

### Random block (drawn, not chosen)

Four trees drawn with `random.Random(213).sample` from the 262 seed-13 sentences (1000 sentences learned) that are neither right- nor left-branching; the draw returned sentence indices 110, 314, 163, 219.

| # | TRELLIS (seed 13) | NLTK reference |
|---|---|---|
| 110 | `[[[[one day] he] saw] [a [big ball]]]` | `[one day] he [saw [a big ball]]` |
| 314 | `[[the girl] [was [happy [to help]]]]` | `[the girl] [was happy] [to help]` |
| 163 | `[[[she ran] fast] [to [get [the cake]]]]` | `she [ran fast] [[to get] [the cake]]` |
| 219 | `[[i will] [help [you [said tom]]]]` | `i [[will help] you] [said tom]` |

110 and 314 look good, 163 groups the subject with the verb and puts "to" with the verb phrase on the right, and 219 mis-parses a sentence that has two clauses ("i will help you" / "said tom") by joining "you said tom".

### Looks wrong

Held-out sentence 358, seed 13 (the same tree in all four seeds):

* TRELLIS: `[[[[suddenly the] little] girl] [had [an idea]]]`
* reference: `suddenly [the little girl] [had [an idea]]`

The adverb "suddenly" is bracketed with the determiner ("suddenly the", posterior .44), so the noun phrase "the little girl" is never a constituent: it is built left to right as ((suddenly the) little) girl. A likely cause (not tested): "suddenly the" and "the little" are both frequent word pairs, and the parser picks the pairing by what has been seen. All four seeds produce this tree; this is a systematic error, not noise.

## 4. Case studies

Each case lists the sentence, TRELLIS's tree of every seed beside the NLTK reference, then each chunk of the seed-13 tree with its posterior (summed over splits), whether the reference has it, which other seeds bracket the same span, whether memory stored the string as a composite (and how often it was a bracket in a training tree), how often the string occurs in the 400 held-out sentences and how often each seed brackets it as that span, and the concept it falls under. Full text for 14 sentences in `r213/cases.txt`.

**Case 1: "the tree had no friends" (366, chosen).** Trees: seed 13 `[[the tree] [[had no] friends]]`; seed 17 `[[[the tree] had] [no friends]]`; seed 19 `[[the tree] [had [no friends]]]`; seed 23 `[[[[the tree] had] no] friends]`. Reference `[the tree] [had [no friends]]`.
* "the tree" (post .58): reference has it; all 4 seeds bracket it here. Stored; a training bracket 3 times. In the held-out set it occurs 3 times and is bracketed as that span in 1/3, 1/3, 2/3, 1/3 of the occurrences by seeds 13/17/19/23. Concept: boy said; lived with; wanted to.
* "had no" (post .44): not in the reference, no other seed brackets it. Stored, never a training bracket. Concept: a boy; have no; do not; day sue; car went.
* "had no friends" (post .44): reference has it; only seed 19 agrees. Not stored.
Four seeds, four trees, but "the tree" is a constituent in all of them.

**Case 2: "the bird looked at spot" (202, chosen).** Seed 13 `[[the bird] [[looked at] spot]]`; seed 17 `[[[[the bird] looked] at] spot]`; seeds 19 and 23 `[[the bird] [looked [at spot]]]`, equal to the reference.
* "the bird" (post .66): all 4 seeds, reference has it, stored, training bracket 14 times; held-out: 13 occurrences, bracketed as that span 9/13, 9/13, 8/13, 10/13 (seeds 13/17/19/23). Concept: happy too; the animals; was inside; with tim; the tree; feel better.
* "looked at" (post .58): not in the reference, only seed 13 brackets it (here and in 1 of its 3 occurrences). Training bracket once.
* "looked at spot" (.52): reference, seeds 19 and 23 agree.
The same string "the bird" is bracketed in 8 to 10 of 13 held-out occurrences by each seed: the other 3 to 5 occurrences are attached to a preceding word ("... and the bird").

**Case 3: "one day tom went to the park" (178, chosen).** All four seeds give the identical tree `[[[[one day] tom] went] [to [the park]]]`; reference `[one day] [[tom went] [to [the park]]]`.
* "one day" (post .96): all seeds, reference has it; 6 of 6 held-out occurrences are bracketed as that span by every seed; stored; a training bracket 9 times. Concept c118844: at the; he knew; inside the; as he; the little; he played.
* "the park" (.97): all seeds, reference; 7 of 7 occurrences by every seed; training bracket 19 times. Concept: their room; the house; back home.
* "to the park" (.74): reference; all seeds; 2 of 2. Concept: to each other; to play with; into the water; to her mommy.
* "one day tom" (.97): not in the reference, all seeds; stored; training bracket twice.
* "one day tom went" (.48): not stored, all seeds still bracket it.
This is the clearest consistency in the study: the opener and the PP are the same in all seeds, wherever they occur.

**Case 4: "the cat ran after the ball" (301, chosen).** Seeds 13, 17, 23 `[[[the cat] ran] [after [the ball]]]`; seed 19 `[[the cat] [ran [after [the ball]]]]` (the reference).
* "the cat" (.96): 4 seeds; stored; training bracket 13; held-out occurrences 10, bracketed as that span 5/10, 7/10, 4/10, 6/10 by seeds 13/17/19/23. Concept: max lived; once there; she lived; now tim.
* "the ball" (.98): 4 seeds; training bracket 5; 6 occurrences, bracketed 4/6, 4/6, 4/6, 3/6. Wider concept c141465 (7.6 bits, 23 member strings): tom and; tom thought; they found; she played; they were; now you; he said.
* "the cat ran" (.52): not in the reference; seeds 17 and 23. "after the ball" (.66): reference, all seeds, not stored.

**Case 5: "all the animals were very happy" (205, chosen).** Seeds 13, 17, 23 `[[[[all the] animals] were] [very happy]]`; seed 19 `[[[all the] animals] [were [very happy]]]`; reference `[all the animals] [were [very happy]]`.
* "all the" (.97): not in the reference, all seeds. Concept: and i; tim's friend; the cat; they do; tim's friends; one day; he asked; the fish.
* "very happy" (.86): reference, all seeds here, but across the held-out set it occurs 21 times and is bracketed as that span only 5/21, 3/21, 9/21, 3/21 by seeds 13/17/19/23. A frequent, stored (4 training brackets) chunk with no consistent handling (the pattern behind the 21 occurrences was not examined).
* "all the animals" (.84): reference; all seeds; stored; training bracket once.

**Case 6: "it was a fun day for everyone" (328, chosen).** Seed 13 `[[it was] [[a [fun day]] [for everyone]]]`; seed 17 `[[[it was] a] [fun [day [for everyone]]]]`; seeds 19, 23 `[[it was] [a [fun [day [for everyone]]]]]`; reference `it [was [a fun day] [for everyone]]`.
* "it was" (.89): 4 seeds; training bracket 9; yet of its 12 held-out occurrences each seed brackets 5/12, 4/12, 3/12, 5/12.
* "a fun day" (.52): reference, seed 13 only; "fun day" (.51): only seed 13. "for everyone" (.87): reference; all four.
Seed 13's tree matches the reference's NP and PP and the other three do not.

**Case 7: "amy loved to play in the park" (147, chosen).** Seeds 13, 17, 23 `[[[amy loved] to] [play [in [the park]]]]`; seed 19 `[[amy loved] [[to play] [in [the park]]]]`, which is the reference.
* "in the park" (.92): reference; all four seeds; stored; training bracket 11. Concept c20186 (8.0 bits, 15 strings): in the sun; with his friends; his friends outside; her friend tom; his toys there; with the bear; with his ball; his toy car; run all day; with their toys.
* "amy loved" (.94): reference; all seeds; not stored (so built from words at parse time). "amy loved to" (.91): not in the reference; seeds 17, 23.
* "play in the park" (.83): not in the reference; seeds 17, 23; stored, training bracket once. Concept c98073: play with his ball; play near the water; play with her friends; play at the park; play with her ball.

**Case 8: "suddenly the little girl had an idea" (358, looks wrong).** All four seeds `[[[[suddenly the] little] girl] [had [an idea]]]` against `suddenly [the little girl] [had [an idea]]`.
* "suddenly the" (.44): not in the reference, not stored. "suddenly the little" (.68) and "suddenly the little girl" (.70): not stored. "an idea" (.91): reference; stored but never a training bracket (0). "had an idea" (.46): reference, stored, training bracket 0.
* This case is why the consistency numbers must not be read as correctness: all four seeds agree, and the agreement is on an error.

**Case 9: "one day he saw a big ball" (110, random draw).** Seeds 13, 17, 19 `[[[[one day] he] saw] [a [big ball]]]`; seed 23 `[[[[[one day] he] saw] a] [big ball]]`; reference `[one day] he [saw [a big ball]]`. "one day" (.98) all seeds; "one day he" (.99): all seeds, not in the reference, stored, training bracket once; "big ball" (.98): not in the reference, not stored; "a big ball" (.53): reference; seeds 17, 19 agree. Concept of "one day he": c92066 (10.8 bits): one day max; tim was playing; they found it; she said no; it was a; she put it.

**Case 10: "the girl was happy to help" (314, random draw).** Seeds 13, 19, 23 `[[the girl] [was [happy [to help]]]]`; seed 17 `[[[the girl] was] [happy [to help]]]`; reference `[the girl] [was happy] [to help]`. "the girl" (.91), "to help" (.94): reference and all seeds; "happy to help" (.56): not in the reference, all seeds. Concept of "the girl": with it; with max; their friend; it back; in it; it on; the bird; him happy.

## 5. Chunk biographies

Counts are training sentences (by index in the seed's stream) in which the string was a bracket of the learned tree. "Concept" is the deepest concept on the string's content path with 4 or more member strings, at that checkpoint. Full lists in `r213/cases.txt`.

**"the bird"** (determiner + noun). First bracketed at sentence 27 (seed 13), 59 (17), 27 (19), 37 (23). Count at 100 / 250 / 500 / 750 / 1000 sentences: seed 13: 2, 5, 8, 13, 14; seed 17: 3, 12, 18, 24, 28; seed 19: 1, 7, 14, 17, 22; seed 23: 3, 7, 10, 12, 20. First example in seed 13: "the bird saw a big cat". Stored as a composite from 250 on in every seed. Concept in seed 13 at 250 / 500 / 1000: (all day; had fun; day long; very happy; of fun; their toy), (he was; he liked; yes you; they played), (happy too; the animals; was inside; with tim; the tree; feel better); i.e. a different set of company at every checkpoint.

**"one day"** (fixed opener). First bracketed at sentence 250 (seed 13: "one day max saw a big door"), 106 (17), 71 (19), 18 (23). Count at 100 / 250 / 500 / 750 / 1000: seed 13: 0, 0, 4, 8, 9; seed 17: 0, 2, 5, 11, 14; seed 19: 1, 2, 3, 8, 17; seed 23: 2, 2, 6, 9, 12. The seed 13 chunk appeared only after the first 250 sentences, but was stored as a composite by 250 (it existed in memory before any learned tree used it). Concept in seed 13 at 250 / 500 / 1000: (tim was; they found; she said; she put), (the cat; the little; tim was; at the; but the; sue started; then he), (at the; he knew; inside the; as he; the little; he played): it shares a concept with "tim was" and "the little", other openers and subject chunks, not with "every day" or "all day".

**"tim was"** (name + be). First bracketed at sentence 32 (seed 13), 3 (17), 52 (19), 103 (23). Count at 100 / 250 / 500 / 750 / 1000: seed 13: 2, 4, 7, 12, 19; seed 17: 2, 4, 8, 13, 15; seed 19: 2, 3, 6, 8, 15; seed 23: 0, 1, 7, 14, 17. Seed 17 at 250 shares a concept with sue was; tim felt; tom was; sara was; bob was; everyone was (a clean "name + was" concept) and by 1000 with they both; they all; the bird; the tree; they were; tom was. Seed 13 at 1000: one day; they said; they found; she put; he walked; the park.

**"in the park"** (PP). First bracketed at sentence 194 (seed 13: "lily and tom were playing in the park"), 107 (17), 129 (19), 155 (23). Count at 100 / 250 / 500 / 750 / 1000: seed 13: 0, 1, 4, 7, 11; seed 17: 0, 3, 6, 6, 7; seed 19: 0, 1, 3, 3, 3; seed 23: 0, 3, 4, 6, 8. Concepts at 1000 in seed 13: in the sun; with his friends; his friends outside; her friend tom; his toys there; with the bear; with his ball. In seeds 19 and 23 a different group. The chunk appears as the training bracket in every seed by sentence 200, with "playing in the park" and "play in the park" as the first contexts.

## 6. What this says, and does not

* TRELLIS finds, by 1000 sentences, a chunk inventory dominated by 2-word strings of determiner+noun, subject+auxiliary, name+and, and fixed time words ("one day", "every day"). The same inventory comes out of every seed at about 1/3 Jaccard in the top 50, roughly 40 times the random rate.
* That agreement is lower than for raw n-gram frequency or BPE on the same data (.33 vs .51 and .43 top-50 at 1000). Chunking by TRELLIS's criterion adds nothing in cross-seed agreement over frequency; whatever it adds (3+ word strings such as "they played together", the preferences for subject+verb strings) is not more stable.
* Parses agree across seeds at bracket F1 .65 to .72 (random binary .34, BPE .68); stability across checkpoints .85 to .89.
* 82% of held-out parses differ from right-branching (66% are neither right- nor left-branching). On those, TRELLIS scores .28 to .32 F1 against NLTK, below right-branching on the same sentences (.45 to .48), so the non-right-branching structure is not on net a gain against this reference.
* The clear examples (case 3, "one day ..., to the park") are real and recur in every seed. A frequent chunk can also be handled inconsistently ("very happy": bracketed in 3 to 9 of 21 occurrences), and all four seeds can share the same error ("suddenly the little girl").
* Content concepts group chunks by shared contexts. Only a few are clean grammatical categories (names + "and", "he wanted" with "she wanted"/"he liked"), and a chunk's concept changes along the stream.

## Rule notes

One trivial `python -c "print(1)"` and one `python3 --version` ran on the login node, and an `import nltk` check on the login node was blocked by the auto-mode classifier before it ran; everything else, the training, dumps and analyses, ran as batch jobs on compute nodes. All four learning jobs and the analysis jobs were `sbatch --requeue` on the overcap partition with the standard exclude list, no jobs or waiters left running.
