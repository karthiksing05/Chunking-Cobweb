# Round 213 (chess): which chunks fall out of the cs73L memory, how consistent they are across seeds and along the stream, and example board decompositions

Language counterpart: [`../story_gen/r213_chunks.md`](../story_gen/r213_chunks.md).
Raw numbers: `r213_chess_stats.txt` (inventory statistics and the unit table of 30 boards), `r213_chess_boards.txt` (text diagrams of six boards, all four seeds, cross-seed decomposition agreement), dumps `r213_chess_s{13,17,19,23}.json`.
Scripts: `experiments/v3/diag/r213_chess.py` (learn, dump), `r213_chess_analyze.py` (statistics), `r213_chess_boards.py` (diagrams). Nothing in `src/` changed.

## Setup

* Memory: the cs73L recipe (`logs/v3_cs73L_submit.sh`): GraphTrellis, `store: best`, bands naming, `restructure: sample`, `type_names: sample`, `type_key: concept`, `chess_flat`, `chess_join: any`, chunks of up to 4 pieces. The Chase and Simon exact protocol around it (K = 7, recall as generation) was not rerun here: only the learning loop. Learning is deterministic, so it reproduces what cs73L learned: seed 13 stores 14,681 chunks after 4,000 positions here and in the cs73L log (`logs/v3-cs73L-13.log`).
* Stream: `train_long.fen` (27,459 positions), shuffled by the seed, learned in order. cs73L ran the whole stream (about 23 hours for 27,459); **these runs stop at the first 4,000 positions** (2,500 to 3,000 seconds each), with checkpoints at 500, 1000, 2000 and 4000. For seeds 13 and 17 the 4,000 positions are the ones cs73L learned first. Seeds 13, 17, 19, 23 (cs73L itself ran only 13 and 17).
* A **chunk** = a committed multi-piece unit (2 to 4 pieces on their squares) that was chosen in the parse of a learned position. Its **use count** = the number of training positions where it was chosen; "stored" = the memory holds it as a composite. A chunk is written `f2:wP|g1:wK|g2:wP|h2:wP` (square, colour and piece). Counts: seed 13 has 1,860 distinct chunks used at least twice at 4,000 positions (seed 17 1,893, seed 19 1,795, seed 23 1,965); sizes for seed 13: 275 of 2 pieces, 219 of 3, 1,366 of 4. Total uses 9,277 / 9,052 / 9,886 / 10,556 (seeds 13 / 17 / 19 / 23).
* The names for chunks ("castled king with pawn shield", "pawn chain", "corner rook with its pawns") are labels assigned by a small rule set in `r213_chess_analyze.py::classify`, not by the memory; "mixed piece group" is everything the rules do not name. The rules: king on the back rank, files f-h (or a-c), with at least two own pawns within two ranks in front and one file either side = castled with shield; two pawns of one colour on adjacent files and ranks = chain; same rank, adjacent files = duo; a back-rank corner rook with a pawn = corner rook; a bishop on g2/b2/g7/b7 behind a pawn on g3/b3/g6/b6 = fianchetto.
* Board decompositions: for 20 fixed middle-game positions (the first 20 of `middle.fen`) and 10 end-game positions, parse with learning off and take the units the memory recognises (content plus context recognition >= 16 bits, as in `chess_inspect.py` and the cs73 protocol), at each checkpoint, per seed.

## 1. Which chunks fall out

Most-used chunks per seed at 4,000 positions (use counts):

| rank | seed 13 | seed 17 | seed 19 | seed 23 |
|---|---|---|---|---|
| 1 | f2 g1 g2 h2 (white castled, shield) 496 | e6 f7 g7 h7 (black pawn chain) 229 | f2 g1 g2 h2 449 | f2 g1 g2 h2 507 |
| 2 | f2 g1 g2 h3 135 | f8 rook, g8 king 171 | f2 g1 g2 h3 300 | f2 g1 g2 h3 262 |
| 3 | f7 g6 g8 h7 (black king, h-pawn, g6) 115 | e6 f7 g7 h6 103 | f2 g1 g3 h2 176 | f2 g1 g3 h2 186 |
| 4 | f7 g7 g8 h7 (black castled, shield) 112 | a1 rook, a2 b2 c2 pawns 80 | e6 f7 g7 h7 167 | e6 f7 g7 h7 174 |
| 5 | e6 f7 g7 h7 94 | a7 a8 rook, b7 c7 pawns 78 | f8 rook g8 king 156 | g7 h7 pawns 120 |
| 6 | f2 g1 g3 h2 94 | f1 rook, f2 pawn, f3 knight, g1 king 75 | f8 rook, g7 h7 pawns, g8 king 121 | d5 e6 f7 g7 102 |
| 7 | g7 g8 h7 (3 pieces) 83 | a7 a8 rook 70 | g7 h6 92 | f7 g7 h7 102 |
| 8 | a7 b7 65 | a1 rook, a2 b2 c3 pawns 69 | d5 e6 f7 g7 80 | g7 h6 99 |

(Full top 15 per seed in `r213_chess_stats.txt`; all squares are as printed there, pieces are pawns unless named.)

What kinds of chunks (my labels, section "Setup"):

| class | share of the per-seed top-50 (pooled over seeds) | top-200 | of all uses |
|---|---|---|---|
| pawn chain (diagonal pawns) | .35 | .36 | .32 |
| castled king (kingside) + pawn shield | .21 | .16 | .17 |
| corner rook with its pawns | .14 | .12 | .11 |
| mixed piece group | .12 | .20 | .24 |
| pawn duo / phalanx (side by side) | .07 | .06 | .04 |
| back-rank pieces (e.g. f8 rook + g8 king) | .05 | .05 | .04 |
| king in corner + one pawn | .01 | .03 | .03 |
| fianchetto, queenside castle, pawns of both colours | .01 each | .01 each | .01 each |

Plainly: the memory's chunks are pawn structures and castled kings. The big ones are four-piece chunks (of the per-seed top-50, 133 of 200 chunk slots are 4 pieces, 18 are 3, 49 are 2; of the top-200, 577 / 79 / 144), and the most-used are the white kingside castle with its three-pawn shield (f2 g1 g2 h2 and its h3 and g3 variants), the black chains e6 f7 g7 h7 and d5 e6 f7 g7, the a-file rook and queenside pawns (a1 rook behind a2 b2 pawns), and pairs such as g7 h6 and a7 b7. Piece-only groups of 3 or 4 (queen, rooks, minor pieces on their starting squares: a8 rook, d8 queen, e7 bishop, f8 rook) fall in the "mixed" row. Fianchetto bishops, queenside castles and any non-pawn attack patterns are rare (about 1% each of the top lists), in line with the site's line that the chunks are pawn structures, batteries and castled kings, not attacks.
The most-used chunks are on their usual squares (white castled king on g1, black on g8, chains on the kingside and queenside files). White castled-king shields lead (f2 g1 g2 h2 is the top chunk in three seeds).
One seed differs: seed 17 uses f2 g1 g2 h3 only 3 times (against 135, 300, 262), and its top-8 holds a rook-pawn corner structure and "f1 rook, f2 pawn, f3 knight, g1 king" instead of the white shield. Where that chunk is missing, seed 17 uses other white-castle chunks (f1 rook, f2 pawn, f3 knight, g1 king: 75 uses).

## 2. Consistency across seeds and along the stream

Mean over the 6 seed pairs; top-k by use count; "exact" = the same pieces on the same squares; "shape" = the same pieces in the same relative squares (translation-invariant: the chunk moved as a rigid unit by whole files and ranks, same colours).

| positions learned | top-k | exact | shape | random draw from the seed's own chunks (exact) | pair-frequency baseline (exact / shape) | TRELLIS 2-piece chunks only (exact) |
|---|---|---|---|---|---|---|
| 500 | 50 | .201 | .207 | .010 | .725 / .567 | .317 |
| 500 | 200 | .134 | .138 | .039 | .781 / .743 | .276 |
| 1000 | 50 | .252 | .313 | .006 | .859 / .645 | .355 |
| 1000 | 200 | .172 | .174 | .027 | .871 / .797 | .283 |
| 2000 | 50 | .309 | .347 | .004 | .842 / .681 | .392 |
| 2000 | 200 | .219 | .239 | .019 | .903 / .848 | .283 |
| 4000 | 50 | **.307** | **.343** | .004 | .899 / .701 | .451 |
| 4000 | 200 | **.240** | **.265** | .016 | .945 / .853 | .302 |

* The random baseline draws k chunks uniformly from the chunks the seed used at least twice (union pool of all seeds: .005 at 4000, top-50, same order). TRELLIS inventories overlap 60 to 80 times as much as that at top-50 (.307 against .004) and 15 times as much at top-200 (.240 against .016).
* **Translation-invariant matching helps a little** (.343 against .307 at top-50; .265 against .240 at top-200): most top chunks sit on their usual squares and do not move, and a few move (for example the diagonal pawn pair b2 c3 appears at 16 different origin squares in seed 13: d4 x16, a3 x12, c3 x8, b4 x7 and others; the diagonal pair a6 b7 at 17).
* **The frequency baseline overlaps far more than TRELLIS.** The baseline counts every pair of pieces within two squares of each other in each training position and takes the top-k pairs by raw count. Its top-50 has Jaccard .899 (exact) across seeds at 4,000 positions, TRELLIS's .307; compared with TRELLIS's own 2-piece chunks (.451), still far below. The baseline lists are the commonest simple relations of real games (g7 h7, a7 b7, f2 g2, and similar), which every seed sees about equally often. TRELLIS's chunks are larger (most of them are 4 pieces) and more specific, so the same chunk in two seeds needs the same training positions. It is a different inventory, not a worse one for recall, but on this metric it does not agree across seeds more than frequency does. A matching frequency baseline for 3 and 4-piece chunks was not built.
* Count-weighted overlap (sum of the smaller use count over the sum of the larger): exact .141, .179, .206, .229 and shape .183, .216, .241, .262 at 500, 1000, 2000, 4000.
* **Chunks every seed finds**: 9 chunks are in all four per-seed top-50s (exact), 34 in all four top-200s; as shapes, 11 and 37. Of the 5,152 chunks used at least twice by some seed, 3,663 are used twice or more by one seed only, 846 by two, 414 by three, 229 by all four. The nine in every top-50 (use counts for seeds 13 / 17 / 19 / 23):

| chunk | uses | label |
|---|---|---|
| e6 f7 g7 h7 (black) | 94 / 229 / 167 / 174 | pawn chain |
| g7 h6 | 57 / 47 / 92 / 99 | pawn chain |
| e6 f7 g7 h6 | 52 / 103 / 44 / 65 | pawn chain |
| a7 pawn, a8 rook | 52 / 70 / 44 / 34 | corner rook with pawn |
| e6 f7 g6 h7 | 36 / 60 / 49 / 51 | pawn chain |
| a6 b7 | 47 / 32 / 53 / 57 | pawn chain |
| g6 h7 | 38 / 28 / 53 / 44 | pawn chain |
| a7 b6 | 39 / 26 / 39 / 43 | pawn chain |
| a6 b5 | 44 / 28 / 32 / 34 | pawn chain |

All nine are black pawn structures. The white castled-king shield (f2 g1 g2 h2, f2 g1 g2 h3, f2 g1 g3 h2) is the most-used chunk in three seeds but not found by seed 17, so it is not among the nine.

**Along the stream** (within a seed; top-k Jaccard of the checkpoint with the final 4,000-position list; mean over seeds):

| checkpoint | top-50 exact / shape | top-200 exact / shape | pair-frequency baseline top-50 exact | final top-50 already chosen at least once |
|---|---|---|---|---|
| 500 | .367 / .410 | .269 / .293 | .818 | .85 |
| 1000 | .559 / .603 | .399 / .443 | .888 | .96 |
| 2000 | .722 / .759 | .606 / .652 | .896 | 1.00 |

Random reference for top-50 against its own final list: .013 to .023. The chunk ranking stabilises more slowly than the pair-frequency ranking (.367 at 500 against .818) but is far from random from 500 positions on, and the final top-50 chunks have mostly appeared by 1,000 positions (.96 chosen at least once).

## 3. Example boards and their decompositions

Seed 13 at 4,000 positions, parse with learning off, 30 fixed boards (20 middle games and 10 end games). Over the 30 boards, seed 13 recognises 2.93 multi-piece units per board and puts 41% of the pieces into one; 60% of its multi-piece units are stored chunks (the others are recognised from concepts without an exact stored copy). End games have at most one 2 to 4 piece unit each (some none), mostly unstored. Cross-seed agreement of the decompositions (6 pairs, 30 boards): Jaccard of the sets of multi-piece units **.211**, Jaccard of the covered squares **.506**, against .334 for squares covered at random by the same number of pieces. So the seeds agree on where the chunks are twice as often as on the exact units, and the squares agreement is above, but not far above, a random cover.

Diagrams: left, the pieces (white upper case, black lower case); right, the chunk letter on each piece that is in a multi-piece chunk, `o` for a piece in no chunk.

### Showcase (chosen: clean, recognisable decompositions)

**Board 1** (middle game, 24 pieces). Three stored four-piece chunks: white's queenside pawn chain, black's pawn chain, white's castled king with a pushed h-pawn.
```
   8  . . r n r . k .     8  . . o D o . o .
   7  p p . . . . p .     7  B B . . . . o .
   6  . . . q p p Q p     6  . . . D B o o o
   5  . . N p . . . .     5  . . o B . . . .
   4  . . . P . . . .     4  . . . A . . . .
   3  . . P . . . . P     3  . . A . . . . C
   2  P P . . R P P .     2  A A . . o C C .
   1  . . . . R . K .     1  . . . . o . C .
      a b c d e f g h        a b c d e f g h
   A = Pa2 Pb2 Pc3 Pd4 (white queenside pawn chain, stored)
   B = pa7 pb7 pd5 pe6 (black pawn chain, stored)
   C = Pf2 Kg1 Pg2 Ph3 (white castled king, shield, stored)
   D = qd6 nd8 (not stored)
```
**Board 14** (middle game, 26 pieces).
```
   8  . r . . . . k .     8  . C . . . . o .
   7  . . . . q p b .     7  . . . . C A o .
   6  . . n . p . p n     6  . . C . A . A o
   5  . . . p P . N p     5  . . . A o . o o
   4  . p p P . P . .     4  . o o o . B . .
   3  . . P . N . . .     3  . . D . o . . .
   2  . P Q B . . P P     2  . D o o . . B B
   1  R . . . . . K .     1  o . . . . . B .
      a b c d e f g h        a b c d e f g h
   A = pd5 pe6 pf7 pg6 (black pawn chain, stored)
   B = Pf4 Kg1 Pg2 Ph2 (white castled king, shield, stored)
   C = rb8 nc6 qe7 (not stored)
   D = Pb2 Pc3 (white diagonal pawn pair, stored)
```
**Board 6** (middle game, 25 pieces).
```
   8  r . . . r . k .     8  D . . . o . C .
   7  p p . . . p p .     7  D o . . . C C .
   6  . . . p . . b p     6  . . . o . . o C
   5  . . p P . P . .     5  . . o B . o . .
   4  . . P b . . P q     4  . . B o . . o o
   3  P . N . . . . P     3  B . A . . . . o
   2  . P . Q . . . .     2  . B . A . . . .
   1  R . . . . R . K     1  A . . . . A . o
      a b c d e f g h        a b c d e f g h
   A = Ra1 Nc3 Qd2 Rf1 (white pieces, stored)
   B = Pa3 Pb2 Pc4 Pd5 (white pawn chain, stored)
   C = pf7 pg7 kg8 ph6 (black castled king with h-pawn on h6, stored)
   D = pa7 ra8 (rook behind rook-pawn, stored)
```
**Board 17** (middle game, 26 pieces).
```
   8  r . . . . r k .     8  o . . . . o F .
   7  . . q . . p p p     7  . . E . . C C o
   6  p . . b p n . .     6  o . . E C F . .
   5  . . . p . . . .     5  . . . C . . . .
   4  . P p P P . . .     4  . D o B o . . .
   3  P . . N . P . P     3  D . . o . B . o
   2  . B . . Q . P .     2  . A . . A . B .
   1  . . R . R . K .     1  . . A . A . B .
      a b c d e f g h        a b c d e f g h
   A = Bb2 Rc1 Re1 Qe2 (white back pieces, not stored)
   B = Pd4 Pf3 Kg1 Pg2 (white castled king, stored)
   C = pd5 pe6 pf7 pg7 (black pawn chain, stored)
   D = Pa3 Pb4 (white diagonal pawn pair, stored)
   E = qc7 bd6 (not stored)
   F = nf6 kg8 (stored)
```
In all four, the chunks that are pawn structures and castled kings are stored chunks; most of the piece groups of three or four pieces (queen and rooks, knight and queen) are recognised without a stored copy.

### Case studies (the four showcase boards, plus two more)

For each unit: uses per seed at 4,000 positions (13 / 17 / 19 / 23), whether the shape also occurs on other squares, the stream position where seed 13 first chose it, and how many of the four seeds use it at least twice. "Seeds' decomposition" = how many of seed 13's multi-piece units on this board appear identically in each other seed's decomposition of the same board.

**Case 1: board 1.** Seeds' decomposition: seed 17 0 of 4, seed 19 4 of 4, seed 23 3 of 4.
* White castled king f2 g1 g2 h3: uses 135 / 3 / 300 / 262; shape uses 140 / 3 / 304 / 272, on two origin squares in seed 13 (f1 x135, a1 x5: the second is the same shape as a queenside castle, king on b1 with pawns a2 b2 c3); first chosen at stream position 101; used by 4 of 4 seeds (seed 17 only 3 times).
* White chain a2 b2 c3 d4: uses 20 / 18 / 79 / 70; two origin squares (a2 x20, b2 x2); first chosen at 743; 4 of 4.
* Black chain a7 b7 d5 e6: uses 11 / 2 / 10 / 6, one origin square; first chosen at 359; 4 of 4 (seed 17 twice).
* Queen d6 and knight d8: not stored by any seed's top list; seeds 19 and 23 recognise the same pair.
Seed 17 breaks the board differently: b2 pawn with the two rooks and king (e1 e2 g1), the pawns c3 d4 g2 h3 together, and e8 g8 rook and king, none of which is a seed-13 unit.

**Case 2: board 14.** Seeds' decomposition: seed 17 1 of 4, seed 19 2 of 4, seed 23 2 of 4.
* Black chain d5 e6 f7 g6: uses 35 / 2 / 39 / 42, shape uses 40 / 12 / 42 / 54, two origin squares (d5 x35, e5 x5); first at 408; 4 of 4 seeds use it twice or more (seed 17 exactly 2).
* White castled king f4 g1 g2 h2: uses 51 / 0 / 53 / 44; two origin squares (f1 x51, d1 x2); first at 75; 3 of 4.
* White diagonal pair b2 c3: uses 5 / 7 / 5 / 3, but the shape (a diagonal pawn pair of one colour) has 89 / 138 / 76 / 79 uses and appears on 16 different origin squares in seed 13 (d4 x16, a3 x12, c3 x8, b4 x7 and others). Seeds 17, 19 and 23 decompose the same pawns as a larger unit b2 c3 d4 (or b2 c3 d4 e5), so the pair is a translated small chunk and also a part of a bigger one.
* Black rook, knight, queen b8 c6 e7: not stored; seed 17 finds the same three, seed 19 splits it (c6 e7 and b8 g7 g8).
* Seeds 17, 19 and 23 also give the white back pieces a1 c2 d2 e3 (rook, queen, bishop, knight) as a unit, which seed 13 leaves as singletons.

**Case 3: board 6.** Seeds' decomposition: seed 17 0 of 4, seed 19 2 of 4, seed 23 2 of 4.
* Black rook-pawn corner a7 a8: uses 52 / 70 / 44 / 34; shape uses 68 / 96 / 53 / 54 on two origin squares (a7 x52, h7 x16: the mirror h-file rook corner); first at 90; 4 of 4.
* Black castled king f7 g7 g8 h6: uses 46 / 20 / 18 / 0; one origin square; first at 172; 3 of 4.
* White chain a3 b2 c4 d5: uses 2 / 5 / 14 / 6; first at 2,804; 4 of 4. A late chunk.
* White pieces a1 c3 d2 f1: stored but chosen 0 / 0 / 4 / 0 times; seeds 19 and 23 recognise it, 13 and 17 do not as a stored chunk.
Seed 17's decomposition of the same board has a1 a3 b2 c4 (rook plus three pawns) and a7 a8 b7 e8 as larger units: the same structures, grouped differently.

**Case 4: board 17.** Seeds' decomposition: seed 17 0 of 6, seed 19 3 of 6, seed 23 3 of 6.
* Black chain d5 e6 f7 g7: uses 55 / 4 / 80 / 102, shape uses 65 / 12 / 95 / 113 on three origin squares (d5 x55, c5 x6, e5 x4); first at 453; 4 of 4.
* White diagonal pair a3 b4: uses 12 / 16 / 16 / 17, shape 89 / 138 / 76 / 79 on 16 origin squares; first at stream position 20 (the earliest-learned chunk of these cases); 4 of 4.
* White castled king d4 f3 g1 g2: stored, but chosen 0 / 0 / 0 / 3 times; a rare variant. The f3 g2 h3 d4 neighbours (e4 f3 g2 h3) appear in seed 17.
* Black knight and king f6 g8: uses 15 / 0 / 0 / 9, first at 929; 2 of 4.
* Rooks a8 f8, knight f6 and king g8 as one unit: found by seeds 17, 19 and 23 (not by seed 13, which keeps f6 g8 as a pair).

**Case 5: board 9** (24 pieces; 22 of them in six chunks). Seeds' decomposition: seed 17 0 of 6, seeds 19 and 23 3 of 6.
* White castled king f2 g1 g3 h2: uses 94 / 0 / 176 / 186; first at 79; 3 of 4.
* Black a7 c6 d5 chain: uses 2 / 5 / 0 / 0 (late and rare). Black b7 g6 g8 h7: stored, never chosen as a chunk by any seed (a unit recognised by concept, not a common stored shape).
* White a1 c1 e2 f1 (rook, queen, knight, rook) and white pawns a3 b2 c2 e3: not stored by seed 13; seeds 19 and 23 find the same two units.
The decomposition is the largest in the sample, and half of it agrees across two other seeds; none with seed 17.

**Case 6: board 0** (26 pieces; four four-piece units, 16 pieces). Seeds' decomposition: seed 17 1 of 4, seed 19 1 of 4, seed 23 2 of 4.
* Black chain e6 f7 g6 h7: uses 36 / 60 / 49 / 51; shape uses 40 / 64 / 54 / 56, three origin squares (e6 x36, e4 x2, e5 x2); first at 70; 4 of 4 (identical in every seed's decomposition).
* White castled king f3 g2 h1 h2: uses 9 / 0 / 15 / 11, shape 25 / 0 / 25 / 24 on three origin squares (e1 x13, f1 x9, a1 x3); first at 106; 3 of 4.
* Black rook, queen, knight, rook a8 b6 d7 f8; white rooks, knight, queen c1 d3 f1 h6: not stored, found by seed 13 from concepts. Seeds 17, 19 and 23 give c1 e2 f1 h6 or d3 e2 f1 h1 etc.: similar but different groupings of the white pieces.

## 4. What this says, and does not

* The chunks that fall out are pawn structures, castled kings with pawn shields, corner rooks with their pawns, and groups of back-rank pieces. They are four-piece chunks mostly, and sit on their usual squares; the diagonal pawn pair is the one shape that moves (16 origin squares).
* Across seeds and training positions, the chunk inventory agrees at Jaccard .31 (top-50) and .24 (top-200) at 4,000 positions, against .004 and .016 for random chunks; translation-invariant matching adds about .03; it grows with the stream (.20 to .31 over 500 to 4000 for top-50) and the final top chunks have mostly appeared by 1,000 positions.
* A frequency baseline (commonest nearby piece pairs by raw count) agrees across seeds at .90 (top-50) and .95 (top-200). On this metric TRELLIS's inventory is less consistent than the plain frequency of simple relations; its chunks are bigger and more specific, and no matched baseline for 3 to 4 piece chunks was built.
* Board decompositions agree across seeds at a unit-set Jaccard of .21 and a covered-squares Jaccard of .51 (random cover .33): the same pawn structures and castles are found on the same boards by most seeds, but groupings of piece groups and the choice between a pair and a larger chain vary. Seed 17 deviates most.
* Limits: 4,000 positions, not the 27,459 of cs73L, and four seeds; the class names are my rules, not the memory's; the 30 boards are the first 20 middle and 10 end positions of the pools, not random draws, and the six boards of section 3 are picked for clean decompositions (the 30-board agreement numbers are over all of them).

## Rule notes

All training, dumps and analyses ran as batch jobs on compute nodes (`sbatch --requeue`, overcap, standard exclude list, explicit `--mem`). Two trivial login-node python invocations ran earlier in the round (`python -c "print(1)"` and `python3 --version`). A waiter loop (`timeout 6000`) waited for the four training jobs and finished; no jobs or waiters are left running.
