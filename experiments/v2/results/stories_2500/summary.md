TinyStories, sentences of 3–5 words over the 100 most frequent words: 2500 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 26.1 |
| word 2-gram model | 13.2 |
| word 3-gram model | 13.0 |
| TRELLIS v2 | 11.8 |

Grammar: 42 categories, 49 chunk types; 100% of training sentences analysed as one tree (1.00 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 38.0% | 35.8% | 57.5% | 24.4% |
| real | 65.9% | 66.7% | 49.1% | 80.5% |
| new and real | 3.9% | 2.5% | 6.6% | 4.9% |
| 3–5 words long | 94.6% | 95.2% | 68.9% | 91.4% |
| real, among those of 3–5 words | 69.0% | 69.5% | 65.0% | 86.7% |
| new and real, among those of 3–5 words | 3.5% | 2.1% | 3.3% | 3.9% |
| word pairs in TinyStories | 94.7% | 94.8% | 100.0% | 100.0% |
| word triples in TinyStories | 80.2% | 81.5% | 84.8% | 100.0% |
| mean length | 4.0 | 4.0 | 4.0 | 4.0 |
| perceived with the analysis it was generated from | 99.2% | 98.9% | – | – |
| chunks inside sentences found in TinyStories | 89.8% | 90.9% | – | – |

Largest categories:

- S14 (2500): they are happy, they were very happy, tim was very happy, he was very happy, they have fun, she was very happy, tim was so happy, he was so happy
- S1 (1039): the, a, have, is, to, very, you, had
- S9 (1025): very happy, sad, so happy, happy, very sad, not happy, so sad, very very sad
- S11 (1024): fun, happy, it, together, sad, help, bird, dog
- S4 (1022): tim was, he was, she was, they were, lily was, tom was, sue was, the bird was
- S28 (860): they, she, he, it, tim, i, the, we
- S0 (843): tim, he, she, lily, tom, sue, the bird, spot
- S33 (843): was
- S8 (815): have fun, to play together, to help, had fun, the bird, together all day, not happy, very happy
- S15 (697): very, so, not, all
- S16 (696): happy, sad, fun
- S5 (689): it was, they played, they liked, she is, he is, tim felt, we can, it is

Most frequent chunks inside training analyses: *very happy* (401), *tim was* (192), *they were* (192), *so happy* (190), *they are* (139), *he was* (138), *she was* (119), *very sad* (104), *lily was* (79), *to help* (76), *to play* (73), *the bird* (68), *tom was* (65), *not happy* (64), *sue was* (63), *have fun* (60), *can i* (54), *it was* (50), *the dog* (47), *the cat* (46)


The grammar's own sentences (analysis as generated):

    [[[can i] have] [the bird]]  (new)
    [[they were] happy]
    [[they [want to]] [have it]]  (new)
    [[you did] [[not like] that]]  (new)
    [[they felt] [so happy]]
    [[lily was] sad]
    [what [do [i do]]]
    [[spot was] [very happy]]
    [[she was] [very sad]]
    [[they are] happy]
    [[[[i can] play] [and sue]] said]  (new)
    [[she is] [not friends]]  (new)
    [[they said] asked]  (new)
    [[they [had fun]] [[play have] [to [to play]]]]  (new)
    [[they were] friends]
    [[you did] [[not [and happy]] back]]  (new)
    [[tim was] sad]
    [[they were] happy]
    [[he was] [very sad]]
    [[sue was] [very happy]]
    [[they [have fun]] together]
    [[tim was] [very happy]]
    [[he was] [so happy]]
    [[[[can we] all] with] you]  (new)
    [[she felt] happy]

All samples, including partial analyses (pieces joined by ·):

    [[she looked] sad]
    [[[they were] happy] too]
    [[[tim [and sam]] were] sad]
    [[and cat] [their ben]]  (new)
    [[lily was] [very sad]]
    [[she was] sad]
    [[[her mom] was] [not happy]]
    [[i saw] [a dog]]
    [[they are] happy]
    [[tim was] [very happy]]
    [[they were] friends]
    [[he was] [so happy]]
    [[they are] [very happy]]
    [[they wanted] you]  (new)
    [[he is] [very happy]]

Held-out sentences (minimum-risk analysis):

    [[she was] [very happy]]
    [[they liked] [it too]]
    [[they do] [[not like] tom]]
    [[i have] [a [new toy]]]
    [[they are] [[not a] toy]]
    [[lily was] [not happy]]
    [[they are] friends]
    [[she was] [so happy]]
    [they [have fun]]
    [[they are] happy]
    [and [[they were] [very happy]]]
    [[they played] [together [all day]]]
    [they [have fun]]
    [[sue was] sad]
    [[they played] [all day]]
