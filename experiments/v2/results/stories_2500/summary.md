TinyStories, sentences of 3–5 words over the 100 most frequent words: 2500 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 26.1 |
| word 2-gram model | 13.2 |
| word 3-gram model | 13.0 |
| TRELLIS v2 | 16.0 |

Grammar: 34 categories, 50 chunk types; 65% of training sentences analysed as one tree (1.47 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 33.3% | 56.9% | 58.4% | 23.1% |
| real | 71.4% | 46.9% | 48.1% | 81.0% |
| new and real | 4.7% | 3.8% | 6.5% | 4.1% |
| 3–5 words long | 94.3% | 85.2% | 70.5% | 92.8% |
| real, among those of 3–5 words | 75.3% | 54.8% | 63.0% | 86.3% |
| new and real, among those of 3–5 words | 4.6% | 4.2% | 4.0% | 3.4% |
| word pairs in TinyStories | 96.2% | 85.1% | 100.0% | 100.0% |
| word triples in TinyStories | 84.5% | 54.4% | 84.2% | 100.0% |
| mean length | 3.8 | 4.0 | 4.0 | 4.0 |
| perceived with the analysis it was generated from | 100.0% | 98.7% | – | – |
| chunks inside sentences found in TinyStories | 92.2% | 88.3% | – | – |

Largest categories:

- S6 (2046): it, together, too, you, said, friends, that, is
- S7 (1619): they are happy, they were very happy, tim was very happy, he was very happy, they have fun, she was very happy, tim was so happy, he was so happy
- S8 (1151): they, the, you, a, he, she, to, i
- S15 (1151): play, do, did, like, was, happy, mom, with
- S1 (1097): tim was, he was, she was, lily was, sue was, tom was, the bird was, spot was
- S11 (1087): tim, he, she, lily, tom, sue, the bird, it
- S3 (1086): very happy, sad, so happy, happy, very sad, not happy, fun, said
- S12 (1086): was, is, felt, wanted to help, looked, asked, wanted to play, friends were
- S24 (770): very, so, not, happy, a, all, at
- S23 (758): happy, sad, fun, dog, cat, very, tom, big
- S9 (430): they, lily and tom, lily and ben, tim and sam, tim and sue, tom and sam, tom and lily, ben and sam
- S22 (430): were, are, felt, bird, cat

Most frequent chunks inside training analyses: *very happy* (401), *tim was* (192), *they were* (192), *so happy* (190), *they are* (137), *he was* (137), *she was* (119), *very sad* (104), *lily was* (79), *to help* (76), *have fun* (71), *tom was* (65), *not happy* (64), *sue was* (63), *can i* (54), *the bird* (49), *wanted to help* (49), *had fun* (45), *to play* (41), *can we* (40)


The grammar's own sentences (analysis as generated):

    [[[tom [and tom]] were] happy]  (new)
    [[she was] [so happy]]
    [[lily was] [very happy]]
    [i [had fun]]  (new)
    [[they are] happy]
    [[they were] [so happy]]
    [[tim was] [so [to play]]]  (new)
    [[he was] [so happy]]
    [[spot was] said]  (new)
    [[he was] [very sad]]
    [[sam saw] sad]  (new)
    [[he was] happy]
    [[lily was] sad]
    [[lily was] [so sad]]
    [[tim was] [very fun]]  (new)
    [[tim was] [very happy]]
    [they [have fun]]
    [[tim was] [happy happy]]  (new)
    [[lily was] happy]
    [[tim was] sad]
    [[they were] happy]
    [[she was] [all happy]]  (new)
    [[[the [little bird]] was] sad]
    [[lily was] sad]
    [[tim was] [so happy]]

All samples, including partial analyses (pieces joined by ·):

    [[they are] happy]
    [[[the [have happy]] was] sam]  (new)
    [[[the bird] was] happy]
    [she go] · [i [friend tree]]  (new)
    [i play] · [she liked]  (new)
    [[she was] [very happy]]
    friends · [she go] · [to [very tree]]  (new)
    [[they were] sad]
    [[[lily [and tom]] were] [very fun]]  (new)
    [they fun] · [she little] · [to help]  (new)
    [[he was] [very fun]]  (new)
    it · you  (new)
    [[i are] do] · [[can are] liked]  (new)
    together · [mom [like to]]  (new)
    [[she was] [very happy]]

Held-out sentences (minimum-risk analysis):

    [[she was] [very happy]]
    [[they liked] [it too]]
    [[they do] [[not like] tom]]
    [[i [have a]] [new toy]]
    [[they are] [not [a toy]]]
    [[lily was] [not happy]]
    [[they are] friends]
    [[she was] [so happy]]
    [they [have fun]]
    [[they are] happy]
    [and [[they were] [very happy]]]
    [[[they played] together] [all day]]
    [they [have fun]]
    [[sue was] sad]
    [[[they played] all] day]
