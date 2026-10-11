TinyStories, sentences of 3–5 words over the 100 most frequent words: 2500 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 26.1 |
| word 2-gram model | 13.2 |
| word 3-gram model | 13.0 |
| TRELLIS v2 | 12.0 |

Grammar: 42 categories, 47 chunk types; 100% of training sentences analysed as one tree (1.00 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 35.4% | 39.6% | 57.5% | 23.8% |
| real | 67.3% | 63.1% | 49.0% | 80.6% |
| new and real | 2.7% | 2.7% | 6.5% | 4.4% |
| 3–5 words long | 96.5% | 96.1% | 69.4% | 92.0% |
| real, among those of 3–5 words | 69.1% | 65.1% | 64.3% | 86.4% |
| new and real, among those of 3–5 words | 2.2% | 2.3% | 3.0% | 3.6% |
| word pairs in TinyStories | 95.3% | 94.3% | 100.0% | 100.0% |
| word triples in TinyStories | 83.0% | 78.6% | 85.0% | 100.0% |
| mean length | 4.0 | 4.0 | 4.1 | 4.0 |
| perceived with the analysis it was generated from | 99.7% | 99.5% | – | – |
| chunks inside sentences found in TinyStories | 91.2% | 90.1% | – | – |

Largest categories:

- S15 (2500): they are happy, they were very happy, tim was very happy, he was very happy, they have fun, she was very happy, tim was so happy, he was so happy
- S0 (928): tim was, he was, she was, lily was, sue was, tom was, the bird was, spot was
- S30 (928): was, felt, looked
- S21 (772): tim, he, she, lily, sue, tom, spot, max
- S9 (752): very happy, so happy, very sad, not happy, happy to help, so sad, not fun, so fun
- S35 (752): very, so, not, happy, all
- S34 (739): happy, sad, fun
- S39 (718): fun, together, bird, it, happy, dog, help, day
- S22 (692): they, she, he, it, what, i, tim, tom
- S1 (653): a, have, the, to, to play, you, and, not like
- S10 (649): have fun, to play together, had fun, to help, the bird, together all day, you want to play, a big tree
- S6 (605): it was, they played, it is, what is, he is, they liked, she is, i like

Most frequent chunks inside training analyses: *very happy* (401), *tim was* (192), *they were* (191), *so happy* (190), *they are* (139), *he was* (138), *she was* (119), *very sad* (104), *lily was* (79), *to help* (76), *to play* (73), *the bird* (68), *not happy* (66), *tom was* (65), *sue was* (63), *have fun* (61), *it was* (50), *can i* (48), *the dog* (47), *the cat* (46)


The grammar's own sentences (analysis as generated):

    [[they were] happy]
    [[it was] [a cat]]
    [[he was] [not happy]]
    [[[the boy] looked] [so happy]]  (new)
    [[it looked] happy]
    [[she [wanted [to help]]] him]
    [[[they felt] play] together]  (new)
    [came [the tree]]  (new)
    [[he saw] happy]  (new)
    [[it is] [a cat]]  (new)
    [[sue was] [so happy]]
    [[tim was] happy]
    [[she was] [very [to help]]]  (new)
    [[she was] happy]
    [[[lily [and ben]] are] friends]
    [[they go] [to [the went]]]  (new)
    [[she was] [so happy]]
    [[[can we] go] mom]
    [[they are] happy]
    [[lily is] [a dog]]
    [[tim was] sad]
    [[[the dog] was] [so happy]]
    [[[tim [and sam]] were] [very happy]]  (new)
    [[tom was] happy]
    [[[tom [and max]] were] sad]

All samples, including partial analyses (pieces joined by ·):

    [[they were] happy]
    [[tim like] [a cat]]  (new)
    [[i [little do]] it]  (new)
    [[they were] [very happy]]
    [[lily was] [so happy]]
    [[she [wanted [to play]]] together]
    [[tim [wanted [to help]]] sue]
    [[sue asked] [had [little bird]]]  (new)
    [[tim [wanted [to play]]] too]
    [[lily was] sad]
    [[[tom play] with] you]  (new)
    [[they like] happy]  (new)
    [[she was] sad]
    [[tom was] happy]
    [[he loved] [[to play] together]]  (new)

Held-out sentences (minimum-risk analysis):

    [[she was] [very happy]]
    [[they liked] [it too]]
    [[they do] [[not like] tom]]
    [[i have] [[a new] toy]]
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
