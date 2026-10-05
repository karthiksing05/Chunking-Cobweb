TinyStories, sentences of 3–5 words over the 100 most frequent words: 5000 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 26.1 |
| word 2-gram model | 12.8 |
| word 3-gram model | 12.0 |
| TRELLIS v2 | 15.0 |

Grammar: 58 categories, 87 chunk types; 73% of training sentences analysed as one tree (1.33 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 39.3% | 57.1% | 53.6% | 21.1% |
| real | 61.9% | 44.3% | 52.4% | 81.4% |
| new and real | 1.2% | 1.4% | 6.0% | 2.5% |
| 3–5 words long | 97.3% | 89.9% | 72.9% | 92.3% |
| real, among those of 3–5 words | 63.6% | 49.2% | 64.7% | 86.7% |
| new and real, among those of 3–5 words | 1.2% | 1.4% | 1.1% | 1.2% |
| word pairs in TinyStories | 90.9% | 83.6% | 100.0% | 100.0% |
| word triples in TinyStories | 75.5% | 55.1% | 84.6% | 100.0% |
| mean length | 3.9 | 4.0 | 4.0 | 4.0 |
| perceived with the analysis it was generated from | 99.6% | 98.9% | – | – |
| chunks inside sentences found in TinyStories | 86.9% | 82.2% | – | – |

Largest categories:

- S20 (3673): they are happy, they were very happy, tim was very happy, he was very happy, tim was so happy, he was so happy, she was very happy, they were happy
- S21 (2990): you, do, that, it was, said, they played, too, but
- S7 (1728): very happy, so happy, sad, happy, very sad, not happy, so sad, so fun
- S27 (1722): was, felt, looked, had
- S6 (1713): was very happy, was so happy, was sad, was happy, was very sad, was not happy, felt very sad, was so sad
- S46 (1478): they, she, he, it, i, tim, what, tom
- S26 (1436): tim, he, she, lily, tom, sue, spot, max
- S29 (1197): very, so, not, a, big, his, their, named
- S28 (1194): happy, sad, fun, dog, friend, toy, mom, max
- S17 (867): was, saw, is, played, can, asked, did not, said
- S16 (843): a, the, like, and, together, her, his, their
- S15 (834): it, ball, mom, toy, dog, all day, had fun, friend

Most frequent chunks inside training analyses: *very happy* (777), *was very happy* (542), *so happy* (384), *they were* (377), *was sad* (338), *was so happy* (335), *was happy* (270), *they are* (260), *very sad* (222), *was very sad* (178), *to help* (162), *to play* (139), *the bird* (132), *not happy* (128), *wanted to help* (116), *can i* (108), *felt sad* (104), *have fun* (81), *had fun* (79), *was not happy* (78)


The grammar's own sentences (analysis as generated):

    [[[they friends] is] [very happy]]  (new)
    [[the bird] [[liked all] together]]  (new)
    [[it [little tree]] fun]  (new)
    [[they are] sad]
    [max [looked sad]]  (new)
    [[he play] it]  (new)
    [she [was [so dog]]]  (new)
    [tom [liked [to help]]]  (new)
    [[they are] happy]
    [lily [the bird]]  (new)
    [they [[[and ben] are] happy]]  (new)
    [[the dog] [was [so toy]]]  (new)
    [he [was [very sad]]]
    [[her all] asked]  (new)
    [[they were] happy]
    [he [felt sad]]
    [[he [wanted [to play]]] friends]  (new)
    [[the dog] too]  (new)
    [[they are] help]  (new)
    [tim [was [so happy]]]
    [she [was [so happy]]]
    [he [the bird]]  (new)
    [we [[[and lily] are] happy]]  (new)
    [tim [was sad]]
    [[they were] [to happy]]  (new)

All samples, including partial analyses (pieces joined by ·):

    [her mom] · [[[his friend] play] with] · [want [little tree]]  (new)
    [we [did ball]]  (new)
    [new fun] · [you [looked play]]  (new)
    [lily [[[and sam] were] sad]]  (new)
    [[her is] it]  (new)
    lily · and  (new)
    [[[his were] [was happy]] fun]  (new)
    [they [have fun]]
    [[[they were] [liked [had help]]] sad]  (new)
    [tim saw] · [it played] · [i played] · [he saw]  (new)
    [he [was [so sad]]]
    [[they is] sue]  (new)
    [[the played] [was happy]]  (new)
    [sam is] · [it is]  (new)
    [he [was not]] · [a mom]  (new)

Held-out sentences (minimum-risk analysis):

    [she [was [very happy]]]
    [[[they liked] it] too]
    [[[they do] not] [like tom]]
    [[i have] [a [new toy]]]
    [[[they are] not] [a toy]]
    [lily [was [not happy]]]
    [[they are] friends]
    [she [was [so happy]]]
    [they [have fun]]
    [[they are] happy]
    [and [[they were] [very happy]]]
    [[they played] [together [all day]]]
    [they [have fun]]
    [sue [was sad]]
    [[they played] [all day]]
