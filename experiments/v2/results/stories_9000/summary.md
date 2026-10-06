TinyStories, sentences of 3–5 words over the 100 most frequent words: 9000 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 26.1 |
| word 2-gram model | 12.7 |
| word 3-gram model | 11.5 |
| TRELLIS v2 | 11.0 |

Grammar: 68 categories, 79 chunk types; 66% of training sentences analysed as one tree (1.48 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 7.9% | 28.4% | 57.3% | 21.8% |
| real | 92.6% | 71.9% | 48.8% | 79.6% |
| new and real | 0.5% | 0.3% | 6.1% | 1.4% |
| 3–5 words long | 99.3% | 92.6% | 69.5% | 92.1% |
| real, among those of 3–5 words | 93.0% | 77.4% | 61.7% | 84.9% |
| new and real, among those of 3–5 words | 0.2% | 0.1% | 0.3% | 0.0% |
| word pairs in TinyStories | 99.2% | 98.3% | 100.0% | 100.0% |
| word triples in TinyStories | 96.1% | 89.4% | 84.5% | 100.0% |
| mean length | 3.8 | 4.0 | 4.0 | 3.9 |
| perceived with the analysis it was generated from | 99.7% | 99.2% | – | – |
| chunks inside sentences found in TinyStories | 97.8% | 97.5% | – | – |

Largest categories:

- S4 (7400): too, it was, that, said, can we, what, they played, do you
- S3 (5949): they are happy, they were very happy, tim was very happy, he was very happy, he was so happy, they were happy, tim was so happy, she was very happy
- S14 (3688): tim, he, she, lily, tom, sue, the bird, the cat
- S20 (3681): was very happy, was so happy, was sad, was happy, was very sad, felt sad, was not happy, looked sad
- S1 (2999): very happy, so happy, sad, happy, very sad, not happy
- S50 (2974): was, felt
- S62 (2648): very, so, not, help, was, play
- S51 (2625): happy, sad, big
- S12 (2016): to, a, be, do, like, play, her, the
- S35 (1935): it, together, play, fun, happy, friends, friend, you
- S27 (1303): they, are
- S42 (977): are, were, to

Most frequent chunks inside training analyses: *very happy* (1376), *was very happy* (981), *so happy* (706), *they were* (655), *was so happy* (626), *was sad* (626), *was happy* (467), *they are* (454), *very sad* (379), *was very sad* (290), *wanted to* (284), *not happy* (247), *the bird* (226), *wanted to help* (200), *can i* (182), *felt sad* (181), *have fun* (142), *play with* (135), *to play* (127), *was not happy* (126)


The grammar's own sentences (analysis as generated):

    [[[can were] [play with]] it]  (new)
    [tim [was happy]]
    [[they [[wanted [little boy]] at]] mom]  (new)
    [[they are] sad]
    [[they are] happy]
    [[they were] [very sad]]
    [[can [[saw to] help]] sue]  (new)
    [tim [[[and sam] were] sad]]
    [[the bird] [looked sad]]
    [[[they are] were] asked]  (new)
    [tim [[[and sam] are] friends]]
    [[the [little girl]] [was sad]]
    [asked [the bird]]
    [tim [[[and sam] were] sad]]
    [they [have fun]]
    [he [was [so happy]]]
    [[they [little help]] it]  (new)
    [tim [was [very happy]]]
    [tim [was [so happy]]]
    [he [was [very sad]]]
    [[the bird] [[wanted to] help]]
    [she [was [very happy]]]
    [[the [little girl]] [was sad]]
    [[[they are] all] friends]
    [[she liked] it]

All samples, including partial analyses (pieces joined by ·):

    [it is] · big
    [she [was sad]]
    [[they are] happy]
    [the cat] · [not a] · ball  (new)
    [spot [was [very happy]]]
    [and so] · [they would]  (new)
    [they played] · [all day] · [[can i] do]  (new)
    [do you] · want  (new)
    [it was] · a  (new)
    [can we] · do  (new)
    [they [[wanted to] play]]
    [she [[to play] girl]] · asked  (new)
    [she said] · [she found]  (new)
    [they [have fun]]
    [[lily [[wanted to] help]] sam]

Held-out sentences (minimum-risk analysis):

    [she [was [very happy]]]
    [[they [liked it]] too]
    [[they [do not]] [like tom]]
    [[[i have] a] [new toy]]
    [[[[they are] not] a] toy]
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
