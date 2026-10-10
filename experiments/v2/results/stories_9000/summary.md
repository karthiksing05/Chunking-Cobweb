TinyStories, sentences of 3–5 words over the 100 most frequent words: 9000 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 26.1 |
| word 2-gram model | 12.7 |
| word 3-gram model | 11.5 |
| TRELLIS v2 | 10.7 |

Grammar: 69 categories, 94 chunk types; 100% of training sentences analysed as one tree (1.00 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 22.8% | 21.8% | 56.0% | 20.3% |
| real | 77.4% | 78.9% | 49.6% | 81.1% |
| new and real | 0.2% | 0.7% | 5.6% | 1.4% |
| 3–5 words long | 98.8% | 98.6% | 70.8% | 93.3% |
| real, among those of 3–5 words | 78.2% | 79.5% | 62.4% | 85.5% |
| new and real, among those of 3–5 words | 0.1% | 0.2% | 0.3% | 0.1% |
| word pairs in TinyStories | 97.7% | 97.6% | 100.0% | 100.0% |
| word triples in TinyStories | 91.2% | 90.8% | 85.0% | 100.0% |
| mean length | 4.0 | 4.0 | 3.9 | 3.9 |
| perceived with the analysis it was generated from | 99.8% | 100.0% | – | – |
| chunks inside sentences found in TinyStories | 95.4% | 95.5% | – | – |

Largest categories:

- S48 (9000): they are happy, they were very happy, tim was very happy, he was very happy, he was so happy, they were happy, tim was so happy, she was very happy
- S30 (3441): very happy, sad, so happy, happy, very sad, not happy, so sad, so fun
- S39 (3423): was, felt, looked
- S55 (3404): was very happy, was so happy, was sad, was happy, was very sad, felt sad, was not happy, looked sad
- S3 (2865): tim, he, she, lily, tom, sue, spot, max
- S28 (2651): very, so, not, all, his, new
- S29 (2646): happy, sad, fun, friends, big, friend, little
- S35 (1988): it, help, day, together, friends, happy, fun, bird
- S4 (1630): they, we, tim, i, he, and, spot
- S17 (1498): were, are, all, felt, make
- S16 (1363): he, she, tim, they, lily, do, tom, her
- S1 (1218): it, happy, fun, that, be friends, not happy, sad, help you

Most frequent chunks inside training analyses: *very happy* (1384), *was very happy* (982), *so happy* (706), *they were* (654), *was sad* (628), *was so happy* (627), *was happy* (464), *they are* (445), *very sad* (379), *was very sad* (303), *not happy* (247), *the bird* (242), *wanted to* (218), *can i* (183), *to play* (175), *have fun* (174), *felt sad* (173), *it was* (163), *wanted to help* (156), *was not happy* (152)


The grammar's own sentences (analysis as generated):

    [[[can i] play] too]
    [[the bird] [was sad]]
    [[they [[wanted to] help]] tim]  (new)
    [sam [[[and tom] are] friends]]
    [sue [was [so happy]]]
    [tim [was [very happy]]]
    [tom [[[and sue] are] friends]]
    [[sue [was happy]] too]
    [[they were] [so happy]]
    [[max [and tom]] [like [a cat]]]  (new)
    [[[can i] [do ben]] you]  (new)
    [but [she [was [very sad]]]]  (new)
    [sam [was [not happy]]]
    [they [have fun]]
    [[they are] [very happy]]
    [[they were] [very happy]]
    [tim [was [very happy]]]
    [tim [was [very happy]]]
    [[she [was happy]] [to help]]
    [[he is] sad]
    [[do [have fun]] together]  (new)
    [[he [was happy]] too]
    [[[can i] [play with]] you]
    [[they are] friends]
    [[it is] sad]  (new)

All samples, including partial analyses (pieces joined by ·):

    [[they were] friends]
    [lily [was sad]]
    [she [felt [very sad]]]
    [[[his friends] play] asked]  (new)
    [[they wanted] [the toy]]  (new)
    [[what is] it]
    [[the cat] [was [very happy]]]
    [[they are] [very happy]]
    [[tom asked] sam]
    [[they are] friends]
    [[they were] [very happy]]
    [she [was [very happy]]]
    [he [felt sad]]
    [[they [[want to] play]] too]  (new)
    [[what is] it]

Held-out sentences (minimum-risk analysis):

    [she [was [very happy]]]
    [they [liked [it too]]]
    [[they [do not]] [like tom]]
    [[i have] [a [new toy]]]
    [[they [are not]] [a toy]]
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
