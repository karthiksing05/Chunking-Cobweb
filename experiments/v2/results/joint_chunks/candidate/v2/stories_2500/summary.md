TinyStories, sentences of 3–5 words over the 100 most frequent words: 2500 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 26.1 |
| word 2-gram model | 13.2 |
| word 3-gram model | 13.0 |
| TRELLIS v2 | 11.7 |

Grammar: 37 categories, 48 chunk types; 100% of training sentences analysed as one tree (1.00 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 36.4% | 31.3% | 57.8% | 24.4% |
| real | 67.8% | 71.2% | 48.6% | 80.2% |
| new and real | 4.2% | 2.5% | 6.4% | 4.6% |
| 3–5 words long | 97.2% | 97.0% | 68.6% | 91.5% |
| real, among those of 3–5 words | 69.5% | 73.2% | 64.6% | 86.3% |
| new and real, among those of 3–5 words | 4.1% | 2.4% | 3.1% | 3.7% |
| word pairs in TinyStories | 95.3% | 96.4% | 100.0% | 100.0% |
| word triples in TinyStories | 81.9% | 85.0% | 84.6% | 100.0% |
| mean length | 4.0 | 4.0 | 4.1 | 4.0 |
| perceived with the analysis it was generated from | 99.4% | 99.4% | – | – |
| chunks inside sentences found in TinyStories | 89.8% | 92.2% | – | – |

Largest categories:

- S27 (2500): they are happy, they were very happy, tim was very happy, he was very happy, they have fun, she was very happy, tim was so happy, he was so happy
- S16 (1141): a, the, to, happy, not, you, to play, like
- S17 (1135): too, together, happy, fun, it, friends, that, bird
- S18 (932): was, felt, looked
- S31 (932): was very happy, was so happy, was sad, was happy, was very sad, was not happy, felt sad, looked sad
- S3 (831): they were, what, they played, it was, said, but, we can, can we
- S32 (814): to play together, happy too, to help, be friends, the bird, together all day, not happy, to play too
- S2 (778): tim, he, she, lily, sue, tom, spot, max
- S15 (724): very happy, so happy, very sad, not happy, happy to help, so sad, tom and lily, fun
- S5 (720): very, so, not, happy
- S4 (708): happy, sad
- S10 (681): can, she, he, it, i, we, tim, tom

Most frequent chunks inside training analyses: *very happy* (390), *was very happy* (275), *so happy* (189), *was sad* (162), *was so happy* (161), *was happy* (112), *very sad* (103), *are happy* (94), *was very sad* (82), *to help* (76), *were very happy* (75), *to play* (73), *have fun* (69), *the bird* (68), *are friends* (67), *not happy* (60), *the dog* (47), *the cat* (46), *were happy* (46), *were sad* (46)


The grammar's own sentences (analysis as generated):

    [[his [we [play help]]] it]  (new)
    [he [was [so happy]]]
    [max [was happy]]
    [she [felt [very sad]]]
    [they [had [so happy]]]  (new)
    [[she asked] lily]  (new)
    [[tim did] [not friend]]  (new)
    [she [felt [very sad]]]
    [they [are happy]]
    [she [was [very [to help]]]]  (new)
    [tim [felt [very happy]]]  (new)
    [[he is] [a up]]  (new)
    [[he saw] [a cat]]  (new)
    [[max is] asked]  (new)
    [they [were [very happy]]]
    [she [felt [very happy]]]
    [[she said] [and played]]  (new)
    [they [are [very happy]]]
    [[lily [and ben]] [are friends]]
    [[she asked] [the boy]]  (new)
    [[can i] [have [could happy]]]  (new)
    [[they were] [happy together]]
    [they [were sad]]
    [he [was [very happy]]]
    [they [are [very happy]]]

All samples, including partial analyses (pieces joined by ·):

    [sue [was sad]]
    [they [are [very happy]]]
    [she [was [not happy]]]
    [they [are happy]]
    [go [and [they did]]]  (new)
    [so [they [all [[had fun] together]]]]  (new)
    [and [it together]]  (new)
    [[spot [wanted [to help]]] it]  (new)
    [he [was [very happy]]]
    [tim [was sad]]
    [tim [was happy]]
    [[it [want [to help]]] sue]  (new)
    [tim [was [very sad]]]
    [tim [was [very happy]]]
    [[the cat] [looked sad]]

Held-out sentences (minimum-risk analysis):

    [she [was [very happy]]]
    [[they liked] [it too]]
    [[they do] [not [like tom]]]
    [[i have] [a [new toy]]]
    [[they are] [not [a toy]]]
    [lily [was [not happy]]]
    [they [are friends]]
    [she [was [so happy]]]
    [they [have fun]]
    [they [are happy]]
    [and [[they were] [very happy]]]
    [[they played] [together [all day]]]
    [they [have fun]]
    [sue [was sad]]
    [[they played] [all day]]
