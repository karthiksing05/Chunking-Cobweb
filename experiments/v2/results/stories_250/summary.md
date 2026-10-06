TinyStories, sentences of 3–8 words over the 250 most frequent words: 2500 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 40.8 |
| word 2-gram model | 25.8 |
| word 3-gram model | 29.0 |
| TRELLIS v2 | 24.9 |

Grammar: 30 categories, 43 chunk types; 10% of training sentences analysed as one tree (3.70 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 25.9% | 88.8% | 87.4% | 65.3% |
| real | 88.2% | 19.8% | 24.3% | 47.8% |
| new and real | 14.1% | 8.6% | 11.7% | 13.1% |
| 3–8 words long | 99.9% | 73.3% | 67.4% | 85.4% |
| real, among those of 3–8 words | 88.3% | 24.1% | 27.4% | 54.2% |
| new and real, among those of 3–8 words | 14.1% | 8.9% | 8.8% | 13.6% |
| word pairs in TinyStories | 98.4% | 89.6% | 100.0% | 100.0% |
| word triples in TinyStories | 95.9% | 65.2% | 76.4% | 100.0% |
| mean length | 3.8 | 5.4 | 5.4 | 5.4 |
| perceived with the analysis it was generated from | 99.6% | 95.9% | – | – |
| chunks inside sentences found in TinyStories | 99.6% | 97.0% | – | – |

Largest categories:

- S0 (9014): they, and, it, you, i, together, fun, play
- S9 (1768): was, cat, big, you, mom, bird, not, ball
- S14 (1574): the, a, his, thank, did, her, so, was
- S20 (824): tim, he, she, tom, lily, it, sue, then
- S5 (594): and, said, was, wanted to, loved to, was happy, saw, asked
- S23 (416): happy, sad, surprised, excited, scared, pretty, nice, kind
- S15 (376): to
- S24 (319): was, very
- S22 (312): very, so, little, old
- S2 (290): very happy, so happy, very sad, very surprised, very excited, so excited, so surprised, so pretty
- S13 (268): want, wanted, liked, like, went, have, what, back
- S3 (254): then something unexpected happened, tim was very happy, he was very happy, tim was so happy, tim was very surprised, she was very happy, he was so happy, tom was sad

Most frequent chunks inside training analyses: *very happy* (116), *so happy* (73), *something unexpected* (73), *something unexpected happened* (73), *was very happy* (71), *wanted to* (64), *an idea* (64), *was so happy* (58), *was happy* (43), *was sad* (42), *very sad* (39), *a big* (28), *loved to* (27), *was very sad* (25), *very surprised* (20), *little girl* (17), *was very surprised* (17), *was scared* (16), *little boy* (13), *liked to* (12)


The grammar's own sentences (analysis as generated):

    [down [[something unexpected] happened]]  (new)
    [it [was [very sad]]]  (new)
    [lily [was [very happy]]]
    [she [was [so happy]]]
    [tim [was [so happy]]]
    [do [was [very excited]]]  (new)
    [then [[something unexpected] happened]]
    [amy [was sad]]
    [lucy [was [so happy]]]
    [she [was sad]]
    [tim [was [so happy]]]
    [lily [was happy]]
    [tim [was [very happy]]]
    [she [was [very happy]]]
    [sue [was [very happy]]]
    [tim [was [so happy]]]
    [it [was [very happy]]]  (new)
    [tim [was [so happy]]]
    [lily [was [very happy]]]
    [lily [was [very happy]]]
    [then [[something unexpected] happened]]
    [tom [was [very happy]]]
    [ben [was happy]]  (new)
    [tom [was [very happy]]]
    [lily [was [very surprised]]]

All samples, including partial analyses (pieces joined by ·):

    [he [was surprised]] · but · [lots girl] · asked  (new)
    tim · laughed · [the [little girl]]  (new)
    yes · you · are  (new)
    [she [was [so pretty]]] · too · more · in · [the park] · now  (new)
    they · [like to] · share · them · it  (new)
    [he [loved to]] · play · together  (new)
    [but said] · some  (new)
    [he [wanted to]] · make  (new)
    [the cat] · [wanted to]  (new)
    [then [[something unexpected] happened]]
    you · are · [so much] · fun · together · [thank you]  (new)
    [but and] · [had [an idea]] · what · is · it · [she was] · proud · happy · and · share  (new)
    [[some were] cat] · looked  (new)
    [tim said] · let's · [fun with]  (new)
    they · played · and · scared  (new)

Held-out sentences (minimum-risk analysis):

    [[[she was] [playing with]] him]
    [but [then [[something unexpected] happened]]]
    [[[can you] [help me]] [find it]]
    [[[amy [said to]] [tom look]] [at [my toy]]]
    [[[he [wanted to]] know] [[what was] [in it]]]
    [[[they played] together] [[all day] long]]
    [[she thought] [[[it was] too] hard]]
    [[tom liked] [the idea]]
    [tim [was [so happy]]]
    [[[they laughed] [[and had] lots]] [of fun]]
    [[[soon [[he was] home]] [and he]] [went inside]]
    [but [then [[something unexpected] happened]]]
    [but [[[then he] found] [[a big] box]]]
    [[i [am a]] [nice dog]]
    [[[it [wanted to]] be] [[friends with] bob]]
