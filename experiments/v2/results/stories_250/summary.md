TinyStories, sentences of 3–8 words over the 250 most frequent words: 2500 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 40.8 |
| word 2-gram model | 25.8 |
| word 3-gram model | 29.0 |
| TRELLIS v2 | 26.1 |

Grammar: 34 categories, 45 chunk types; 10% of training sentences analysed as one tree (3.71 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 36.0% | 92.9% | 88.2% | 64.7% |
| real | 80.6% | 13.0% | 23.6% | 48.8% |
| new and real | 16.6% | 5.9% | 11.8% | 13.5% |
| 3–8 words long | 100.0% | 72.7% | 67.1% | 85.8% |
| real, among those of 3–8 words | 80.6% | 15.1% | 26.4% | 55.1% |
| new and real, among those of 3–8 words | 16.6% | 5.4% | 8.8% | 14.0% |
| word pairs in TinyStories | 97.0% | 91.3% | 100.0% | 100.0% |
| word triples in TinyStories | 90.9% | 60.4% | 76.2% | 100.0% |
| mean length | 3.8 | 5.4 | 5.3 | 5.5 |
| perceived with the analysis it was generated from | 99.5% | 95.9% | – | – |
| chunks inside sentences found in TinyStories | 97.8% | 96.0% | – | – |

Largest categories:

- S0 (9023): they, and, it, you, i, together, fun, are
- S8 (1612): the, his, a, it, thank, they, he, her
- S10 (1594): was, had, cat, mom, you, ball, dog, big
- S32 (807): tim, he, she, tom, lily, sue, then, max
- S6 (590): and, said, wanted to, loved to, saw, was happy, had, asked
- S12 (453): happy, sad, surprised, happened, excited, scared, pretty, kind
- S18 (368): to, big
- S2 (311): very happy, so happy, very sad, very surprised, very excited, so excited, so surprised, very kind
- S11 (311): very, so
- S13 (308): was, very
- S28 (262): want, wanted, liked, like, went, have, what, back
- S3 (255): then something unexpected happened, tim was very happy, he was very happy, tim was so happy, tim was very surprised, she was very happy, he was so happy, tom was sad

Most frequent chunks inside training analyses: *very happy* (116), *so happy* (73), *something unexpected* (73), *something unexpected happened* (73), *was very happy* (71), *wanted to* (66), *an idea* (64), *was so happy* (58), *was sad* (42), *was happy* (42), *very sad* (39), *loved to* (27), *was very sad* (25), *a big* (25), *very surprised* (20), *little girl* (17), *was very surprised* (17), *was scared* (16), *little boy* (13), *the dog* (12)


The grammar's own sentences (analysis as generated):

    [down [was [very kind]]]  (new)
    [tim [was [very scared]]]  (new)
    [lily [was [very happy]]]
    [tom [was [so happy]]]
    [she [was [so happy]]]
    [[the car] [[something unexpected] happened]]  (new)
    [lily [was [very happy]]]
    [tim [was [very happy]]]
    [she [was [very kind]]]  (new)
    [tim [was [very happy]]]
    [tim [was [very happy]]]
    [sam [was [very happy]]]  (new)
    [she [was [very surprised]]]
    [[they could] [was [so happy]]]  (new)
    [tom [was scared]]  (new)
    [tim [was [very nice]]]  (new)
    [then [[something unexpected] happened]]
    [sara [was [very sad]]]  (new)
    [she [was [very happy]]]
    [tim [was [very happy]]]
    [then [[something unexpected] happened]]
    [he [was happy]]
    [mia [was [very happy]]]
    [lily [was happy]]
    [she [was [so happy]]]

All samples, including partial analyses (pieces joined by ·):

    [lucy asked] · max  (new)
    [they had] · down · and · play · together · all · for · [the cat] · [the car] · in  (new)
    i · will · help · [his mom] · [had [an idea]] · [the park] · day · together · [mia boy]  (new)
    can · you  (new)
    [she lots] · of  (new)
    it · will · sorry · for  (new)
    [she liked] · [his car] · [felt much] · fun · anymore · [very liked] · [a big] · [red car]  (new)
    he · felt  (new)
    [sue said] · [thank you] · [want to] · play · bear · said · [thank you] · feel · he  (new)
    [she [was [very sad]]]
    what · [did not] · want · her  (new)
    you · are · happy · to · [play with] · it  (new)
    [but liked] · [each other] · [tom said]  (new)
    but · tim's · looked · surprised · happy  (new)
    they · laughed · together · and · sad · anymore · we  (new)

Held-out sentences (minimum-risk analysis):

    [[[she was] [playing with]] him]
    [but [then [[something unexpected] happened]]]
    [[[can you] [help me]] [find it]]
    [[amy [said to]] [[[tom look] at] [my toy]]]
    [[[he [wanted to]] know] [[what was] [in it]]]
    [[[they played] together] [[all day] long]]
    [[she thought] [[it was] [too hard]]]
    [[tom liked] [the idea]]
    [tim [was [so happy]]]
    [[[they laughed] [[and had] lots]] [of fun]]
    [[[soon [he was]] [home and]] [[he went] inside]]
    [but [then [[something unexpected] happened]]]
    [but [[[then he] found] [[a big] box]]]
    [[i [am a]] [nice dog]]
    [[[it [wanted to]] be] [[friends with] bob]]
