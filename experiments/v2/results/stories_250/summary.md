TinyStories, sentences of 3–8 words over the 250 most frequent words: 2500 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 40.8 |
| word 2-gram model | 25.8 |
| word 3-gram model | 29.0 |
| TRELLIS v2 | 24.7 |

Grammar: 29 categories, 59 chunk types; 100% of training sentences analysed as one tree (1.00 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 84.5% | 85.4% | 89.7% | 67.0% |
| real | 20.8% | 20.4% | 21.6% | 46.7% |
| new and real | 5.3% | 5.8% | 11.3% | 13.7% |
| 3–8 words long | 95.8% | 96.8% | 67.0% | 86.9% |
| real, among those of 3–8 words | 21.6% | 21.1% | 23.9% | 51.9% |
| new and real, among those of 3–8 words | 5.4% | 6.0% | 8.5% | 13.9% |
| word pairs in TinyStories | 83.3% | 83.0% | 100.0% | 100.0% |
| word triples in TinyStories | 57.4% | 55.4% | 77.7% | 100.0% |
| mean length | 5.4 | 5.4 | 5.5 | 5.3 |
| perceived with the analysis it was generated from | 96.5% | 96.7% | – | – |
| chunks inside sentences found in TinyStories | 58.0% | 58.1% | – | – |

Largest categories:

- S15 (3237): and, the, play, help, not, it, very, said
- S7 (2500): but then … happened, then something unexpected happened, they had … fun, they played … day, tim was very happy, they are happy, you are … friend, he was very happy
- S11 (2269): happy, fun, friends, it, sad, happened, idea, together
- S10 (1592): something unexpected happened, are friends, a lot of fun, together every day, help you, play together, so much fun, and had fun
- S8 (1490): of fun, an idea, the park, every day, all day, had fun, in the park, very happy
- S0 (1119): the cat, it was, tim and, tom and, the bird, lily and, the dog, he said
- S2 (1107): play with, something unexpected, a big, want to, thank you, his mom, a lot, her mom
- S28 (1098): a, the, play, something, his, want, her, thank
- S24 (1073): to, with, unexpected, big, you, mom, were, was
- S21 (1058): he, tim, she, tom, lily, it, sue, max
- S4 (1052): an idea, and excited, play together, play with it, was sad, was very happy, not know … do, play together every day
- S6 (785): then something unexpected happened, something unexpected happened, you want … me, will help you, are a good friend, are very kind, did not know, played with … day

Most frequent chunks inside training analyses: *very happy* (112), *wanted to* (99), *play with* (79), *something unexpected* (74), *so happy* (73), *something unexpected happened* (73), *was very happy* (71), *the cat* (67), *an idea* (64), *a big* (61), *was happy* (59), *was so happy* (58), *was sad* (52), *want to* (51), *the dog* (50), *the bird* (49), *it was* (42), *thank you* (42), *very sad* (40), *loved to* (38)


The grammar's own sentences (analysis as generated):

    [about [was [very rock]]]  (new)
    [[she had] [[a lot] [of fun]]]  (new)
    [they [are [long [like ball]]]]  (new)
    [let's [go [play happened]]]  (new)
    [many [found [it [everyone [the [was [[a big] tree]]]]]]]  (new)
    [[max said] [sue [sky happy]]]  (new)
    [[tim said] [[she toys] scared]]  (new)
    [[he did] [not [know happened]]]  (new)
    [[max and] [[sue were] [very sad]]]  (new)
    [[[max and] did] [not know]]  (new)
    [she [[his friend] are]]  (new)
    [[he [bear to]] [[play with] it]]  (new)
    [i'm [sorry [my [best mom]]]]  (new)
    [[he saw] [[his friend] [happy [the dog]]]]  (new)
    [i [[a little] better]]  (new)
    [i [don't [[his mom] [i [come spot]]]]]  (new)
    [[he had] [an idea]]
    [[it was] [[a toy] [[what park] [was there]]]]  (new)
    [[lily and] [[max saw] [[a girl] [named lily]]]]  (new)
    [[he did] [not [tom [had [inside [i tree]]]]]]  (new)
    [mia [was sad]]
    [they [were [very happy]]]
    [not [thought [[cake saw] [[they to] do]]]]  (new)
    [i [can [do [it happy]]]]  (new)
    [[they [liked to]] [play together]]

All samples, including partial analyses (pieces joined by ·):

    [[he [was happy]] [to [play together]]]  (new)
    [now [was sad]]  (new)
    [spot [day [[the dog] [had sam]]]]  (new)
    [they [are [not [good bird]]]]  (new)
    [[she says] [sorry [and happened]]]  (new)
    [i [like [but dog]]]  (new)
    [but [be [nice [and lucy]]]]  (new)
    [[door make] [very surprised]]  (new)
    [[she [was happy]] [to [help [her mom]]]]  (new)
    [[sam and] [lily [felt sorry]]]  (new)
    [they [played together]]  (new)
    [they [were [all excited]]]  (new)
    [[she liked] [his [[a nice] toys]]]  (new)
    [and [you [you [and excited]]]]  (new)
    [they [are [friends [and laughed]]]]  (new)

Held-out sentences (minimum-risk analysis):

    [[she was] [[playing with] him]]
    [but [then [[something unexpected] happened]]]
    [can [you [help [me [find it]]]]]
    [[amy said] [to [tom [look [at [my toy]]]]]]
    [[he [wanted to]] [know [[what was] [in it]]]]
    [they [played [together [all [day long]]]]]
    [she [thought [it [was [too hard]]]]]
    [[tom liked] [the idea]]
    [tim [was [so happy]]]
    [they [laughed [and [had [lots [of fun]]]]]]
    [soon [[he was] [home [and [[he went] inside]]]]]
    [but [then [[something unexpected] happened]]]
    [but [then [[he found] [[a big] box]]]]
    [i [am [[a nice] dog]]]
    [[it [wanted to]] [be [[friends with] bob]]]
