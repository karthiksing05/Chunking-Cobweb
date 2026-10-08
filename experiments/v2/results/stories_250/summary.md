TinyStories, sentences of 3–8 words over the 250 most frequent words: 2500 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 40.8 |
| word 2-gram model | 25.8 |
| word 3-gram model | 29.0 |
| TRELLIS v2 | 24.3 |

Grammar: 25 categories, 63 chunk types; 100% of training sentences analysed as one tree (1.00 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 86.3% | 84.6% | 89.3% | 66.6% |
| real | 17.7% | 20.8% | 22.3% | 47.8% |
| new and real | 4.0% | 5.4% | 11.6% | 14.4% |
| 3–8 words long | 93.2% | 93.2% | 66.3% | 86.7% |
| real, among those of 3–8 words | 19.0% | 22.2% | 24.6% | 53.2% |
| new and real, among those of 3–8 words | 4.3% | 5.7% | 8.4% | 14.6% |
| word pairs in TinyStories | 84.2% | 85.2% | 100.0% | 100.0% |
| word triples in TinyStories | 57.4% | 57.8% | 78.5% | 100.0% |
| mean length | 5.5 | 5.4 | 5.4 | 5.3 |
| perceived with the analysis it was generated from | 94.9% | 93.7% | – | – |
| chunks inside sentences found in TinyStories | 58.6% | 58.8% | – | – |

Largest categories:

- S15 (4022): and, are, the, play, said, is, were, help
- S24 (3217): an idea, something unexpected happened, play together, the park, of fun, all day, every day, and had fun
- S21 (2500): but then … happened, then something unexpected happened, they had … fun, they played … day, tim was very happy, they are happy, you are … friend, he was very happy
- S1 (2477): the, a, he, tim, she, they, tom, something
- S20 (2149): had an idea, then something unexpected happened, are happy, so much fun, a lot of fun, have fun, you want … me, and excited
- S14 (2029): fun, friends, it, happy, together, day, too, friend
- S0 (2022): to, and, was, with, had, said, unexpected, big
- S13 (1270): they, i, but, he, you, it, she, we
- S23 (1232): play with, something unexpected, a big, did not, want to, wanted to, and had, thank you
- S19 (981): they had, the cat, it was, tim and, tom and, the bird, lily and, the dog
- S10 (434): happy, sad, surprised, excited, scared, pretty, kind, nice
- S7 (328): was, the, started, a

Most frequent chunks inside training analyses: *very happy* (116), *wanted to* (99), *play with* (79), *something unexpected* (74), *so happy* (73), *something unexpected happened* (73), *was very happy* (71), *the cat* (67), *an idea* (64), *thank you* (63), *had an idea* (61), *a big* (61), *was happy* (61), *it was* (58), *was so happy* (58), *did not* (55), *was sad* (53), *want to* (51), *the dog* (50), *the bird* (49)


The grammar's own sentences (analysis as generated):

    [[thanked [was [very happy]]] [and proud]]  (new)
    [[she says] [yes [let's [[play with] it]]]]  (new)
    [be [careful [ben man]]]  (new)
    [[thank you] [bird it]]  (new)
    [they [both [said lily]]]  (new)
    [could [was [very happy]]]  (new)
    [yes [we [played tim's]]]  (new)
    [[the bird] [room [play again]]]  (new)
    [[tim was] [[very fast] bird]]  (new)
    [[she asked] [smiled [the [his [friends to]]]]]  (new)
    [water [together [but [[and had] fun]]]]  (new)
    [they [were [good friends]]]
    [cake [[man you] [are [my [best friend]]]]]  (new)
    [[[they [were [very happy]]] to] [see [his max]]]  (new)
    [[[his [was [very scared]]] [wanted to]] [look you]]  (new)
    [he [felt [safe [and [said [yes tim]]]]]]  (new)
    [he [was [so happy]]]
    [but [they [inside [and [tom [[she run] car]]]]]]  (new)
    [but [then [[something unexpected] happened]]]
    [they [[did not] [are happy]]]  (new)
    [this [is [go away]]]  (new)
    [[she liked] [[[were [very happy]] with] [her friends]]]  (new)
    [it [will [be [[the best] [day sad]]]]]  (new)
    [let's [help [the cat]]]
    [[he just] [then [[played not] happened]]]  (new)

All samples, including partial analyses (pieces joined by ·):

    [i [will [be [opened play]]]]  (new)
    [tim [was [very surprised]]]
    [[the special] [we [will bird]]]  (new)
    [they [thought [it would]]]  (new)
    [look [[[little bird] were] [good [[tim asked] [for [[a big] tree]]]]]]  (new)
    [you [can careful]]  (new)
    [lily [let's play]]
    [you [are [my [you sue]]]]  (new)
    [[he opened] [[he had] [an idea]]]  (new)
    [[the dog] [[did not] [know fun]]]  (new)
    [[he said] [no [but day]]]  (new)
    [i [have fun]]  (new)
    [come [thanked them]]  (new)
    [one [day [[tom saw] [sam excited]]]]  (new)
    [let's [now mom]]  (new)

Held-out sentences (minimum-risk analysis):

    [[she was] [[playing with] him]]
    [but [then [[something unexpected] happened]]]
    [can [you [help [me [find it]]]]]
    [[[amy said] to] [tom [look [at [my toy]]]]]
    [[he [wanted to]] [know [[what was] [in it]]]]
    [they [played [together [all [day long]]]]]
    [she [thought [[it was] [too hard]]]]
    [[tom liked] [the idea]]
    [tim [was [so happy]]]
    [they [laughed [[and had] [lots [of fun]]]]]
    [soon [[he was] [home [and [he [went inside]]]]]]
    [but [then [[something unexpected] happened]]]
    [but [[[then he] found] [[a big] box]]]
    [i [am [[a nice] dog]]]
    [[it [wanted to]] [be [[friends with] bob]]]
