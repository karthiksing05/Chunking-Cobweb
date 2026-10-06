Prompts: the first 1–3 words of held-out sentences (211 seen as the beginning of a training sentence, 22 unseen), each completed 5 times. Grammar: `experiments/v2/results/stories/grammar.pkl`.

| Completions | Prompts | Of the training length | Real | Every word triple attested | New | New and every triple attested |
|---|---|---|---|---|---|---|
| TRELLIS v2, temperature 1 | seen | 87% | 66.1% | 79.8% | 36.6% | 16.4% |
| TRELLIS v2, temperature 1 | unseen | 86% | 17.9% | 46.3% | 100.0% | 46.3% |
| TRELLIS v2, temperature 0.5 | seen | 87% | 71.7% | 86.7% | 30.4% | 17.1% |
| TRELLIS v2, temperature 0.5 | unseen | 84% | 16.3% | 46.7% | 100.0% | 46.7% |
| word bigram | seen | 71% | 52.9% | 79.5% | 49.9% | 29.5% |
| word bigram | unseen | 62% | 7.4% | 50.0% | 100.0% | 50.0% |
| word trigram | seen | 93% | 76.0% | 100.0% | 27.3% | 27.3% |
| word trigram | unseen | 75% | 34.9% | 85.5% | 100.0% | 85.5% |

Bits of the held-out sentence's actual continuation, given its prompt:

| Model | Seen prompts | Unseen prompts |
|---|---|---|
| TRELLIS v2 | 5.58 | 8.36 |
| word bigram | 6.96 | 8.16 |
| word trigram | 6.35 | 10.26 |

Completions (the prompt, then | ; the analysis as drawn, open chunks of the scaffold in ⟨ ⟩):

    seen    she                → ⟨⟨she⟩ | [did not]⟩ · [like it]
    seen    she                → ⟨⟨she⟩ | asked⟩ · [lily [did not]]
    seen    she                → ⟨⟨she⟩ | [was happy]⟩
    seen    she                → ⟨⟨she⟩ | [was sad]⟩
    seen    she                → ⟨⟨she⟩ | [was sad]⟩
    seen    they               → ⟨⟨they⟩ | [[liked [to play]] together]⟩
    seen    they               → ⟨⟨⟨they⟩ | were⟩ happy⟩
    seen    they               → ⟨⟨they⟩ | [had fun]⟩
    seen    they               → ⟨⟨⟨they⟩ | felt⟩ [[liked [to play]] together]⟩
    seen    they               → ⟨⟨they⟩ | played⟩ · [and [tim happy]]
    seen    i                  → ⟨⟨⟨i⟩ | found⟩ it⟩
    seen    i                  → ⟨⟨i⟩ | can⟩ · help
    seen    i                  → ⟨⟨⟨i⟩ | want⟩ to⟩ · [the dog] · [not fun]
    seen    i                  → ⟨⟨i⟩ | can⟩ · [it was] · [so sad]
    seen    i                  → ⟨⟨i⟩ | like⟩ · that
    seen    lily               → ⟨⟨lily⟩ | [was happy]⟩
    seen    lily               → ⟨⟨lily⟩ | [was [so happy]]⟩
    seen    lily               → ⟨⟨lily⟩ | [[[and tom] are] friends]⟩
    seen    lily               → ⟨⟨lily⟩ | [[[and tom] were] happy]⟩
    seen    lily               → ⟨⟨lily⟩ | [was [very happy]]⟩
    seen    and                → ⟨⟨and⟩ | [they did]⟩
    seen    and                → ⟨⟨⟨and⟩ | is⟩ happy⟩
    seen    and                → ⟨⟨and⟩ | [[they wanted] [to help]]⟩
    seen    and                → ⟨⟨and⟩ | [the park]⟩
    seen    and                → ⟨⟨⟨and⟩ | like⟩ said⟩
    seen    sue                → ⟨⟨sue⟩ | [was [so happy]]⟩
    seen    sue                → ⟨⟨sue⟩ | [was [very sad]]⟩
    seen    sue                → ⟨⟨sue⟩ | [was [very happy]]⟩
    seen    sue                → ⟨⟨sue⟩ | [was [so happy]]⟩
    seen    sue                → ⟨⟨sue⟩ | [was [so happy]]⟩
