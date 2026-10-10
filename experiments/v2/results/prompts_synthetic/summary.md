Prompts: the first 1–4 tokens of the held-out sentences (v1 splits, seeds 13,17,7,42,100), each completed 5 times; completions judged by the target grammar. Rates summed over seeds.

**Prompts seen as the beginning of a training sentence**

| Condition | Prompts | TRELLIS v2, gold trees: grammatical / novel and grammatical | TRELLIS v2, sentences alone: grammatical / novel and grammatical | word bigram: grammatical / novel and grammatical | word trigram: grammatical / novel and grammatical |
|---|---|---|---|---|---|
| small | 396 | 99.9% / 58.1% | 100.0% / 58.2% | 45.7% / 27.2% | 45.2% / 22.4% |
| med | 370 | 99.3% / 96.6% | 99.7% / 95.9% | 37.9% / 35.8% | 36.8% / 34.1% |
| large | 299 | 88.6% / 87.7% | 92.8% / 91.7% | 30.7% / 30.2% | 34.0% / 32.3% |
| term_low | 293 | 99.9% / 90.2% | 99.8% / 89.5% | 33.3% / 26.6% | 31.4% / 23.8% |
| term_med | 364 | 98.4% / 95.9% | 99.8% / 97.5% | 32.3% / 31.1% | 34.5% / 32.5% |
| term_high | 281 | 97.4% / 97.2% | 99.2% / 98.9% | 26.3% / 26.0% | 26.6% / 26.3% |

**Prompts unseen as the beginning of a training sentence**

| Condition | Prompts | TRELLIS v2, gold trees: grammatical / novel and grammatical | TRELLIS v2, sentences alone: grammatical / novel and grammatical | word bigram: grammatical / novel and grammatical | word trigram: grammatical / novel and grammatical |
|---|---|---|---|---|---|
| small | 8 | 100.0% / 100.0% | 100.0% / 100.0% | 42.5% / 42.5% | 52.5% / 52.5% |
| med | 92 | 99.6% / 99.6% | 100.0% / 100.0% | 26.7% / 26.7% | 28.0% / 28.0% |
| large | 255 | 86.8% / 86.8% | 90.6% / 90.6% | 38.6% / 38.6% | 40.2% / 40.2% |
| term_low | 1 | 100.0% / 100.0% | 100.0% / 100.0% | 40.0% / 40.0% | 0.0% / 0.0% |
| term_med | 92 | 98.7% / 98.7% | 99.8% / 99.8% | 28.5% / 28.5% | 26.1% / 26.1% |
| term_high | 285 | 97.5% / 97.5% | 99.0% / 99.0% | 36.9% / 36.9% | 38.2% / 38.2% |

Bits of the held-out sentences' actual continuations given their prompts (mean over prompts and seeds):

| Condition | TRELLIS v2, gold trees (seen / unseen) | TRELLIS v2, sentences alone (seen / unseen) | word bigram (seen / unseen) | word trigram (seen / unseen) |
|---|---|---|---|---|
| small | 3.6 / 2.6 | 3.6 / 2.6 | 4.9 / 3.6 | 5.3 / 4.0 |
| med | 12.2 / 11.8 | 12.1 / 11.8 | 15.0 / 14.5 | 16.8 / 17.1 |
| large | 14.9 / 13.8 | 14.6 / 13.5 | 17.5 / 16.2 | 22.0 / 20.9 |
| term_low | 10.6 / 7.2 | 10.6 / 7.2 | 14.1 / 9.9 | 13.2 / 9.3 |
| term_med | 15.8 / 15.4 | 15.8 / 15.4 | 19.5 / 19.2 | 21.7 / 21.7 |
| term_high | 22.8 / 18.9 | 22.7 / 18.8 | 27.7 / 23.0 | 35.5 / 29.6 |

Completions of unseen prompts by TRELLIS v2 from sentences alone (✓ grammatical):

    small     the man chased the       → the man chased the woman ✓
    small     the man chased the       → the man chased the woman ✓
    small     the man chased the       → the man chased the cat ✓
    small     the telescope admired the → the telescope admired the telescope ✓
    small     the telescope admired the → the telescope admired the woman ✓
    small     the telescope admired the → the telescope admired the cat ✓
    small     the cat admired the      → the cat admired the telescope ✓
    small     the cat admired the      → the cat admired the park ✓
    small     the cat admired the      → the cat admired the cat ✓
    small     the telescope admired the → the telescope admired the telescope ✓
    small     the telescope admired the → the telescope admired the park ✓
    small     the telescope admired the → the telescope admired the woman ✓
    small     the dog liked the        → the dog liked the telescope ✓
    small     the dog liked the        → the dog liked the woman ✓
    small     the dog liked the        → the dog liked the telescope ✓
    med       a lazy red dog           → a lazy red dog found the man ✓
    med       a lazy red dog           → a lazy red dog chased a man ✓
    med       a lazy red dog           → a lazy red dog liked a woman ✓
