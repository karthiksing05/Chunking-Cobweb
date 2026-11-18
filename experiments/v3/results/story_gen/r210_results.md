# Round 210: M3 repeat tracking (edge + same) at n=1000, w7

Arm: `"edge":"exact"`, `"repeat":"same"`, cfg `same_k`, window 7, temps 1.3 / 1.6 / 2.0, n=1000, seeds 13/17/19/23 (jobs 4788544-47, judge 4788548, fair 4788549, compare 4788600).
Edge baseline = r205 (same w7, same temps, same seeds). Sources: r210-fair-n1000.txt / .perseed.tsv, r210-compare-n1000.txt (script experiments/v3/diag/r210_compare.py).
Repeated-word rate = share of distinct new sentences with a word occurring twice anywhere in the sentence.

## Edge -> edge+same, mean over 4 seeds (wins = seeds where same > edge)

| temp | fair | strict | dist/draw | tri\|d | gram\|tri | repeated-word rate (all / among ungrammatical) |
|---|---|---|---|---|---|---|
| 1.3 | .313 -> .314 (+.001, 3/4) | .290 -> .288 (-.002, 1/4) | .740 -> .748 | .756 -> .774 (+.018, 4/4) | .559 -> .542 (-.017, 2/4) | .089 -> .047 / .171 -> .065 |
| 1.6 | .329 -> .331 (+.002, 3/4) | .299 -> .304 (+.005, 3/4) | .779 -> .788 | .791 -> .807 (+.016, 3/4) | .534 -> .521 (-.013, 1/4) | .090 -> .050 / .160 -> .069 |
| 2.0 | .338 -> .332 (-.006, 1/4) | .307 -> .296 (-.010, 1/4) | .814 -> .816 | .836 -> .834 | .497 -> .487 (-.010, 1/4) | .089 -> .047 / .153 -> .057 |

Selection (13/17) / held-out (19/23) of the fair diff: t1.3 +.009 / -.007; t1.6 +.009 / -.005; t2.0 -.008 / -.005.
gram|tri diff held-out: -.037 / -.035 / -.025.

## Best setting chosen on seeds 13/17: edge+same w7 t1.6

sel .331, held .331 (per seed 13 .331, 17 .332, 19 .345, 23 .318). Edge alone at t1.6 held .336; best edge alone (t2.0) held .339.
All four seeds at the chosen setting: .331 / .332 / .345 / .318 (edge t2.0 r205: .337 / .337 / .353 / .326).

## Paired wins against neural baselines (fair, LSTM and GPT at t0.8)

- Chosen setting (t1.6), held-out 19/23: vs LSTM 1 of 2, mean -.020 (r205 edge: -.012); vs GPT 1 of 2, +.019 (r205 edge: +.027).
- Best over all seeds (edge+same t2.0, .332): vs LSTM 2 of 4, -.012; vs GPT 2 of 4, +.007. (r205 edge t2.0: LSTM 2/4 -.006, GPT 3/4 +.014.)

## Verdict

- The mechanism works as designed on its target: the repeated-word rate roughly halves (.09 -> .05 of distinct sentences; among ungrammatical ones .16 -> .06), and tri|d rises by about .017 at t1.3 / t1.6.
- It does not raise gram|tri. Same lowers it at every temperature, -.010 to -.017 on 4 seeds and -.025 to -.037 on held-out seeds, even though tri is higher, not matched. Removing repeats does not fix the remaining ungrammaticality, and the extra novel trigrams are no more grammatical.
- It does not raise the fair count: +.001, +.002, -.006 (all within seed noise; held-out -.007 / -.005 / -.005). It does not beat the r205 edge baseline (.338), and it loses ground to LSTM (-.012 vs -.006 at best setting) and GPT (+.007 vs +.014).
- Negative result: M3 (store_same + repeat_read) is not a win at n=1000. The M3 pre-test finding (98% of ungrammatical repeat sentences repeat within the window) was a correct count, but repeats were only about 16% of ungrammatical sentences, so suppressing them leaves most failures.
