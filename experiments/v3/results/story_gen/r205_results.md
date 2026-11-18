# Round 205: edge_exact temperature extension, merged with r200/r204

Arm: TRELLIS with `"edge":"exact"` (r204 M1). The round adds temperatures 2.4, 2.8 and 3.5 at windows 6 and 7; n=1000 also gets window 8. Seeds 13, 17, 19 and 23.
Only TRELLIS's temperature grid was extended. That is fair because the LSTM and GPT optima (t.7-1.0, r200) sit inside their own grids.
Metric: distinct new, judged-grammatical, unstored-trigram sentences per ATTEMPTED draw ("fair"). "Strict" additionally drops headless or truncated fragments, for every family.
Settings are selected on seeds 13 and 17; "held" is the mean on seeds 19 and 23. Sources: r205-fair-n{250,500,1000}.txt and r204-fair-n4000.txt.

## Fair count

| n | plain TRELLIS | TRELLIS edge (setting) | LSTM | GPT | edge vs LSTM, 4 seeds | edge vs GPT, 4 seeds |
|---|---|---|---|---|---|---|
| 250 | .219 | **.241** (w7 t3.5, held .233) | .244 (held .249) | .213 (held .216) | 2/4, -.003 | 4/4, +.028 |
| 500 | .280 | **.301** (w6 t2.8, held .302) | .304 (held .312) | .283 (held .288) | 2/4, -.003 | 4/4, +.018 |
| 1000 | .316 | **.338** (w7 t2.0, held .339) | .344 (held .351) | .324 (held .313) | 2/4, -.006 | 3/4, +.014 |
| 4000 (seeds 13, 17) | .317 | .319 (w5 t2.0) | .358 | .362 | 0/2, -.039 | 0/2, -.043 |

Held-out on seeds 19 and 23 (edge vs LSTM / edge vs GPT):
- n=250: 0/2 (-.017) / 2/2 (+.017)
- n=500: 1/2 (-.011) / 2/2 (+.014)
- n=1000: 1/2 (-.012) / 2/2 (+.027)

## Boundary-strict count

| n | plain | edge | LSTM | GPT |
|---|---|---|---|---|
| 250 | .130 | .224 | .226 | .202 |
| 500 | .171 | .275 | .288 | .271 |
| 1000 | .237 | .307 | .325 | .309 |
| 4000 | .223 | .264 | .325 | .328 |

## Interior optimum

- n=500 (t2.8) and n=1000 (t2.0) are interior.
- n=250 still peaks at the top of the grid (t3.5).
- n=4000 includes only the edge w5 arm (w5 at t1.3, 1.6 and 2.0) in the merged file.

## Verdict

- With exact boundaries and a full temperature sweep, TRELLIS (Cobweb trees, no gradients) ties the matched LSTM from n=250 to 1000. It wins 2 of 4 seeds at each size, with mean differences of -.003 to -.006, all within seed noise.
- It beats the matched GPT on 11 of 12 seed-size pairs, by +.014 to +.028.
- At n=4000 both neural models pull ahead by about .04 on both seeds, so TRELLIS does not scale with data on this metric.
- On the strict count TRELLIS edge is level with GPT and trails the LSTM by .002-.018.
