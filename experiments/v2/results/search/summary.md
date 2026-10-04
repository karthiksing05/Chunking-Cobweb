Plain-PCFG code of the training sentences (bits) and the commission of the maximum-likelihood grammar read off the analyses.

| Condition | Gold trees | Greedy, final word classes | Greedy, best of 12 starts | Beam 4, best of 12 starts | Beam 16, best of 12 starts | Spearman(code, commission) |
|---|---|---|---|---|---|---|
| small | 3,189 | 3,189 (0.0%) | 3,189 (0.0%) | 3,189 (0.0%) | 3,189 (0.0%) | 0.68 |
| med | 6,651 | 6,727 (48.5%) | 6,585 (11.8%) | 6,539 (14.6%) | 6,512 (5.9%) | 0.90 |
| large | 7,771 | 8,589 (60.3%) | 8,168 (34.2%) | 7,800 (8.2%) | 7,740 (6.1%) | 0.98 |
| term_low | 5,629 | 6,513 (78.1%) | 6,304 (72.2%) | 5,884 (56.4%) | 5,660 (21.6%) | 0.94 |
| term_med | 7,770 | 8,644 (75.1%) | 8,206 (53.2%) | 8,033 (54.9%) | 7,979 (48.1%) | 0.92 |
| term_high | 10,511 | 11,294 (70.1%) | 11,019 (57.4%) | 10,589 (47.4%) | 10,833 (57.9%) | 0.72 |
