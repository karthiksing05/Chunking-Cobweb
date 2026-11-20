# Changes in revision 2 (all PENDING markers resolved)

- Generation re-judged with judge 2 (Qwen3-8B + phi-4 syntax/meaning passes): headline now judge 2; earlier Qwen2.5-3B results moved to the appendix (Table genold, agreement Table agree). The GPT win at 250/500 sentences does not survive judge 2; the text says so.
- Language compression (r225): code lengths vs BPE, n-grams, LSTM, GPT with chunks-off / concepts-off controls (Fig. bits, Tables ablation and bits). Concluding remarks and limitations rewritten around the chess-vs-language contrast.
- Regimes: new section "Abstraction Degrades Meaning Faster Than Grammar" with the law (Table law, Fig. law), judged showcase (Table showcase), oracle protocols A and B.
- r223 context length and abstraction: Table ctx (configuration C9).
- r221 chess parse-then-regenerate, where/what dial: Table chesslaw, Fig. boards (configuration C11).
- Configuration table extended to C8-C11; C1 and C6 rows updated.
- New figures: abstraction_law.png, chess_regime_boards.png, generation_fair_count_judge2.png, language_bits_curve.png (make_figures_rev2.py). Checked visually for legibility and overlaps.
- Prose pass: contrast and "we did not X" constructions rewritten as positive statements; "how many" removed.
- Build: tectonic, 42 pages, 0 undefined references or citations, 0 overfull boxes (instances figure narrowed to .88 text width; \raggedbottom added).
- Open: legality curve in chess panel is the absolute share (identity = 1.00); the file-symmetry hypothesis for the realism classifier is untested.
