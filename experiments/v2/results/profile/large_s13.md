Profile `large_s13.prof`: 593 s profiled, 453 s wall without the profiler

| Group (self time, library calls charged to the caller) | Seconds | Share |
|---|---|---|
| grammar read-out: cuts, code lengths (grammar.py, mdl.py) | 349.9 | 59.0% |
| Cobweb calls (cobweb_cu) | 84.5 | 14.2% |
| structure search (mdl_search: moves, scoring, apply) | 64.3 | 10.8% |
| chart passes (chart.py) | 48.0 | 8.1% |
| memory: instances, replay driver (memory.py) | 32.8 | 5.5% |
| rest | 13.7 | 2.3% |
| word classes (mdl_search.word_classes) | 0.0 | 0.0% |

| Function (self + library time) | Seconds | Share |
|---|---|---|
| `mdl.py:33(dm_code)` | 179.6 | 30.3% |
| `profile_night.py:66(_cobweb_ifit)` | 84.3 | 14.2% |
| `grammar.py:587(cost)` | 46.7 | 7.9% |
| `grammar.py:557(_compact)` | 41.7 | 7.0% |
| `chart.py:32(__init__)` | 32.6 | 5.5% |
| `mdl_search.py:464(signature)` | 20.8 | 3.5% |
| `grammar.py:674(compile_grammar)` | 17.0 | 2.9% |
| `grammar.py:195(hill_climb)` | 16.0 | 2.7% |
| `grammar.py:100(evidence_cut)` | 14.9 | 2.5% |
| `memory.py:221(instance)` | 11.4 | 1.9% |
| `mdl_search.py:259(rename)` | 11.3 | 1.9% |
| `chart.py:268(fill)` | 10.8 | 1.8% |
| `mdl.py:59(_before)` | 9.9 | 1.7% |
| `model.py:60(consolidate)` | 9.0 | 1.5% |
| `memory.py:357(top_level_nats)` | 6.7 | 1.1% |
| `grammar.py:164(merge_symbols)` | 5.9 | 1.0% |
| `memory.py:233(cat)` | 5.3 | 0.9% |
| `mdl_search.py:316(_merged_rows)` | 5.0 | 0.8% |
| `grammar.py:61(__init__)` | 4.4 | 0.7% |
| `grammar.py:602(_factored_code)` | 4.2 | 0.7% |
| `mdl_search.py:334(scored_moves)` | 3.7 | 0.6% |
| `mdl_search.py:364(_merge_terms)` | 3.1 | 0.5% |
| `memory.py:300(surface)` | 2.8 | 0.5% |
| `mdl_search.py:181(_phi)` | 2.8 | 0.5% |
| `mdl_search.py:268(_phis)` | 2.7 | 0.5% |
| `chart.py:348(sample_tree)` | 2.6 | 0.4% |
| `mdl.py:97(code)` | 2.3 | 0.4% |
| `mdl_search.py:278(_nats)` | 2.2 | 0.4% |
| `mdl_search.py:253(analyses)` | 2.2 | 0.4% |
| `memory.py:384(build_hierarchy)` | 2.1 | 0.4% |
| `chart.py:247(viterbi_tree)` | 1.9 | 0.3% |
| `model.py:162(fit)` | 1.7 | 0.3% |
| `mdl_search.py:287(_labels)` | 1.6 | 0.3% |
| `memory.py:329(_top_level)` | 1.6 | 0.3% |
| `mdl_search.py:492(chunk_and_merge)` | 1.5 | 0.3% |
| `grammar.py:528(_elements)` | 1.4 | 0.2% |
| `mdl_search.py:397(apply)` | 1.4 | 0.2% |
| `memory.py:318(describe)` | 1.1 | 0.2% |
| `mdl_search.py:291(pairs)` | 1.1 | 0.2% |
| `profile_night.py:59(ifit)` | 1.1 | 0.2% |

| Function (cumulative) | Seconds | Calls |
|---|---|---|
| `unsupervised.py:115(sleep)` | 593.3 | 1 |
| `model.py:162(fit)` | 479.7 | 41 |
| `model.py:60(consolidate)` | 476.9 | 41 |
| `grammar.py:674(compile_grammar)` | 360.9 | 246 |
| `unsupervised.py:223(_consolidate)` | 330.7 | 8 |
| `unsupervised.py:162(<genexpr>)` | 308.0 | 7 |
| `unsupervised.py:257(_fit)` | 296.3 | 22 |
| `grammar.py:587(cost)` | 253.8 | 546130 |
| `unsupervised.py:196(_sampled)` | 209.8 | 1 |
| `mdl.py:33(dm_code)` | 172.2 | 1843648 |
| `grammar.py:153(search_cut)` | 148.8 | 672 |
| `grammar.py:195(hill_climb)` | 148.8 | 1344 |
| `grammar.py:164(merge_symbols)` | 146.6 | 246 |
| `memory.py:384(build_hierarchy)` | 106.9 | 246 |
| `profile_night.py:59(ifit)` | 85.4 | 1058762 |
| `profile_night.py:66(_cobweb_ifit)` | 84.3 | 1058762 |
| `unsupervised.py:240(_map)` | 64.2 | 1 |
| `unsupervised.py:288(_search)` | 64.2 | 12 |
| `mdl_search.py:492(chunk_and_merge)` | 64.2 | 12 |
| `memory.py:357(top_level_nats)` | 55.0 | 546376 |
| `grammar.py:557(_compact)` | 40.6 | 592399 |
| `mdl_search.py:517(<genexpr>)` | 34.3 | 5387 |
| `mdl_search.py:464(signature)` | 34.2 | 2810 |
| `chart.py:32(__init__)` | 33.3 | 8640 |
| `grammar.py:631(cost)` | 33.3 | 45039 |
| `mdl_search.py:508(<genexpr>)` | 25.0 | 760898 |
| `mdl_search.py:334(scored_moves)` | 24.7 | 763175 |
| `memory.py:221(instance)` | 19.5 | 1047228 |
| `grammar.py:602(_factored_code)` | 17.9 | 3078 |
| `mdl.py:59(_before)` | 17.7 | 48792 |
| `mdl.py:97(code)` | 16.0 | 18809 |
| `grammar.py:100(evidence_cut)` | 14.8 | 492 |
| `mdl_search.py:253(analyses)` | 13.5 | 2205 |
| `mdl_search.py:364(_merge_terms)` | 13.0 | 691404 |
| `chart.py:247(viterbi_tree)` | 11.5 | 5760 |
| `mdl_search.py:259(rename)` | 11.3 | 8745037 |
| `chart.py:268(fill)` | 9.4 | 136746 |
| `mdl_search.py:316(_merged_rows)` | 5.5 | 313322 |
| `grammar.py:61(__init__)` | 5.4 | 492 |
| `memory.py:233(cat)` | 5.3 | 12248928 |
| `mdl_search.py:268(_phis)` | 4.1 | 697885 |
| `mdl_search.py:278(_nats)` | 3.3 | 764121 |
| `memory.py:300(surface)` | 2.8 | 1047228 |
| `mdl_search.py:181(_phi)` | 2.8 | 4027464 |
| `chart.py:348(sample_tree)` | 2.7 | 2880 |
