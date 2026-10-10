Profile `complex_deep_300_s13.prof`: 1471 s profiled

| Group (self time, library calls charged to the caller) | Seconds | Share |
|---|---|---|
| grammar read-out: cuts, code lengths (grammar.py, mdl.py) | 567.6 | 38.6% |
| Cobweb calls (cobweb_cu) | 391.6 | 26.6% |
| chart passes (chart.py) | 259.8 | 17.7% |
| structure search (mdl_search: moves, scoring, apply) | 136.1 | 9.3% |
| memory: instances, replay driver (memory.py) | 62.8 | 4.3% |
| rest | 53.0 | 3.6% |
| word classes (mdl_search.word_classes) | 0.1 | 0.0% |

| Function (self + library time) | Seconds | Share |
|---|---|---|
| `profile_night.py:66(_cobweb_ifit)` | 390.6 | 26.6% |
| `mdl.py:33(dm_code)` | 234.0 | 15.9% |
| `chart.py:268(fill)` | 124.0 | 8.4% |
| `chart.py:32(__init__)` | 120.4 | 8.2% |
| `grammar.py:587(cost)` | 69.3 | 4.7% |
| `grammar.py:557(_compact)` | 66.4 | 4.5% |
| `grammar.py:674(compile_grammar)` | 44.9 | 3.1% |
| `mdl_search.py:464(signature)` | 42.7 | 2.9% |
| `grammar.py:100(evidence_cut)` | 39.8 | 2.7% |
| `model.py:60(consolidate)` | 37.7 | 2.6% |
| `grammar.py:195(hill_climb)` | 30.8 | 2.1% |
| `memory.py:221(instance)` | 24.8 | 1.7% |
| `mdl_search.py:259(rename)` | 24.5 | 1.7% |
| `mdl.py:59(_before)` | 22.3 | 1.5% |
| `grammar.py:61(__init__)` | 17.2 | 1.2% |
| `mdl_search.py:316(_merged_rows)` | 12.9 | 0.9% |
| `memory.py:233(cat)` | 10.5 | 0.7% |
| `grammar.py:602(_factored_code)` | 9.9 | 0.7% |
| `memory.py:357(top_level_nats)` | 8.2 | 0.6% |
| `grammar.py:164(merge_symbols)` | 7.7 | 0.5% |
| `chart.py:247(viterbi_tree)` | 7.3 | 0.5% |
| `mdl_search.py:253(analyses)` | 7.2 | 0.5% |
| `mdl_search.py:287(_labels)` | 6.8 | 0.5% |
| `memory.py:300(surface)` | 5.6 | 0.4% |
| `model.py:162(fit)` | 5.6 | 0.4% |
| `mdl_search.py:268(_phis)` | 5.0 | 0.3% |
| `mdl_search.py:334(scored_moves)` | 4.8 | 0.3% |
| `grammar.py:528(_elements)` | 4.6 | 0.3% |
| `mdl.py:97(code)` | 4.6 | 0.3% |
| `chart.py:354(pick)` | 4.4 | 0.3% |
| `memory.py:384(build_hierarchy)` | 4.4 | 0.3% |
| `mdl_search.py:364(_merge_terms)` | 4.3 | 0.3% |
| `mdl_search.py:181(_phi)` | 4.3 | 0.3% |
| `chart.py:348(sample_tree)` | 3.7 | 0.2% |
| `mdl_search.py:291(pairs)` | 3.5 | 0.2% |
| `mdl_search.py:492(chunk_and_merge)` | 3.5 | 0.2% |
| `unsupervised.py:223(_consolidate)` | 3.2 | 0.2% |
| `mdl_search.py:278(_nats)` | 3.0 | 0.2% |
| `memory.py:149(add)` | 2.8 | 0.2% |
| `mdl_search.py:397(apply)` | 2.6 | 0.2% |

| Function (cumulative) | Seconds | Calls |
|---|---|---|
| `unsupervised.py:115(sleep)` | 1471.1 | 1 |
| `model.py:162(fit)` | 1084.3 | 53 |
| `model.py:60(consolidate)` | 1075.6 | 53 |
| `unsupervised.py:223(_consolidate)` | 956.9 | 9 |
| `unsupervised.py:257(_fit)` | 749.8 | 34 |
| `unsupervised.py:162(<genexpr>)` | 729.4 | 7 |
| `grammar.py:674(compile_grammar)` | 600.4 | 318 |
| `memory.py:384(build_hierarchy)` | 437.2 | 318 |
| `unsupervised.py:196(_sampled)` | 423.9 | 1 |
| `profile_night.py:59(ifit)` | 392.4 | 2751216 |
| `profile_night.py:66(_cobweb_ifit)` | 390.6 | 2751216 |
| `grammar.py:587(cost)` | 333.4 | 564689 |
| `grammar.py:153(search_cut)` | 254.2 | 870 |
| `grammar.py:195(hill_climb)` | 254.2 | 1740 |
| `mdl.py:33(dm_code)` | 216.3 | 2027802 |
| `grammar.py:164(merge_symbols)` | 168.1 | 318 |
| `unsupervised.py:240(_map)` | 135.5 | 1 |
| `unsupervised.py:288(_search)` | 135.5 | 12 |
| `mdl_search.py:492(chunk_and_merge)` | 135.4 | 12 |
| `chart.py:32(__init__)` | 120.6 | 10500 |
| `chart.py:247(viterbi_tree)` | 114.7 | 7800 |
| `chart.py:268(fill)` | 105.2 | 1232114 |
| `grammar.py:631(cost)` | 79.8 | 73474 |
| `mdl_search.py:517(<genexpr>)` | 74.7 | 6141 |
| `mdl_search.py:464(signature)` | 74.3 | 2988 |
| `grammar.py:557(_compact)` | 60.1 | 639753 |
| `mdl.py:59(_before)` | 55.8 | 79306 |
| `memory.py:357(top_level_nats)` | 51.2 | 565007 |
| `mdl.py:97(code)` | 50.0 | 32165 |
| `mdl_search.py:508(<genexpr>)` | 49.9 | 1163182 |
| `mdl_search.py:334(scored_moves)` | 49.4 | 1165624 |
| `grammar.py:602(_factored_code)` | 45.0 | 3996 |
| `memory.py:221(instance)` | 40.9 | 2721858 |
| `grammar.py:100(evidence_cut)` | 39.6 | 636 |
| `mdl_search.py:253(analyses)` | 31.6 | 2486 |
| `mdl_search.py:364(_merge_terms)` | 27.6 | 886592 |
| `mdl_search.py:259(rename)` | 24.4 | 18591231 |
| `grammar.py:61(__init__)` | 20.1 | 636 |
| `mdl_search.py:316(_merged_rows)` | 14.6 | 457419 |
| `memory.py:233(cat)` | 10.5 | 30331104 |
| `mdl_search.py:268(_phis)` | 8.5 | 1229030 |
| `chart.py:348(sample_tree)` | 8.1 | 2700 |
| `mdl_search.py:291(pairs)` | 7.8 | 3268 |
| `mdl_search.py:287(_labels)` | 6.8 | 1519200 |
| `mdl_search.py:397(apply)` | 6.3 | 4647 |
