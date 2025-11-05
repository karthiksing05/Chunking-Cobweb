# Data Documentation

A helper file where I can document how each folder's LTM was manually trained.

*   `grammar2_fullparse/` - a manually trained LTM for grammar 1 with a human-intuitive parse logic. 
*   `grammar2_fullparse/` - a manually trained LTM with 10 parse trees of "NP V" and "NP V NP" sentences where noun phrases were grouped first and the verb was grouped with the left noun phrase.
*   `grammar2_fullparse_no_ms/` - a manually trained LTM with 10 parse trees of "NP V" and "NP V NP" sentences where noun phrases were grouped first and the verb was grouped with the left noun phrase - note that this LTM was also built without evaluating the MERGE and SPLIT actions for Cobweb (so only BEST and NEW) evaluated.
    *   We see some fragile behavior here!
*   `grammar2_partialparse/` - a manually trained LTM with initial parse trees only grouping the noun phrases and latter parse trees completing the whole parse tree in the same fashion as above.
