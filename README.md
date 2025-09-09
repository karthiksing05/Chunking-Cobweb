# Chunking Cobweb

Work with Dr. Pat Langley and Dr. Chris Maclellan on ISLE Internship to model attributes of the psychological principle of cognitive Chunking through a Cobweb-backed framework.

## Updates to Cobweb:

A couple modifications were made to Cobweb to support some of the edge-case behavior of the chunking framework:

*   Mode 4 - BEST + NEW
*   Made AV_Counts editable
*   Added new fringe split logic to preserve root's concept hash when root is added
*   Modify Cobweb such that it contains a DEBUG mode and logs all create/delete actions (by concept hash, new and old)

## Design Decisions:

One of the most important things to keep track of over the course of this project is the importance of labeling and defining design decisions. These decisions are influenced and ascertained by any choice / assumption made over the course of defining the framework. While current design decisions have been made to support ease of access regarding the programming of the framework, they'll definitely be revisited as we aggregate results and rationale from our tests.

*   The distinction between primitive and composite instances is a pretty unique design decision - additionally, the decision to include more than one concept element and the relations we've fixed (currently sequential, left-right) can be changed as well
*   How is the most optimal chunk candidate added?
    *   I think this is currently being done by log-probability - the chunk candidate that best matches a concept within the long-term hierarchy is added to the tree (each candidate is categorized to identify the best possible match)
    *   In future instances, we could do this by some kind of collocation score?
*   How do we select the best chunk candidate label?
    *   Currently done by basic-level nodes! There are talks of a Pointwise Mutual Information metric coming into play but subject to change.
*   How do we denote when parse tree construction terminates?
    *   Partial parse tree based on the metric by which we add nodes to the parse tree
    *   Full parse tree every time until one node reached
*   The decision to include the SPLIT / MERGE operations is incredibly interesting
    *   Prior studies done with Cobweb/4L indicate that SPLIT and MERGE were barely used but that they definitely could be in a longer term
    *   SPLIT and MERGE contribute to long-term inference efficiency (removing of redundant concepts) which is super relevant for beginning and end behavior
*   The data we choose to read in is subject to change - currently, we're reading language by sentences, but future iterations can do larger-scale windows and slide the window / create parse trees flexibly over time.
    *   We read in the first n primitives and parse them, then we read the next n primitives (keeping the root node of the primitives)
    *   [COOL STUFF] If we construct partial parse trees, we can continue iteratively adding primitives to our "active working memory" until the threshold for partial parse trees have been met, and then we can dump everything into 

## Test Planning:

Below is a list of all tests confirming and acknowledging the use of the framework. Use ```pytest -s test.py``` to run!

*   ```parse_tree_test.py``` - small test to confirm the correct implementation of parse trees and parse tree composition
*   ```gen_learn_test.py``` - another small test to also confirm the logic of parse tree addition and processing

## Long-term goals

After we effectively demonstrate that this system can cleanly parse language and related associations, we can implement this system in more puzzle frameworks. Word searches, Chess, and Gomoku (which TAIL already implements) are all valuable implementations.

Additionally (and this is more tentative) creating an embeddings prior that overlaps the tree structure can potentially result in more comprehensive embeddings.