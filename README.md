# Chunking Cobweb

Work on ISLE Internship to model attributes of the psychological principle of "Cognitive Chunking" through a Cobweb-based framework.

## Updates to Cobweb:

A couple modifications were made to Cobweb to support some of the edge-case behavior of the chunking principles:

*   Mode 4 - BEST + NEW
*   Made AV_Counts editable
*   Added new fringe split logic to preserve root's concept hash when root is added
*   Modify Cobweb such that it contains a DEBUG mode and logs all actions (by concept hash)

## Design Decisions:

One of the most important things to keep track of over the course of this project is the importance of labeling and defining design decisions. These decisions are influenced and ascertained by any choice / assumption made over the course of defining the framework. While current design decisions have been made to support ease of access regarding the programming of the framework, they'll definitely be revisited as we aggregate results and rationale from our tests.

*   The distinction between primitive and composite instances is a pretty unique design decision - additionally, the decision to include more than one concept element and the relations we've fixed (currently sequential but subject to change) can be 
*   How is the most optimal chunk candidate added?
    *   I think this is currently being done by categorical utility - the chunk candidate that best matches a concept within the long-term hierarchy is added to the tree
    *   Another option is to do it by log-probability 
*   How do we select the best chunk candidate label?
    *   Currently done by basic-level nodes!
*   How do we denote when parse tree construction terminates?
    *   Partial parse tree based on categorical utility (need to print categorical utility here to find appropriate thresholds)
    *   Full parse tree every time until 1 node reached

## Test Planning:

Below is a list of all tests confirming and acknowledging the use of the framework. Use ```pytest -s test.py to run!```

*   ```parse_tree_test.py``` - small test to confirm the correct implementation of parse trees and parse tree composition
*   ```gen_learn_test.py``` - another small test to also confirm the logic of parse tree addition and processing

## Long-term goals

After we effectively demonstrate that this system can cleanly parse language and related associations, we can implement this system in more puzzle frameworks. Word searches, Chess, and Gomoku (which TAIL already implements) are all valuable implementations.

Additionally (and this is more tentative) creating an embeddings prior that overlaps the tree structure can potentially result in more comprehensive embeddings.