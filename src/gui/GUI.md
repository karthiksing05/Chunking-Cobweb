# GUI Plan!

My goal is to build a GUI that facilitates a better analysis of our new framework - specifically, one that allows us to revise on some of the initial design iterations throughout our framework instantiation!
*   GUI For Parse Tree Generation:
    *   Have a selector for Finite + Continuous Parse Trees (currently, only Finite should be supported, just throw an error for )
    *   We should have a method that manually merges instances when we select two consecutive pairs and shows the score of that pair - so we can create optimal and suboptimal parse trees (in addition to the score, print the best possible match from the long-term memory).
        *   When we select two instances, it should show statistics for those instances (score-wise)
        *   There should also be a button that auto-selects the two best instances to merge
    *   Have a button for stopping the parse tree addition and then a confirmation pop-up box that adds the parse tree to the LTM.
    *   Have a button that opens up a separate window to inspect the LTM in general - want to find out if similar parts of speech are characterized (move around through 3d space of the picture)

*   GUI for LTM Inspection:
    *   Have links between nodes to other nodes in the parse tree so that we can identify what 
        *   Can have a hyperlinking situation where each concept ID links to a pop-up display of the node that it's a part of, and those nodes should also have hyperlinks to other nodes
    *   The general vibes are flying camera - should be able to pan around the tree structure but then upon clicking concept IDs within the tables for content-left, content-right, context-before, context-after, it should show a pop-up of that concept ID's table

## Tech Stack

*   Probably best to do this in HTML + CSS + Javascript; GUI outline is pretty fleshed out with the visualizations of the parse trees and LTMs, just need to bring it to a dynamic setting