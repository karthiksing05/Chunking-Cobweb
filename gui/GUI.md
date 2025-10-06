# GUI Plan!

My goal is to build a GUI that facilitates a better analysis of our new framework - specifically, one that allows us to revise on some of the initial design iterations throughout our framework instantiation!

Rough Plan
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

## Tech Stack + Code Plan

*   Probably best to do this in HTML + CSS + Javascript and make it a bootable webpage; GUI outline is pretty fleshed out with the visualizations of the parse trees and LTMs, just need to bring it to a dynamic setting.

*   Code plan for Parse Tree Editor
    *   The screen should have a primary window that shows the tree and then a sidebar that shows a log of all actions made
        *   Perhaps also a title that says "Parse Tree Editor" and a subtitle that says "Current sentence: ..." with the dots being filled in by a random sentence
    *   There's a button for every two consecutive parentless instances: upon selecting a button that marks two consecutive instances, we see, in a popup window:
        *   The best prospective chunk candidate from the long-term memory
        *   The score for this best candidate (as marked by our scoring function)
            *   We should also try to add the different scoring statistics present in the scoring function's debug print as a table!
        *   Also in the popup window, there should be a button to confirm the selection of adding a new chunk candidate (with a popup dialog box for confirmation)
            *   Upon confirmation, it should populate this in the log as well.
    *   An undo button that removes the last added chunk candidate and reverts the tree back to the prior state
        *   Should delete actions from the log as this happens!
    *   A submit button that exports the parse tree as a JSON (with a confirmation box) and then feedback that says the parse tree was exported in the log.

*   Code plan for LTM Inspector (honestly, I think it's probably optimal to make all of these changes to HTMLCobwebDrawer, or even create a new active HTML Cobweb Drawer)
    *   The screen should have a primary window like a "stage" that is similar to a scrollable and zoomable canvas of the long-term memory, with the following features:
        *   The long-term memory, which is a CobwebTree, should be displayed stylistically similarly to the way that HTMLCobwebDrawer shows the CobwebTree with a couple changes:
            *   All basic-level nodes should show in a different highlighted outline / shadow to identify that they are basic-level (perhaps purple!)
            *   Each Concept-HASH ID present in each node table should be able to hyperlink to open up a floating, movable, closeable popup box with that node table present in it
                *   Each of these popup boxes with node tables should also contain hyperlinks for any nodes that represent Concept-HASH IDs, so you should be able to recursively open and close pop-up boxes
    *   There should also be a search bar where the concept hashes slowly fill in as you continue typing in an ID. Upon pressing a search button, the stage should automatically teleport to focus on the concept hash you searched.
    *   A button for loading parse trees that, upon clicking, opens a file selection dialog box to pick a .json file for a parse tree (with necessary error correction to see if it's a parse tree json)
        *   A visualized process for addition to the Cobweb tree that prints all log actions and conveys all actions that are about to be made in a flipbook style (this is optional and potentially will be revised, as it seems like a better change to make to HTMLCobwebDrawer than the ltm inspection code)

*   Combining both ideas:
    *   Currently we just have a button to export the LTM when we need it - proving highly successful!
    *   Going to table the 