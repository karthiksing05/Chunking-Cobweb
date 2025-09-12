# GUI Plan!

My goal is to build a GUI that facilitates a better analysis of our new framework - specifically, one that allows us to revise on some of the initial design iterations throughout our framework instantiation!
*   GUI For Parse Tree Generation:
    *   We should have a method that manually merges instances when we select two consecutive pairs and shows the score of that pair - so we can create optimal and suboptimal parse trees. (In addition to the score, print a TON of statistics about everything, including the best possible thing that matches it from the tree).
    *   Have a button for stopping the parse tree addition and then a confirmation box that adds the parse tree to the LTM.
    *   Have a button that opens up a separate window to inspect the LTM in general - want to find out if similar parts of speech are characterized (move around through 3d space of the picture)

*   GUI for LTM Inspection:
    *   Have links between nodes to other nodes in the parse tree so that we can identify what 
        *   Have a built-in-summary mode that flattens and condenses all extended cases if possible (CATCH RECURSION HERE PLEASE)
    *   Make it extremely floaty, make it able to iterate between basic-level nodes
        *   Can add traversal data but the general vibes are flying camera!
    *   The logic of this system should be similar to a canvas with a frame of reference, and different buttons should allow us to teleport to different places to see different things
