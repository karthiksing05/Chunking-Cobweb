# General Brainstorm for Experimental Features

This is just a general thread for new ideas I'm brainstorming to make the framework more robust (need hacks and gimmicks that achieve what we want with the framework)!! Confident that the methodology is strong, but we need to make sure that, for instance, a path representation actually yields a meaningful result!

---

## Chunk Context Ideas

COMPLETELY NEW UNDERSTANDING?!?!?!?! REDISTRIBUTION IN AN INCREMENTAL WAY IS QUITE OPTIMAL

What if we maintained the basic level cut as we say, and each path has exactly one basic level, and we just represented nodes by their basic level - furthermore, each level of the parse tree would have its own hierarchy!!!

Two main things to test: 
1.  Cobweb as a diffusion model is extremely strong - can we surround chunks with chunk context and show convergence to some pure Gaussian noise
    *   Just 
2.  Emulation of the initial "Cobweb-Forest" idea that I had! Have a separate hierarchy for each layer of complexity in the parse tree to promote better separation
    *   Need to figure out if this is a thing we need to adjust in the context hierarchy, content hierarchy, or both
