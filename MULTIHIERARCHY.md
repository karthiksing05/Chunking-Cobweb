# Multiple Hierarchies Theory!

Whereas before we hypothesized needing multiple hierarchies for multiple levels of granularity or to separate primitives and composites, we now rely on multiple hierarchies to properly separate *content* and *context*.

# Prior problems

*   From our work in the `primitives` branch, we saw that N was getting grouped with N + V and Det grouped with Det + N and Adj + N
    *   This is not what we're after! We want things to be grouped functionally, and it seems like only doing this based off context is unsuccessful!

# Current solution

*   

# Active todo list

*   Need to figure out a context-only weighting style that works (perhaps negative attributes, sorting only by context should yield some valuable results)
    *   Perhaps bag-of-words-style weighting resurfaces here! Honestly, probably not - adjective order ruins the placeholders here which sucks, and a sliding window feels more feasible especially in accordance with the other stuff