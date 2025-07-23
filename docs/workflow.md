# Workflow standard and API levels

To target different needs, such as the trade-off between launching large scale campaigns versus having fine-grain control of `jim`, we present 3 levels of abstraction in `jim`.

The higher the level, the more abstractions are introduced hence the infrastructure tends to be less flexible but more scalable. The lower the level, you have the freedom but also responsibility to make sure the code is behaving in the way you want. 

# Level 2 - Pipelined jim

On this level, you don't handle plotting, you don't handle running jim. You push one button then you look at all the plots that are generated.

# Level 1 - Managed jim

On this level, you interact with `jim` mostly through a parameterized class called the `RunManager`, which manage a `Run` that is a templated `Run` that is commonly used.

# Level 0 - Core jim

This is the level where the developer should be very familar with the internal of `jim`.