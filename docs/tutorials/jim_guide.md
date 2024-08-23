# Jim's parameter transform system

!!! warning
    **Heavy development**: This is a work in progress and is subject to change. If you have any questions, please feel free to open an issue.



A sketch of the transform system is shown below:
![A sketch of the transform system](prior_system_diagram.png)


## Setting up Priors
Prior represents the prior knowledge on the probability density distribution of the event parameters $\theta$. In Jim, we could set up priors using the prior class `jimgw.prior`. Suppose we want to define a uniform prior distribution for the parameter $x$, we could call the `UniformPrior` class:

```
from jimgw.prior import UniformPrior
prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
```

Jim provides a number of built-in prior classes, including:
- `UniformPrior`
- `SinePrior`
- `CosinePrior`
- `PowerLawPrior`

A complete list of prior classes available can be found in the Jim documentation (not available yet).

## Multi-dimensional Priors
When working with multi-dimensional parameter space, we would usually want to define priors for each individual parameter first:

```
from jimgw.prior import UniformPrior
prior_x = UniformPrior(0.0, 1.0, parameter_names=["x"])
prior_y = UniformPrior(0.0, 2.0, parameter_names=["y"])
```

Once we have the individual priors defined, we can call the class `CombinePrior` to combine them into one single prior object:

```
from jimgw.prior import CombinePrior
prior = CombinePrior(
    [
        prior_x,
        prior_y,
    ]
)
```

The `CombinePrior` object can later be used as input to the Jim sampler.

## Setting up Sample Transform


## Setting up Likelihood Transform

