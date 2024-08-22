# Jim's parameter transform system

!!! warning
    **Heavy development**: This is a work in progress and is subject to change. If you have any questions, please feel free to open an issue.



A sketch of the transform system is shown below:
![A sketch of the transform system](prior_system_diagram.jpg)


## Setting up Priors
Prior is the prior knowledge we have on the probability density distribution of the event parameters. In Jim, we could set up prior based a set of event parameters $\theta_{prior}$. To define the prior, we need to call the prior class built in Jim. Suppose we want to define the prior of the parameter x to be uniform, we could call the UniformPrior class:

```
from jimgw.prior import UniformPrior
prior = UniformPrior(0.0, 1.0, parameter_names=["x"])
```

There are a few prior classes available in Jim: ...

## Setting up Sample Transform


## Setting up Likelihood Transform

