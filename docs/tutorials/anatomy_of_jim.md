# Anatomy of Jim

While the actual implementation of classes can be as involve as you like, the top level idea of Jim is rather simple.
We encourage all extension to `jim` follow this pattern, as it make sure your code can interface with the rest of `jim` without a problem.
This guide aims to give you a high level overview of what are the important components of Jim, and how they interact with each other.
## Likelihood

### Data

There should be two main ways to get your data into `jim`, either you fetch it from some public database, or generate synthetic data.

### Model

## Prior

## Sampler

The main workhorse under the hood is a machine learning-enhanced sampler named [flowMC](https://flowmc.readthedocs.io/en/main/).
It shares a similar interface
For a detail guide to what are all the knobs in `flowMC`, there is a tuning guide for flowMC [here](https://flowmc.readthedocs.io/en/main/configuration/).
At its core, `flowMC` is still a MCMC algorithm, so the hyperparameter tuning is similar to other popular MCMC samplers such as [emcee](https://emcee.readthedocs.io/en/latest/), namely:

1. If you can, use more chains, especially on a GPU. Bring the number of chains up until you start to get significant performance hit or run out of memory.
2. Run it longer, in particular the training phase. In fact, most of the computation cost goes into the training part, once you get a reasonably tuned normalizing flow model, the production phase is usually quite cheap. To be concrete, blow `n_loop_training` up until you cannot stand how slow it is.

## Analysis