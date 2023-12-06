# Jim <img src="https://user-images.githubusercontent.com/4642979/218163532-1c8a58e5-6f36-42de-96d3-f245eee93cf8.png" alt="jim" width="35"/> - A JAX-based gravitational-wave inference toolkit

Jim is a set of tools to solve a number of inference problems in the field of gravitational-wave, including single event parameter estimation and population analysis (coming soon!). Jim is written in python, with heavy use of the [JAX](https://github.com/google/jax) and uses [flowMC](https://github.com/kazewong/flowMC) as its sampler. 

!!! warning
    **Jim is still in development**: As we are refactoring and continuing the development of the code, the API is subject to change. If you have any questions, please feel free to open an issue.



## Design philosophy

1. Extensibility over "feature complete"
2. Performance is a feature, lacking performance is a bug
3. We do not do use-case optimization

#