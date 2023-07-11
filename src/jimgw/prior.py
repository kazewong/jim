import jax
import jax.numpy as jnp
from flowMC.nfmodel.base import Distribution
from jaxtyping import Array, Float
from typing import Callable

class Prior(Distribution):
    r"""A thin wrapper build on top of flowMC distributions to do book keeping.

    Should not be used directly since it does not implement any of the real method.
    """

    naming: list[str]
    transforms: list[Callable] = []

    def __init__(self, naming: list[str], transforms: list[Callable] = []):
        pass

    def transform(self, x: Array):
        pass


class Uniform(Prior):

    xmin: Array
    xmax: Array

    def __init__(self, xmin: float, xmax: float, naming: list[str]):
        self.xmax = xmax
        self.xmin = xmin
        self.naming = naming
    
    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
        samples = jax.random.uniform(rng_key, n_samples, minval=self.xmin, maxval=self.xmax)
        return samples # TODO: remember to cast this to a named array

    def log_prob(self, x: Array) -> Float:
        return jnp.log(1./(self.xmax-self.xmin)) 
