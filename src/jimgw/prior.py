from flowMC.nfmodel.base import Distribution
import jax
from jaxtyping import Array, Float

class Uniform(Distribution):

    xmin: Array
    xmax: Array

    def __init__(self, xmin: float, xmax: float):
        self.xmax = xmax
        self.xmin = xmin
    
    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
        return super().sample(rng_key, n_samples)
    
    def log_prob(self, x: Array) -> Float:
        return super().log_prob(x)
    
