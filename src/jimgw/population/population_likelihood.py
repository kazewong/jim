import jax
import jax.numpy as jnp
from jaxtyping import Float

from jimgw.base import LikelihoodBase

class PopulationLikelihood(LikelihoodBase):
    def __init__(self, mass_array, model_class, pop_params):
        self.mass_array = mass_array
        self.population_model = model_class(*pop_params)

    def evaluate(self, posteriors: dict, pop_params: dict) -> Float:
        model_output = self.population_model.evaluate(posteriors, pop_params)
        log_likelihood = jnp.sum(jnp.log(jnp.mean(model_output, axis=1)))
        return log_likelihood
        
        
        
        

    