from abc import ABC, abstractmethod
import jax.numpy as jnp
from jaxtyping import Float

class PopulationModelBase(ABC):
    @abstractmethod
    def __init__(self, *params):
        self.params = params  
        
    @abstractmethod
    def evaluate(self,data: dict, pop_params: dict) -> Float:
        """
        Evaluate the likelihood for a given set of parameters.
        """
        raise NotImplementedError

# class TruncatedPowerLawModel(PopulationModelBase):
#     def __init__(self, *params):
#         super().__init__(*params)  

#     def truncated_power_law(self, x, x_min, x_max, alpha):
#         valid_indices = (x >= x_min) & (x <= x_max)
#         C = (1 - alpha) / (x_max**(1 - alpha) - x_min**(1 - alpha))
        
#         # Ensure x is treated properly and avoid non-concrete indexing
#         pdf = jnp.zeros_like(x)  # Initialize pdf to the same shape as x
#         pdf = jnp.where(valid_indices, C / (x ** alpha), pdf)

#         return pdf

#     def evaluate(self,data: dict, pop_params: dict) -> Float:
#         """Evaluate the truncated power law model with dynamic parameters."""
#         x_min = pop_params[0]
#         x_max = pop_params[1]
#         alpha = pop_params[2]
        
#         return self.truncated_power_law(data, x_min, x_max, alpha)

class TruncatedPowerLawModel(PopulationModelBase):
    def __init__(self, *params):
        super().__init__(*params)  

    def truncated_power_law(self, x, x_min, x_max, alpha):
        valid_indices = (x >= x_min) & (x <= x_max)
        C = (1 - alpha) / (x_max**(1 - alpha) - x_min**(1 - alpha))
        
        # Ensure x is treated properly and avoid non-concrete indexing
        pdf = jnp.zeros_like(x)  # Initialize pdf to the same shape as x
        pdf = jnp.where(valid_indices, C / (x ** alpha), pdf)

        return pdf

    def evaluate(self, pop_params: dict[str, Float], data: dict) -> Float:
        """Evaluate the truncated power law model with dynamic parameters."""
        print("pop_parmas",pop_params)
        m_min = pop_params["m_min"] 
        m_max = pop_params["m_max"]
        alpha = pop_params["alpha"] 
        return self.truncated_power_law(data, m_min, m_max, alpha)
    
    
 

