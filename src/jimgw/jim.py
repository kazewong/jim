from abc import ABC, abstractmethod
from jaxtyping import Array
from jimgw.likelihood import LikelihoodBase
from flowMC.sampler.Sampler import Sampler
from flowMC.nfmodel.base import Distribution
from flowMC.utils.EvolutionaryOptimizer import EvolutionaryOptimizer
import jax

class Jim(object):
    """ Master class for interfacing with flowMC
    
    """

    def __init__(self, sampler: Sampler, likelihood: LikelihoodBase, prior: Distribution, **kwargs):
        self.Sampler = sampler
        self.Likelihood = likelihood
        self.Prior = prior

    def maximize_likleihood(self, bounds: tuple[float,float],set_nwalkers: int = 100, n_loops: int = 2000):
        set_nwalkers = set_nwalkers
        initial_guess = self.Prior.sample(set_nwalkers)

        y = lambda x: -self.Likelihood(x)
        y = jax.jit(jax.vmap(y))
        print("Compiling likelihood function")
        y(initial_guess)
        print("Done compiling")

        print("Starting the optimizer")
        optimizer = EvolutionaryOptimizer(self.Prior.n_dim, verbose = True)
        state = optimizer.optimize(y, bounds, n_loops=n_loops)
        best_fit = optimizer.get_result()[0]
        return best_fit


    def sample(self):
        pass

    def plot(self):
        pass