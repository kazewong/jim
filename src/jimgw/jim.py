from jimgw.likelihood import LikelihoodBase
from flowMC.sampler.Sampler import Sampler
from flowMC.nfmodel.base import Distribution
from flowMC.utils.EvolutionaryOptimizer import EvolutionaryOptimizer
import jax

class Jim(object):
    """ Master class for interfacing with flowMC
    
    """

    def __init__(self, likelihood: LikelihoodBase, prior: Distribution, sampler_kwargs, **kwargs):
        self.Likelihood = likelihood
        self.Prior = prior

        nf_sampler = Sampler(
            self.Prior.n_dim,
            rng_key_set,
            None,
            local_sampler,
            model,
            n_loop_training=n_loop_training,
            n_loop_production = n_loop_production,
            n_local_steps=n_local_steps,
            n_global_steps=n_global_steps,
            n_chains=n_chains,
            n_epochs=num_epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            batch_size=batch_size,
            use_global=True,
            keep_quantile=0.,
            train_thinning = 40,
        )

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