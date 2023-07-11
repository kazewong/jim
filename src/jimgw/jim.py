from jimgw.likelihood import LikelihoodBase
from flowMC.sampler.Sampler import Sampler
from flowMC.sampler.MALA import MALA
from flowMC.nfmodel.base import Distribution
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.utils.EvolutionaryOptimizer import EvolutionaryOptimizer
import jax

class Jim(object):
    """ Master class for interfacing with flowMC
    
    """

    def __init__(self, likelihood: LikelihoodBase, prior: Distribution, sampler_kwargs: dict, **kwargs):
        self.Likelihood = likelihood
        self.Prior = prior
        seed = sampler_kwargs.get("seed", 0)

        rng_key_set = initialize_rng_keys(seed)
        num_layers = sampler_kwargs.get("num_layers", 10)
        hidden_size = sampler_kwargs.get("hidden_size", [128,128])
        num_bins = sampler_kwargs.get("hidden_size", 8)

        local_sampler = MALA(self.Likelihood.evaluate, True, 1e-2) # Remember to add routine to find automated mass matrix

        model = MaskedCouplingRQSpline(self.Prior.n_dim, num_layers, hidden_size, num_bins)
        self.Sampler = Sampler(
            self.Prior.n_dim,
            rng_key_set,
            None,
            local_sampler,
            model,
            **sampler_kwargs)
        
        #     n_loop_training=n_loop_training,
        #     n_loop_production = n_loop_production,
        #     n_local_steps=n_local_steps,
        #     n_global_steps=n_global_steps,
        #     n_chains=n_chains,
        #     n_epochs=num_epochs,
        #     learning_rate=learning_rate,
        #     momentum=momentum,
        #     batch_size=batch_size,
        #     use_global=True,
        #     keep_quantile=0.,
        #     train_thinning = 40,
        # )

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
        self.Sampler.sample(self.Prior.sample())

    def plot(self):
        pass