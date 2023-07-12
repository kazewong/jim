from jimgw.likelihood import LikelihoodBase
from flowMC.sampler.Sampler import Sampler
from flowMC.sampler.MALA import MALA
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.utils.EvolutionaryOptimizer import EvolutionaryOptimizer
from jimgw.prior import Prior
from jaxtyping import Array
import jax
import jax.numpy as jnp

class Jim(object):
    """ Master class for interfacing with flowMC
    
    """

    def __init__(self, likelihood: LikelihoodBase, prior: Prior, **kwargs):
        self.Likelihood = likelihood
        self.Prior = prior
        seed = kwargs.get("seed", 0)
        n_chains = kwargs.get("n_chains", 20)

        rng_key_set = initialize_rng_keys(n_chains, seed=seed)
        num_layers = kwargs.get("num_layers", 10)
        hidden_size = kwargs.get("hidden_size", [128,128])
        num_bins = kwargs.get("hidden_size", 8)

        def posterior(x: Array, data:dict):
            prior = self.Prior.log_prob(x)
            x = self.Prior.transform(x)
            return self.Likelihood.evaluate(x, data) + prior

        self.posterior = posterior

        local_sampler = MALA(posterior, True, {"step_size": 1e-2}) # Remember to add routine to find automated mass matrix

        model = MaskedCouplingRQSpline(self.Prior.n_dim, num_layers, hidden_size, num_bins, rng_key_set[-1])
        self.Sampler = Sampler(
            self.Prior.n_dim,
            rng_key_set,
            None,
            local_sampler,
            model,
            **kwargs)
        
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

    def maximize_likleihood(self, bounds: tuple[Array,Array],set_nwalkers: int = 100, n_loops: int = 2000, seed = 92348):
        bounds = jnp.array(bounds).T
        key = jax.random.PRNGKey(seed)
        set_nwalkers = set_nwalkers
        initial_guess = self.Prior.sample(key, set_nwalkers)

        y = lambda x: -self.posterior(x, None)
        y = jax.jit(jax.vmap(y))
        print("Compiling likelihood function")
        y(initial_guess)
        print("Done compiling")

        print("Starting the optimizer")
        optimizer = EvolutionaryOptimizer(self.Prior.n_dim, verbose = True)
        state = optimizer.optimize(y, bounds, n_loops=n_loops)
        best_fit = optimizer.get_result()[0]
        return best_fit


    def sample(self, key: jax.random.PRNGKey,
               initial_guess: Array = None):
        if initial_guess is None:
            initial_guess = self.Prior.sample(key, self.Sampler.n_chains)
        self.Sampler.sample(initial_guess, None)

    def plot(self):
        pass