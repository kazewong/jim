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
from flowMC.sampler.flowHMC import flowHMC

class Jim(object):
    """
    Master class for interfacing with flowMC
    
    """

    def __init__(self, likelihood: LikelihoodBase, prior: Prior, **kwargs):
        self.Likelihood = likelihood
        self.Prior = prior
        seed = kwargs.get("seed", 0)
        n_chains = kwargs.get("n_chains", 20)

        rng_key_set = initialize_rng_keys(n_chains, seed=seed)
        num_layers = kwargs.get("num_layers", 10)
        hidden_size = kwargs.get("hidden_size", [128,128])
        num_bins = kwargs.get("num_bins", 8)

        local_sampler_arg = kwargs.get("local_sampler_arg", {})

        local_sampler = MALA(self.posterior, True, local_sampler_arg) # Remember to add routine to find automated mass matrix

        model = MaskedCouplingRQSpline(self.Prior.n_dim, num_layers, hidden_size, num_bins, rng_key_set[-1])
        flowHMC_sampler = flowHMC(
            self.posterior,
            True,
            model,
            params={
                "step_size": 1e-2,
                "n_leapfrog": 5,
                "inverse_metric": jnp.ones(prior.n_dim),
            },
        )
        self.Sampler = Sampler(
            self.Prior.n_dim,
            rng_key_set,
            None,
            local_sampler,
            model,
            global_sampler = flowHMC_sampler,
            **kwargs)
        

    def maximize_likelihood(self, bounds: tuple[Array,Array], set_nwalkers: int = 100, n_loops: int = 2000, seed = 92348):
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

    def posterior(self, params: Array, data: dict):
        prior_params = self.Prior.add_name(params.T)
        prior  = self.Prior.log_prob(prior_params)
        return self.Likelihood.evaluate(self.Prior.transform(prior_params), data) + prior

    def sample(self, key: jax.random.PRNGKey,
               initial_guess: Array = None):
        if initial_guess is None:
            initial_guess = self.Prior.sample(key, self.Sampler.n_chains)
            initial_guess = jnp.stack([i for i in initial_guess.values()]).T
        self.Sampler.sample(initial_guess, None)

    def print_summary(self):
        """
        Generate summary of the run

        """

        train_summary = self.Sampler.get_sampler_state(training=True)
        production_summary = self.Sampler.get_sampler_state(training=False)

        training_chain: Array = train_summary["chains"]
        training_log_prob: Array = train_summary["log_prob"]
        training_local_acceptance: Array = train_summary["local_accs"]
        training_global_acceptance: Array = train_summary["global_accs"]
        training_loss: Array = train_summary["loss_vals"]

        production_chain: Array = production_summary["chains"]
        production_log_prob: Array = production_summary["log_prob"]
        production_local_acceptance: Array = production_summary["local_accs"]
        production_global_acceptance: Array = production_summary["global_accs"]

        print("Training summary")
        print('=' * 10)
        for index in range(len(self.Prior.naming)):
            print(f"{self.Prior.naming[index]}: {training_chain[:, :, index].mean():.3f} +/- {training_chain[:, :, index].std():.3f}")
        print(f"Log probability: {training_log_prob.mean():.3f} +/- {training_log_prob.std():.3f}") 
        print(f"Local acceptance: {training_local_acceptance.mean():.3f} +/- {training_local_acceptance.std():.3f}")
        print(f"Global acceptance: {training_global_acceptance.mean():.3f} +/- {training_global_acceptance.std():.3f}")
        print(f"Max loss: {training_loss.max():.3f}, Min loss: {training_loss.min():.3f}")

        print("Production summary")
        print('=' * 10)
        for index in range(len(self.Prior.naming)):
            print(f"{self.Prior.naming[index]}: {production_chain[:, :, index].mean():.3f} +/- {production_chain[:, :, index].std():.3f}")
        print(f"Log probability: {production_log_prob.mean():.3f} +/- {production_log_prob.std():.3f}")
        print(f"Local acceptance: {production_local_acceptance.mean():.3f} +/- {production_local_acceptance.std():.3f}")
        print(f"Global acceptance: {production_global_acceptance.mean():.3f} +/- {production_global_acceptance.std():.3f}")

    def get_samples(self, training: bool = False) -> dict:
        """
        Get the samples from the sampler

        Args:
            training (bool, optional): If True, return the training samples. Defaults to False.

        Returns:
            Array: Samples
        """
        if training:
            chains = self.Sampler.get_sampler_state(training=True)["chains"]
        else:
            chains = self.Sampler.get_sampler_state(training=False)["chains"]

        chains = self.Prior.add_name(chains.transpose(2,0,1), transform_name=True)
        return chains

    def plot(self):
        pass