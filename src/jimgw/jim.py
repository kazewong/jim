from jaxtyping import Array, Float, PRNGKeyArray
import jax
import jax.numpy as jnp

from flowMC.sampler.Sampler import Sampler
from flowMC.sampler.MALA import MALA
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.utils.EvolutionaryOptimizer import EvolutionaryOptimizer
from flowMC.sampler.flowHMC import flowHMC

from jimgw.prior import Prior
from jimgw.base import LikelihoodBase


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
        hidden_size = kwargs.get("hidden_size", [128, 128])
        num_bins = kwargs.get("num_bins", 8)

        local_sampler_arg = kwargs.get("local_sampler_arg", {})

        local_sampler = MALA(
            self.posterior, True, local_sampler_arg
        )  # Remember to add routine to find automated mass matrix

        flowHMC_params = kwargs.get("flowHMC_params", {})
        model = MaskedCouplingRQSpline(
            self.Prior.n_dim, num_layers, hidden_size, num_bins, rng_key_set[-1]
        )
        if len(flowHMC_params) > 0:
            global_sampler = flowHMC(
                self.posterior,
                True,
                model,
                params={
                    "step_size": flowHMC_params["step_size"],
                    "n_leapfrog": flowHMC_params["n_leapfrog"],
                    "condition_matrix": flowHMC_params["condition_matrix"],
                },
            )
        else:
            global_sampler = None

        self.Sampler = Sampler(
            self.Prior.n_dim,
            rng_key_set,
            None,  # type: ignore
            local_sampler,
            model,
            global_sampler=global_sampler,
            **kwargs,
        )

    def posterior(self, params: Float[Array, " n_dim"], data: dict):
        prior_params = self.Prior.add_name(params.T)
        prior = self.Prior.log_prob(prior_params)
        return (
            self.Likelihood.evaluate(self.Prior.transform(prior_params), data) + prior
        )

    def sample(self, key: PRNGKeyArray, initial_guess: Array = jnp.array([])):
        if initial_guess.size == 0:
            initial_guess_named = self.Prior.sample(key, self.Sampler.n_chains)
            initial_guess = jnp.stack([i for i in initial_guess_named.values()]).T
        self.Sampler.sample(initial_guess, None)  # type: ignore

    def maximize_likelihood(
        self,
        bounds: Float[Array, " n_dim 2"],
        set_nwalkers: int = 100,
        n_loops: int = 2000,
        seed=92348,
    ):
        key = jax.random.PRNGKey(seed)
        set_nwalkers = set_nwalkers
        initial_guess = self.Prior.sample(key, set_nwalkers)

        def negative_posterior(x: Float[Array, " n_dim"]):
            return -self.posterior(x, None)  # type: ignore since flowMC does not have typing info, yet

        negative_posterior = jax.jit(jax.vmap(negative_posterior))
        print("Compiling likelihood function")
        negative_posterior(initial_guess)
        print("Done compiling")

        print("Starting the optimizer")
        optimizer = EvolutionaryOptimizer(self.Prior.n_dim, verbose=True)
        _ = optimizer.optimize(negative_posterior, bounds, n_loops=n_loops)
        best_fit = optimizer.get_result()[0]
        return best_fit

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
        print("=" * 10)
        for index in range(len(self.Prior.naming)):
            print(
                f"{self.Prior.naming[index]}: {training_chain[:, :, index].mean():.3f} +/- {training_chain[:, :, index].std():.3f}"
            )
        print(
            f"Log probability: {training_log_prob.mean():.3f} +/- {training_log_prob.std():.3f}"
        )
        print(
            f"Local acceptance: {training_local_acceptance.mean():.3f} +/- {training_local_acceptance.std():.3f}"
        )
        print(
            f"Global acceptance: {training_global_acceptance.mean():.3f} +/- {training_global_acceptance.std():.3f}"
        )
        print(
            f"Max loss: {training_loss.max():.3f}, Min loss: {training_loss.min():.3f}"
        )

        print("Production summary")
        print("=" * 10)
        for index in range(len(self.Prior.naming)):
            print(
                f"{self.Prior.naming[index]}: {production_chain[:, :, index].mean():.3f} +/- {production_chain[:, :, index].std():.3f}"
            )
        print(
            f"Log probability: {production_log_prob.mean():.3f} +/- {production_log_prob.std():.3f}"
        )
        print(
            f"Local acceptance: {production_local_acceptance.mean():.3f} +/- {production_local_acceptance.std():.3f}"
        )
        print(
            f"Global acceptance: {production_global_acceptance.mean():.3f} +/- {production_global_acceptance.std():.3f}"
        )

    def get_samples(self, training: bool = False) -> dict:
        """
        Get the samples from the sampler

        Parameters
        ----------
        training : bool, optional
            Whether to get the training samples or the production samples, by default False

        Returns
        -------
        dict
            Dictionary of samples

        """
        if training:
            chains = self.Sampler.get_sampler_state(training=True)["chains"]
        else:
            chains = self.Sampler.get_sampler_state(training=False)["chains"]

        chains = self.Prior.transform(self.Prior.add_name(chains.transpose(2, 0, 1)))
        return chains

    def plot(self):
        pass
