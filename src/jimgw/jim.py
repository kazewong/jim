import jax
import jax.numpy as jnp
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.proposal.MALA import MALA
from flowMC.Sampler import Sampler
from flowMC.utils.EvolutionaryOptimizer import EvolutionaryOptimizer
from jaxtyping import Array, Float, PRNGKeyArray

from jimgw.base import LikelihoodBase
from jimgw.prior import Prior


class Jim(object):
    """
    Master class for interfacing with flowMC
    """

    likelihood: LikelihoodBase
    prior: Prior

    parameter_names: list[str]
    sampler: Sampler

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        parameter_names: list[str],
        **kwargs,
    ):
        self.likelihood = likelihood
        self.prior = prior
        self.parameter_names = parameter_names

        seed = kwargs.get("seed", 0)

        rng_key = jax.random.PRNGKey(seed)
        num_layers = kwargs.get("num_layers", 10)
        hidden_size = kwargs.get("hidden_size", [128, 128])
        num_bins = kwargs.get("num_bins", 8)

        local_sampler_arg = kwargs.get("local_sampler_arg", {})

        local_sampler = MALA(self.posterior, True, **local_sampler_arg)

        rng_key, subkey = jax.random.split(rng_key)
        model = MaskedCouplingRQSpline(
            self.prior.n_dim, num_layers, hidden_size, num_bins, subkey
        )

        self.Sampler = Sampler(
            self.prior.n_dim,
            rng_key,
            None,  # type: ignore
            local_sampler,
            model,
            **kwargs,
        )

    def posterior(self, params: Float[Array, " n_dim"], data: dict):
        prior_params = self.prior.add_name(params.T)
        prior = self.prior.log_prob(prior_params)
        return (
            self.likelihood.evaluate(self.prior.transform(prior_params), data) + prior
        )

    def sample(self, key: PRNGKeyArray, initial_guess: Array = jnp.array([])):
        if initial_guess.size == 0:
            initial_guess_named = self.prior.sample(key, self.Sampler.n_chains)
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
        initial_guess = self.prior.sample(key, set_nwalkers)

        def negative_posterior(x: Float[Array, " n_dim"]):
            return -self.posterior(x, None)  # type: ignore since flowMC does not have typing info, yet

        negative_posterior = jax.jit(jax.vmap(negative_posterior))
        print("Compiling likelihood function")
        negative_posterior(initial_guess)
        print("Done compiling")

        print("Starting the optimizer")
        optimizer = EvolutionaryOptimizer(self.prior.n_dim, verbose=True)
        _ = optimizer.optimize(negative_posterior, bounds, n_loops=n_loops)
        best_fit = optimizer.get_result()[0]
        return best_fit

    def print_summary(self, transform: bool = True):
        """
        Generate summary of the run

        """

        train_summary = self.Sampler.get_sampler_state(training=True)
        production_summary = self.Sampler.get_sampler_state(training=False)

        training_chain = train_summary["chains"].reshape(-1, self.prior.n_dim).T
        training_chain = self.prior.add_name(training_chain)
        if transform:
            training_chain = self.prior.transform(training_chain)
        training_log_prob = train_summary["log_prob"]
        training_local_acceptance = train_summary["local_accs"]
        training_global_acceptance = train_summary["global_accs"]
        training_loss = train_summary["loss_vals"]

        production_chain = production_summary["chains"].reshape(-1, self.prior.n_dim).T
        production_chain = self.prior.add_name(production_chain)
        if transform:
            production_chain = self.prior.transform(production_chain)
        production_log_prob = production_summary["log_prob"]
        production_local_acceptance = production_summary["local_accs"]
        production_global_acceptance = production_summary["global_accs"]

        print("Training summary")
        print("=" * 10)
        for key, value in training_chain.items():
            print(f"{key}: {value.mean():.3f} +/- {value.std():.3f}")
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
        for key, value in production_chain.items():
            print(f"{key}: {value.mean():.3f} +/- {value.std():.3f}")
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

        chains = self.prior.transform(self.prior.add_name(chains.transpose(2, 0, 1)))
        return chains

    def plot(self):
        pass
