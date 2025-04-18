import jax
import jax.numpy as jnp
from flowMC.resource_strategy_bundle.RQSpline_MALA_PT import RQSpline_MALA_PT_Bundle
from flowMC.resource.buffers import Buffer
from flowMC.Sampler import Sampler
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Optional

from jimgw.base import LikelihoodBase
from jimgw.prior import Prior
from jimgw.transforms import BijectiveTransform, NtoMTransform


class Jim(object):
    """
    Master class for interfacing with flowMC
    """

    likelihood: LikelihoodBase
    prior: Prior

    # Name of parameters to sample from
    sample_transforms: list[BijectiveTransform]
    likelihood_transforms: list[NtoMTransform]
    parameter_names: list[str]
    sampler: Sampler

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: list[BijectiveTransform] = [],
        likelihood_transforms: list[NtoMTransform] = [],
        rng_key: PRNGKeyArray = jax.random.PRNGKey(0),
        n_chains: int = 50,
        n_local_steps: int = 10,
        n_global_steps: int = 10,
        n_training_loops: int = 20,
        n_production_loops: int = 20,
        n_epochs: int = 20,
        mala_step_size: float = 0.01,
        rq_spline_hidden_units: list[int] = [128, 128],
        rq_spline_n_bins: int = 10,
        rq_spline_n_layers: int = 2,
        learning_rate: float = 1e-3,
        batch_size: int = 10000,
        n_max_examples: int = 10000,
        local_thinning: int = 1,
        global_thinning: int = 1,
        n_NFproposal_batch_size: int = 1000,
        history_window: int = 100,
        n_temperatures: int = 5,
        max_temperature: float = 10.0,
        n_tempered_steps: int = 5,
        verbose: bool = False,
    ):
        self.likelihood = likelihood
        self.prior = prior

        self.sample_transforms = sample_transforms
        self.likelihood_transforms = likelihood_transforms
        self.parameter_names = prior.parameter_names

        if len(sample_transforms) == 0:
            print(
                "No sample transforms provided. Using prior parameters as sampling parameters"
            )
        else:
            print("Using sample transforms")
            for transform in sample_transforms:
                self.parameter_names = transform.propagate_name(self.parameter_names)

        if len(likelihood_transforms) == 0:
            print(
                "No likelihood transforms provided. Using prior parameters as likelihood parameters"
            )

        if rng_key is jax.random.PRNGKey(0):
            print("No rng_key provided. Using default key with seed=0.")

        resource_strategy_bundle = RQSpline_MALA_PT_Bundle(
            rng_key=rng_key,
            n_chains=n_chains,
            n_dims=self.prior.n_dim,
            logpdf=self.posterior,
            n_local_steps=n_local_steps,
            n_global_steps=n_global_steps,
            n_training_loops=n_training_loops,
            n_production_loops=n_production_loops,
            n_epochs=n_epochs,
            mala_step_size=mala_step_size,
            rq_spline_hidden_units=rq_spline_hidden_units,
            rq_spline_n_bins=rq_spline_n_bins,
            rq_spline_n_layers=rq_spline_n_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_max_examples=n_max_examples,
            local_thinning=local_thinning,
            global_thinning=global_thinning,
            n_NFproposal_batch_size=n_NFproposal_batch_size,
            history_window=history_window,
            n_temperatures=n_temperatures,
            max_temperature=max_temperature,
            n_tempered_steps=n_tempered_steps,
            logprior=self.evaluate_prior,
            verbose=verbose,
        )

        rng_key, subkey = jax.random.split(rng_key)
        self.sampler = Sampler(
            self.prior.n_dim,
            n_chains,
            subkey,
            resource_strategy_bundles=resource_strategy_bundle,
        )

    def add_name(self, x: Float[Array, " n_dim"]) -> dict[str, Float]:
        """
        Turn an array into a dictionary

        Parameters
        ----------
        x : Array
            An array of parameters. Shape (n_dim,).
        """

        return dict(zip(self.parameter_names, x))

    def evaluate_prior(self, params: Float[Array, " n_dim"], data: dict):
        named_params = self.add_name(params)
        transform_jacobian = 0.0
        for transform in reversed(self.sample_transforms):
            named_params, jacobian = transform.inverse(named_params)
            transform_jacobian += jacobian
        return self.prior.log_prob(named_params) + transform_jacobian

    def posterior(self, params: Float[Array, " n_dim"], data: dict):
        named_params = self.add_name(params)
        transform_jacobian = 0.0
        for transform in reversed(self.sample_transforms):
            named_params, jacobian = transform.inverse(named_params)
            transform_jacobian += jacobian
        prior = self.prior.log_prob(named_params) + transform_jacobian
        for transform in self.likelihood_transforms:
            named_params = transform.forward(named_params)
        return self.likelihood.evaluate(named_params, data) + prior

    def sample(
        self,
        key: Optional[PRNGKeyArray] = None,
        initial_position: Array = jnp.array([]),
    ):
        if initial_position.size == 0:
            initial_position = (
                jnp.zeros((self.sampler.n_chains, self.prior.n_dim)) + jnp.nan
            )

            if key is not None:
                print("Provided key will override the existing sampler RNG key")
                key, self.sampler.rng_key = jax.random.split(key)

            while not jax.tree.reduce(
                jnp.logical_and,
                jax.tree.map(lambda x: jnp.isfinite(x), initial_position),
            ).all():
                non_finite_index = jnp.where(
                    jnp.any(
                        ~jax.tree.reduce(
                            jnp.logical_and,
                            jax.tree.map(lambda x: jnp.isfinite(x), initial_position),
                        ),
                        axis=1,
                    )
                )[0]

                key, subkey = jax.random.split(key)
                guess = self.prior.sample(subkey, self.sampler.n_chains)
                for transform in self.sample_transforms:
                    guess = jax.vmap(transform.forward)(guess)
                guess = jnp.array(
                    jax.tree.leaves({key: guess[key] for key in self.parameter_names})
                ).T
                finite_guess = jnp.where(
                    jnp.all(jax.tree.map(lambda x: jnp.isfinite(x), guess), axis=1)
                )[0]
                common_length = min(len(finite_guess), len(non_finite_index))
                initial_position = initial_position.at[
                    non_finite_index[:common_length]
                ].set(guess[:common_length])
        self.sampler.sample(initial_position, {})  # type: ignore

    def print_summary(self, transform: bool = True):
        """
        Generate summary of the run

        """

        train_summary = self.sampler.get_sampler_state(training=True)
        production_summary = self.sampler.get_sampler_state(training=False)

        training_chain = train_summary["chains"].reshape(-1, self.prior.n_dim).T
        training_chain = self.add_name(training_chain)
        if transform:
            for sample_transform in reversed(self.sample_transforms):
                training_chain = jax.vmap(sample_transform.backward)(training_chain)
        training_log_prob = train_summary["log_prob"]
        training_local_acceptance = train_summary["local_accs"]
        training_global_acceptance = train_summary["global_accs"]
        training_loss = train_summary["loss_vals"]

        production_chain = production_summary["chains"].reshape(-1, self.prior.n_dim).T
        production_chain = self.add_name(production_chain)
        if transform:
            for sample_transform in reversed(self.sample_transforms):
                production_chain = jax.vmap(sample_transform.backward)(production_chain)
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
            assert isinstance(
                chains := self.sampler.resources["positions_training"], Buffer
            )
            chains = chains.data
        else:
            assert isinstance(
                chains := self.sampler.resources["positions_production"], Buffer
            )
            chains = chains.data

        chains = chains.reshape(-1, self.prior.n_dim)
        chains = jax.vmap(self.add_name)(chains)
        for sample_transform in reversed(self.sample_transforms):
            chains = jax.vmap(sample_transform.backward)(chains)
        return chains

    def plot(self):
        pass
