import jax
import jax.numpy as jnp
from flowMC.resource_strategy_bundle.RQSpline_MALA_PT import RQSpline_MALA_PT_Bundle
from flowMC.resource.buffers import Buffer
from flowMC.Sampler import Sampler
from jaxtyping import Array, Float, PRNGKeyArray

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

        rng_key, subkey = jax.random.split(rng_key)

        resource_strategy_bundle = RQSpline_MALA_PT_Bundle(
            rng_key=subkey,
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
        initial_position: Array = jnp.array([]),
    ):
        if initial_position.size == 0:
            rng_key, subkey = jax.random.split(self.sampler.rng_key)

            named_initial_position = self.prior.sample(subkey, self.sampler.n_chains)
            for transform in self.sample_transforms:
                named_initial_position = jax.vmap(transform.forward)(
                    named_initial_position
                )
            initial_position = jnp.array([named_initial_position[key] for key in self.parameter_names]).T
            assert jnp.isnan(initial_position).sum() == 0, (
                "Initial position contains NaN values. "
                "Please check the prior and sample transforms."
            )
            self.sampler.rng_key = rng_key
        else:
            assert initial_position.ndim == 2, "Initial position must be a 2D array."
            assert initial_position.shape[0] == self.sampler.n_chains, (
                f"Initial position must have {self.sampler.n_chains} rows, "
                f"but got {initial_position.shape[0]}."
            )
            assert initial_position.shape[1] == self.prior.n_dim, (
                f"Initial position must have {self.prior.n_dim} columns, "
                f"but got {initial_position.shape[1]}."
            )

        self.sampler.sample(initial_position, {})  # type: ignore

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
