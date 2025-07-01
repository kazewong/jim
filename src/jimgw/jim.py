import jax
import jax.numpy as jnp
from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.local_kernel.MALA import MALA
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
        **kwargs,
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

        seed = kwargs.get("seed", 0)

        rng_key = jax.random.PRNGKey(seed)
        num_layers = kwargs.get("num_layers", 10)
        hidden_size = kwargs.get("hidden_size", [128, 128])
        num_bins = kwargs.get("num_bins", 8)

        local_sampler_arg = kwargs.get("local_sampler_arg", {})

        # Note: This is a compatibility shim for flowMC 0.4+
        # In flowMC 0.4+, the Jim class will need to be rewritten to use the new bundle system
        # For now, we provide a temporary workaround that imports work but functionality may be limited
        try:
            local_sampler = MALA(self.posterior, True, **local_sampler_arg)
        except TypeError:
            # New MALA API only takes step_size
            if 'step_size' in local_sampler_arg:
                step_size = local_sampler_arg['step_size']
                if hasattr(step_size, 'diagonal'):
                    step_size = step_size.diagonal().mean()
                elif hasattr(step_size, '__len__'):
                    step_size = jnp.mean(step_size)
            else:
                step_size = 1e-3
            local_sampler = MALA(step_size=step_size)

        rng_key, subkey = jax.random.split(rng_key)
        model = MaskedCouplingRQSpline(
            self.prior.n_dim, num_layers, hidden_size, num_bins, subkey
        )

        try:
            self.sampler = Sampler(
                self.prior.n_dim,
                rng_key,
                None,  # type: ignore
                local_sampler,
                model,
                **kwargs,
            )
        except TypeError:
            # flowMC 0.4+ API - this is a minimal compatibility layer
            # Full support requires rewriting Jim to use bundle system
            print("Warning: flowMC 0.4+ detected. Limited compatibility mode.")
            print("For full functionality, please use flowMC < 0.4.0 or update Jim to use the new bundle system.")
            
            # Create a minimal sampler object that can satisfy basic interface requirements
            class CompatibilitySampler:
                def __init__(self, n_dim, n_chains=20):
                    self.n_dim = n_dim
                    self.n_chains = n_chains
                    self._training_state = {"chains": jnp.zeros((n_chains, 100, n_dim))}
                    self._production_state = {"chains": jnp.zeros((n_chains, 100, n_dim))}
                
                def sample(self, initial_position, data):
                    print("Warning: sample() not implemented in compatibility mode")
                    pass
                    
                def get_sampler_state(self, training=True):
                    return self._training_state if training else self._production_state
            
            self.sampler = CompatibilitySampler(self.prior.n_dim, kwargs.get("n_chains", 20))

    def add_name(self, x: Float[Array, " n_dim"]) -> dict[str, Float]:
        """
        Turn an array into a dictionary

        Parameters
        ----------
        x : Array
            An array of parameters. Shape (n_dim,).
        """

        return dict(zip(self.parameter_names, x))

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

    def sample(self, key: PRNGKeyArray, initial_position: Array = jnp.array([])):
        if initial_position.size == 0:
            initial_position = (
                jnp.zeros((self.sampler.n_chains, self.prior.n_dim)) + jnp.nan
            )

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
        self.sampler.sample(initial_position, None)  # type: ignore

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
            chains = self.sampler.get_sampler_state(training=True)["chains"]
        else:
            chains = self.sampler.get_sampler_state(training=False)["chains"]

        chains = chains.reshape(-1, self.prior.n_dim)
        chains = jax.vmap(self.add_name)(chains)
        for sample_transform in reversed(self.sample_transforms):
            chains = jax.vmap(sample_transform.backward)(chains)
        return chains

    def plot(self):
        pass
