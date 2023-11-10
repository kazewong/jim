import jax
import jax.numpy as jnp
from flowMC.nfmodel.base import Distribution
from jaxtyping import Array, Float
from typing import Callable
from dataclasses import field


class Prior(Distribution):
    """
    A thin wrapper build on top of flowMC distributions to do book keeping.

    Should not be used directly since it does not implement any of the real method.

    The rationale behind this is to have a class that can be used to keep track of
    the names of the parameters and the transforms that are applied to them.
    """

    naming: list[str]
    transforms: dict[tuple[str, Callable]] = field(default_factory=dict)

    @property
    def n_dim(self):
        return len(self.naming)

    def __init__(self, naming: list[str], transforms: dict[tuple[str, Callable]] = {}):
        """
        Parameters
        ----------
        naming : list[str]
            A list of names for the parameters of the prior.
        transforms : dict[tuple[str,Callable]]
            A dictionary of transforms to apply to the parameters. The keys are
            the names of the parameters and the values are a tuple of the name
            of the transform and the transform itself.
        """
        self.naming = naming
        self.transforms = {}

        def make_lambda(name):
            return lambda x: x[name]

        for name in naming:
            if name in transforms:
                self.transforms[name] = transforms[name]
            else:
                # Without the function, the lambda will refer to the variable name instead of its value,
                # which will make lambda reference the last value of the variable name
                self.transforms[name] = (name, make_lambda(name))

    def transform(self, x: dict) -> dict:
        """
        Apply the transforms to the parameters.

        Parameters
        ----------
        x : dict
            A dictionary of parameters. Names should match the ones in the prior.

        Returns
        -------
        x : dict
            A dictionary of parameters with the transforms applied.
        """
        # output = self.add_name(x, transform_name=False, transform_value=False)
        for i, (key, value) in enumerate(self.transforms.items()):
            x[key] = value[1](x)
        return x

    def add_name(
        self, x: Array, transform_name: bool = False, transform_value: bool = False
    ) -> dict:
        """
        Turn an array into a dictionary

        Parameters
        ----------
        x : Array
            An array of parameters. Shape (n_dim, n_sample).
        """
        if transform_name:
            naming = [value[0] for value in self.transforms.values()]
        else:
            naming = self.naming
        x = dict(zip(naming, x))
        if transform_value:
            return self.transform(x)
        else:
            return x

    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> dict:
        raise NotImplementedError

    def logpdf(self, x: dict) -> Float:
        raise NotImplementedError


class Uniform(Prior):

    xmin: float = 0.0
    xmax: float = 1.0

    def __init__(
        self,
        xmin: float,
        xmax: float,
        naming: list[str],
        transforms: dict[tuple[str, Callable]] = {},
    ):
        super().__init__(naming, transforms)
        assert isinstance(xmin, float), "xmin must be a float"
        assert isinstance(xmax, float), "xmax must be a float"
        assert self.n_dim == 1, "Uniform needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin

    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> dict:
        """
        Sample from a uniform distribution.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Samples from the distribution. The keys are the names of the parameters.

        """
        samples = jax.random.uniform(
            rng_key, (n_samples,), minval=self.xmin, maxval=self.xmax
        )
        return self.add_name(samples[None])

    def log_prob(self, x: dict) -> Float:
        variable = x[self.naming[0]]
        output = jnp.where(
            (variable >= self.xmax) | (variable <= self.xmin),
            jnp.zeros_like(variable) - jnp.inf,
            jnp.zeros_like(variable),
        )
        return output + jnp.log(1.0 / (self.xmax - self.xmin))


class Unconstrained_Uniform(Prior):

    xmin: float = 0.0
    xmax: float = 1.0

    def __init__(
        self,
        xmin: float,
        xmax: float,
        naming: list[str],
        transforms: dict[tuple[str, Callable]] = {},
    ):
        super().__init__(naming, transforms)
        assert isinstance(xmin, float), "xmin must be a float"
        assert isinstance(xmax, float), "xmax must be a float"
        assert self.n_dim == 1, "Unconstrained_Uniform needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin
        local_transform = self.transforms
        def new_transform(param):
            result = (self.xmax - self.xmin) / (1 + jnp.exp(-param[self.naming[0]])) + self.xmin
            return local_transform[self.naming[0]][1]({self.naming[0]:result})
        self.transforms = {
            self.naming[0]: (
                self.naming[0],
                new_transform
            )
        }

    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> dict:
        """
        Sample from a uniform distribution.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : 
            An array of shape (n_samples, n_dim) containing the samples.

        """
        samples = jax.random.uniform(rng_key, (n_samples,), minval=0, maxval=1)
        samples = jnp.log(samples / (1 - samples))
        return self.add_name(samples[None])

    def log_prob(self, x: dict) -> Float:
        variable = x[self.naming[0]]
        y = 1.0 / (1 + jnp.exp(-variable))
        return jnp.log((1 / (self.xmax - self.xmin)) * (1 / (y - y * y)))

class Sphere(Prior):

    def __init__(self, naming: list[str], transforms: dict[tuple[str, Callable]] = {}):
        super().__init__(naming, transforms)

    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
        return super().sample(rng_key, n_samples)
    
    def log_prob(self, x: dict) -> Float:
        return super().log_prob(x)

class Composite(Prior):

    priors: list[Prior] = field(default_factory=list)

    def __init__(self, priors: list[Prior], **kwargs):
        naming = []
        transforms = {}
        for prior in priors:
            naming += prior.naming
            transforms.update(prior.transforms)
        self.priors = priors
        self.naming = naming
        self.transforms = transforms

    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> dict:
        output = {}
        for prior in self.priors:
            rng_key, subkey = jax.random.split(rng_key)
            output.update(prior.sample(subkey, n_samples))
        return output

    def log_prob(self, x: dict) -> Float:
        output = 0.0
        for prior in self.priors:
            output += prior.log_prob(x)
        return output