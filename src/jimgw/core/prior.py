from dataclasses import field

import jax
import jax.numpy as jnp
from jax.scipy.special import logit
from beartype import beartype as typechecker
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped
from abc import abstractmethod
import equinox as eqx

from jimgw.core.transforms import (
    BijectiveTransform,
    LogitTransform,
    ScaleTransform,
    OffsetTransform,
    SineTransform,
    PowerLawTransform,
    RayleighTransform,
    reverse_bijective_transform,
)


class Prior(eqx.Module):
    """
    A base class for prior distributions.

    Should not be used directly since it does not implement any of the real method.

    The rationale behind this is to have a class that can be used to keep track of
    the names of the parameters and the transforms that are applied to them.
    """

    parameter_names: list[str]

    @property
    def n_dims(self) -> int:
        return len(self.parameter_names)

    def __init__(self, parameter_names: list[str]):
        """
        Parameters
        ----------
        parameter_names : list[str]
            A list of names for the parameters of the prior.
        """
        self.parameter_names = parameter_names

    def add_name(self, x: Float[Array, " n_dims"]) -> dict[str, Float]:
        """
        Turn an array into a dictionary

        Parameters
        ----------
        x : Array
            An array of parameters. Shape (n_dims,).
        """

        return dict(zip(self.parameter_names, x))

    def __call__(self, x: dict[str, Float]) -> Float:
        return self.log_prob(x)

    @abstractmethod
    def log_prob(self, z: dict[str, Float]) -> Float:
        raise NotImplementedError

    @abstractmethod
    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        raise NotImplementedError


@jaxtyped(typechecker=typechecker)
class CompositePrior(Prior):
    """
    A prior class that is a composite of multiple priors.
    """

    base_prior: list[Prior]

    def __repr__(self):
        return f"Composite(priors={self.base_prior}, parameter_names={self.parameter_names})"

    def __init__(
        self,
        priors: list[Prior],
    ):
        parameter_names = []
        for prior in priors:
            parameter_names += prior.parameter_names
        self.base_prior = priors
        self.parameter_names = parameter_names

    def trace_prior_parent(self, output: list[Prior] = []) -> list[Prior]:
        for subprior in self.base_prior:
            if isinstance(subprior, CompositePrior):
                output = subprior.trace_prior_parent(output)
            else:
                output.append(subprior)
        return output


@jaxtyped(typechecker=typechecker)
class LogisticDistribution(Prior):

    def __repr__(self):
        return f"LogisticDistribution(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str], **kwargs):
        super().__init__(parameter_names)
        assert self.n_dims == 1, "LogisticDistribution needs to be 1D distributions"

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from a logistic distribution.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Samples from the distribution. The keys are the names of the parameters.

        """
        samples = jax.random.uniform(rng_key, (n_samples,), minval=0.0, maxval=1.0)
        samples = logit(samples)
        return self.add_name(samples[None])

    def log_prob(self, z: dict[str, Float]) -> Float:
        variable = z[self.parameter_names[0]]
        return -variable - 2 * jnp.log(1 + jnp.exp(-variable))


@jaxtyped(typechecker=typechecker)
class StandardNormalDistribution(Prior):

    def __repr__(self):
        return f"StandardNormalDistribution(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str], **kwargs):
        super().__init__(parameter_names)
        assert (
            self.n_dims == 1
        ), "StandardNormalDistribution needs to be 1D distributions"

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from a standard normal distribution.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Samples from the distribution. The keys are the names of the parameters.

        """
        samples = jax.random.normal(rng_key, (n_samples,))
        return self.add_name(samples[None])

    def log_prob(self, z: dict[str, Float]) -> Float:
        variable = z[self.parameter_names[0]]
        return -0.5 * (variable**2 + jnp.log(2 * jnp.pi))


class SequentialTransformPrior(CompositePrior):
    """
    Transform a prior distribution by applying a sequence of transforms.
    The space before the transform is named as x,
    and the space after the transform is named as z
    """

    transforms: list[BijectiveTransform]

    def __repr__(self):
        return f"Sequential(priors={self.base_prior}, parameter_names={self.parameter_names})"

    def __init__(
        self,
        base_prior: list[Prior],
        transforms: list[BijectiveTransform],
    ):

        assert (
            len(base_prior) == 1
        ), "SequentialTransformPrior only takes one base prior"
        self.base_prior = base_prior
        self.transforms = transforms
        self.parameter_names = base_prior[0].parameter_names
        for transform in transforms:
            self.parameter_names = transform.propagate_name(self.parameter_names)

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        output = self.base_prior[0].sample(rng_key, n_samples)
        return jax.vmap(self.transform)(output)

    def log_prob(self, z: dict[str, Float]) -> Float:
        """
        Evaluating the probability of the transformed variable z.
        This is what flowMC should sample from
        """
        output = 0
        for transform in reversed(self.transforms):
            z, log_jacobian = transform.inverse(z)
            output += log_jacobian
        output += self.base_prior[0].log_prob(z)
        return output

    def transform(self, x: dict[str, Float]) -> dict[str, Float]:
        for transform in self.transforms:
            x = transform.forward(x)
        return x


class CombinePrior(CompositePrior):
    """
    A prior class constructed by joinning multiple priors together to form a multivariate prior.
    This assumes the priors composing the Combine class are independent.
    """

    base_prior: list[Prior] = field(default_factory=list)

    def __repr__(self):
        return (
            f"Combine(priors={self.base_prior}, parameter_names={self.parameter_names})"
        )

    def __init__(
        self,
        priors: list[Prior],
    ):
        parameter_names = []
        for prior in priors:
            parameter_names += prior.parameter_names
        self.base_prior = priors
        self.parameter_names = parameter_names

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        output = {}
        for prior in self.base_prior:
            rng_key, subkey = jax.random.split(rng_key)
            output.update(prior.sample(subkey, n_samples))
        return output

    def log_prob(self, z: dict[str, Float]) -> Float:
        output = 0.0
        for prior in self.base_prior:
            output += prior.log_prob(z)
        return output


@jaxtyped(typechecker=typechecker)
class UniformPrior(SequentialTransformPrior):
    xmin: float
    xmax: float

    def __repr__(self):
        return f"UniformPrior(xmin={self.xmin}, xmax={self.xmax}, parameter_names={self.parameter_names})"

    def __init__(
        self,
        xmin: float,
        xmax: float,
        parameter_names: list[str],
    ):
        self.parameter_names = parameter_names
        assert self.n_dims == 1, "UniformPrior needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin
        super().__init__(
            [LogisticDistribution([f"{self.parameter_names[0]}_base"])],
            [
                LogitTransform(
                    (
                        [f"{self.parameter_names[0]}_base"],
                        [f"({self.parameter_names[0]}-({xmin}))/{(xmax-xmin)}"],
                    )
                ),
                ScaleTransform(
                    (
                        [f"({self.parameter_names[0]}-({xmin}))/{(xmax-xmin)}"],
                        [f"{self.parameter_names[0]}-({xmin})"],
                    ),
                    xmax - xmin,
                ),
                OffsetTransform(
                    ([f"{self.parameter_names[0]}-({xmin})"], self.parameter_names),
                    xmin,
                ),
            ],
        )


@jaxtyped(typechecker=typechecker)
class GaussianPrior(SequentialTransformPrior):
    mu: float
    sigma: float

    def __repr__(self):
        return f"GaussianPrior(mu={self.mu}, sigma={self.sigma}, parameter_names={self.parameter_names})"

    def __init__(
        self,
        mu: float,
        sigma: float,
        parameter_names: list[str],
    ):
        """
        A convenient wrapper distribution on top of the StandardNormalDistribution class
        which scale and translate the distribution according to the mean and standard deviation.

        Args
            mu: The mean of the distribution.
            sigma: The standard deviation of the distribution.
            parameter_names: A list of names for the parameters of the prior.
        """
        self.parameter_names = parameter_names
        assert self.n_dims == 1, "GaussianPrior needs to be 1D distributions"
        self.mu = mu
        self.sigma = sigma
        super().__init__(
            [StandardNormalDistribution([f"{self.parameter_names[0]}_base"])],
            [
                ScaleTransform(
                    (
                        [f"{self.parameter_names[0]}_base"],
                        [f"{self.parameter_names[0]}-({mu})"],
                    ),
                    sigma,
                ),
                OffsetTransform(
                    ([f"{self.parameter_names[0]}-({mu})"], self.parameter_names),
                    mu,
                ),
            ],
        )


@jaxtyped(typechecker=typechecker)
class SinePrior(SequentialTransformPrior):
    """
    A prior distribution where the pdf is proportional to sin(x) in the range [0, pi].
    """

    def __repr__(self):
        return f"SinePrior(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str]):
        self.parameter_names = parameter_names
        assert self.n_dims == 1, "SinePrior needs to be 1D distributions"
        super().__init__(
            [CosinePrior([f"{self.parameter_names[0]}-pi/2"])],
            [
                OffsetTransform(
                    (
                        (
                            [f"{self.parameter_names[0]}-pi/2"],
                            [f"{self.parameter_names[0]}"],
                        )
                    ),
                    jnp.pi / 2,
                )
            ],
        )


@jaxtyped(typechecker=typechecker)
class CosinePrior(SequentialTransformPrior):
    """
    A prior distribution where the pdf is proportional to cos(x) in the range [-pi/2, pi/2].
    """

    def __repr__(self):
        return f"CosinePrior(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str]):
        self.parameter_names = parameter_names
        assert self.n_dims == 1, "CosinePrior needs to be 1D distributions"
        super().__init__(
            [UniformPrior(-1.0, 1.0, [f"sin({self.parameter_names[0]})"])],
            [
                reverse_bijective_transform(
                    SineTransform(
                        (
                            [f"{self.parameter_names[0]}"],
                            [f"sin({self.parameter_names[0]})"],
                        )
                    )
                )
            ],
        )


@jaxtyped(typechecker=typechecker)
class UniformSpherePrior(CombinePrior):

    def __repr__(self):
        return f"UniformSpherePrior(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str], max_mag: float = 1.0):
        self.parameter_names = parameter_names
        assert self.n_dims == 1, "UniformSpherePrior only takes the name of the vector"
        self.parameter_names = [
            f"{self.parameter_names[0]}_mag",
            f"{self.parameter_names[0]}_theta",
            f"{self.parameter_names[0]}_phi",
        ]
        super().__init__(
            [
                UniformPrior(0.0, max_mag, [self.parameter_names[0]]),
                SinePrior([self.parameter_names[1]]),
                UniformPrior(0.0, 2 * jnp.pi, [self.parameter_names[2]]),
            ]
        )


@jaxtyped(typechecker=typechecker)
class RayleighPrior(SequentialTransformPrior):
    """
    A prior distribution following the Rayleigh distribution with scale parameter sigma.
    """

    sigma: float

    def __repr__(self):
        return f"RayleighPrior(parameter_names={self.parameter_names})"

    def __init__(
        self,
        sigma: float,
        parameter_names: list[str],
    ):
        self.parameter_names = parameter_names
        assert self.n_dims == 1, "RayleighPrior needs to be 1D distributions"
        self.sigma = sigma
        super().__init__(
            [UniformPrior(0.0, 1.0, [f"{self.parameter_names[0]}_base"])],
            [
                RayleighTransform(
                    ([f"{self.parameter_names[0]}_base"], self.parameter_names),
                    sigma=sigma,
                )
            ],
        )


@jaxtyped(typechecker=typechecker)
class PowerLawPrior(SequentialTransformPrior):
    xmin: float
    xmax: float
    alpha: float

    def __repr__(self):
        return f"PowerLawPrior(xmin={self.xmin}, xmax={self.xmax}, alpha={self.alpha}, naming={self.parameter_names})"

    def __init__(
        self,
        xmin: float,
        xmax: float,
        alpha: float,
        parameter_names: list[str],
    ):
        self.parameter_names = parameter_names
        assert self.n_dims == 1, "Power law needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin
        self.alpha = alpha
        assert self.xmin < self.xmax, "xmin must be less than xmax"
        assert self.xmin > 0.0, "x must be positive"
        super().__init__(
            [LogisticDistribution([f"{self.parameter_names[0]}_base"])],
            [
                LogitTransform(
                    (
                        [f"{self.parameter_names[0]}_base"],
                        [f"{self.parameter_names[0]}_before_transform"],
                    )
                ),
                PowerLawTransform(
                    (
                        [f"{self.parameter_names[0]}_before_transform"],
                        self.parameter_names,
                    ),
                    xmin,
                    xmax,
                    alpha,
                ),
            ],
        )
