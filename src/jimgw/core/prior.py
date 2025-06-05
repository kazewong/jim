from dataclasses import field

import jax
import jax.numpy as jnp
from jax.scipy.special import logit
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Bool, PRNGKeyArray, jaxtyped
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
    Base class for prior distributions.

    This class should not be used directly. It provides a common interface and bookkeeping for parameter names and transforms.
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
    Composite prior consisting of multiple priors, including SequentialTransformPrior and CombinePrior.
    This class is used to create complex prior distributions from simpler ones.

    Attributes:
        base_prior (list[Prior]): List of prior objects.
        parameter_names (list[str]): Names of all parameters in the composite prior.
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
    """
    One-dimensional logistic distribution prior.

    Attributes:
        parameter_names (list[str]): Name of the parameter.
    """

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
    """
    One-dimensional standard normal (Gaussian) distribution prior.

    Attributes:
        parameter_names (list[str]): Name of the parameter.
    """

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
    Prior distribution transformed by a sequence of bijective transforms.

    Attributes:
        base_prior (list[Prior]): The base prior to transform.
        transforms (list[BijectiveTransform]): List of transforms to apply sequentially.
        parameter_names (list[str]): Names of the parameters after all transforms.
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
        super().__init__(base_prior)
        self.transforms = transforms
        for transform in self.transforms:
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


class ConstrainedPrior(CompositePrior):
    """
    An abstract prior class that allow additional constraints be imposed on the parameter space.
    For inputs outside of the constrained support, `log_prob` returns the value `-inf`,
        for rejecting the undesired samples during sampling.

    This class wraps a base prior and applies additional constraints, which must be implemented by subclasses via the `constraints` method.
    The log_prob is set to -inf for any input that does not satisfy the constraints, and the sample method repeatedly draws from the base prior until enough valid samples are found.

    Warning:
        The `log_prob` method under the ConstrainedPrior may not be normalized to 1.
        This will be the case when the constrained support is a proper subset of
            the original support on which the density function is normalised.
        This means the resulting distribution may not be a true probability density function.

    Note:
        For the purpose of MCMC or similar inference methods, the prior need not be normalised,
            since only the probability ratios are needed and the evidence (normalization constant) is not computed.
    """

    def __repr__(self):
        return f"ConstrainedPrior(prior={self.base_prior})"

    def __init__(
        self,
        base_prior: list[Prior],
    ):
        assert len(base_prior) == 1, "ConstrainedPrior takes one base prior only"
        super().__init__(base_prior)

    @abstractmethod
    def constraints(self, x: dict[str, Float]) -> Bool:
        """
        Constraints to be applied to the parameter space.
        Subclasses should overwrite this method to define their constraints.
        """
        raise NotImplementedError

    def log_prob(self, z: dict[str, Float]) -> Float:
        return jnp.where(self.constraints(z), self.base_prior[0].log_prob(z), -jnp.inf)

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        rng_key, subkey = jax.random.split(rng_key)
        samples = self.base_prior[0].sample(subkey, n_samples)

        constraints = jax.vmap(self.constraints)

        mask = constraints(samples)
        valid_samples = jax.tree.map(lambda x: x[mask], samples)
        n_valid = mask.sum()

        # TODO: Add a loop count and a warning message for inefficient resampling,
        #       which is often likely be an issue in the prior than a small sample space.
        while n_valid < n_samples:
            rng_key, subkey = jax.random.split(rng_key)
            new_samples = self.base_prior[0].sample(subkey, n_samples - n_valid)
            new_mask = constraints(new_samples)
            valid_new_samples = jax.tree.map(lambda x: x[new_mask], new_samples)
            valid_samples = jax.tree.map(
                lambda x, y: jnp.concatenate([x, y], axis=0),
                valid_samples,
                valid_new_samples,
            )
            n_valid = valid_samples[list(valid_samples.keys())[0]].shape[0]

        return jax.tree.map(lambda x: x[:n_samples], valid_samples)


@jaxtyped(typechecker=typechecker)
class SimpleConstrainedPrior(ConstrainedPrior):
    """
    A prior class with a constraint being the bounds (xmin, xmax) of the base prior.

    This class wraps a 1D base prior and inspects it for `xmin` and `xmax` attributes.
    If these attributes are present, it constructs a constraint function that enforces the bounds on the parameter space.

    The constraints are enforced in both `log_prob` and `sample` methods via the parent `ConstrainedPrior` class.

    Note:
        - Only works with 1D priors.
        - The resulting prior is not normalized over the constrained region (see `ConstrainedPrior` for details).
    """

    xmin: float
    xmax: float

    def __repr__(self):
        return f"SimpleConstrainedPrior(prior={self.base_prior})"

    def __init__(
        self,
        base_prior: list[Prior],
    ):
        super().__init__(base_prior)

        # Add constraints for xmin/xmax if present
        p = self.base_prior[0]
        assert p.n_dims == 1, "SimpleConstrainedPrior only works with 1D priors"
        self.xmin = getattr(p, "xmin", -jnp.inf)
        self.xmax = getattr(p, "xmax", jnp.inf)

    def constraints(self, x: dict[str, Float]) -> Bool:
        variable = x[self.parameter_names[0]]
        return jnp.logical_and(variable >= self.xmin, variable <= self.xmax)


class CombinePrior(CompositePrior):
    """
    Multivariate prior constructed by joining multiple independent priors.

    Attributes:
        base_prior (list[Prior]): List of independent priors.
        parameter_names (list[str]): Names of all parameters in the combined prior.
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
        super().__init__(priors)

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
    """
    Uniform prior over a finite interval [xmin, xmax].

    Attributes:
        xmin (float): Lower bound of the interval.
        xmax (float): Upper bound of the interval.
        parameter_names (list[str]): Name of the parameter.
    """

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
    """
    Gaussian (normal) prior with specified mean and standard deviation.

    Attributes:
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation of the distribution.
        parameter_names (list[str]): Name of the parameter.
    """

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
    Prior with PDF proportional to sin(x) over [0, pi].

    Attributes:
        parameter_names (list[str]): Name of the parameter.
    """

    xmin: float = 0.0
    xmax: float = jnp.pi

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
    Prior with PDF proportional to cos(x) over [-pi/2, pi/2].

    Attributes:
        parameter_names (list[str]): Name of the parameter.
    """

    xmin: float = -jnp.pi / 2
    xmax: float = jnp.pi / 2

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
    """
    Uniform prior over a sphere, parameterized by magnitude, theta, and phi.

    Attributes:
        parameter_names (list[str]): Names of the vector, theta, and phi parameters.
    """

    def __repr__(self):
        return f"UniformSpherePrior(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str], max_mag: float = 1.0):
        self.parameter_names = parameter_names
        assert self.n_dims == 1, "UniformSpherePrior only takes the name of the vector"
        self.parameter_names = [
            f"{self.parameter_names[0]}_{suffix}" for suffix in ("mag", "theta", "phi")
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
    Rayleigh distribution prior with scale parameter sigma.

    Attributes:
        sigma (float): Scale parameter of the Rayleigh distribution.
        parameter_names (list[str]): Name of the parameter.
    """

    sigma: float
    xmin: float = 0.0

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
    """
    Power-law prior over [xmin, xmax] with exponent alpha.

    Attributes:
        xmin (float): Lower bound of the interval.
        xmax (float): Upper bound of the interval.
        alpha (float): Power-law exponent.
        parameter_names (list[str]): Name of the parameter.
    """

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
