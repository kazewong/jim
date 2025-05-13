from dataclasses import field
from typing import Callable

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Bool, PRNGKeyArray, jaxtyped
from abc import abstractmethod
import equinox as eqx

from jimgw.transforms import (
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
    def n_dim(self) -> int:
        return len(self.parameter_names)

    def __init__(self, parameter_names: list[str]):
        """
        Parameters
        ----------
        parameter_names : list[str]
            A list of names for the parameters of the prior.
        """
        self.parameter_names = parameter_names

    def add_name(self, x: Float[Array, " n_dim"]) -> dict[str, Float]:
        """
        Turn an array into a dictionary

        Parameters
        ----------
        x : Array
            An array of parameters. Shape (n_dim,).
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
    Composite prior consisting of multiple independent priors.

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
        assert self.n_dim == 1, "LogisticDistribution needs to be 1D distributions"

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
        samples = jnp.log(samples / (1 - samples))
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
            self.n_dim == 1
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
        return -0.5 * variable**2 - 0.5 * jnp.log(2 * jnp.pi)


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


@jaxtyped(typechecker=typechecker)
class FullRangePrior(Prior):
    """
    Prior that enforces constraints on the parameter range, returning -inf if constraints are not satisfied.

    Attributes:
        base_prior (SequentialTransformPrior): The base prior distribution.
        constraints (list[Callable]): List of constraint functions to apply.
    """

    constraints: list[Callable]

    def __init__(
        self,
        base_prior: SequentialTransformPrior,
        extra_constraints: list[Callable] = [],
    ):
        super().__init__(base_prior.parameter_names)
        object.__setattr__(self, "base_prior", base_prior)
        # Copy the constraints list to avoid mutating the input list
        self.constraints = list(extra_constraints)
        # --- Closure helper for constraints ---
        def _make_bound_constraint(name, bound, is_min: bool):
            if is_min:
                def constraint(z):
                    return z[name] > bound
            else:
                def constraint(z):
                    return z[name] < bound
            return constraint
        # Add constraints for xmin/xmax if present
        if isinstance(base_prior, CombinePrior):
            # Handle CombinePrior
            for i, name in enumerate(self.parameter_names):
                subprior = base_prior.base_prior[i]
                if hasattr(subprior, "xmin"):
                    xmin = getattr(subprior, "xmin")
                    self.constraints.append(_make_bound_constraint(name, xmin, True))
                if hasattr(subprior, "xmax"):
                    xmax = getattr(subprior, "xmax")
                    self.constraints.append(_make_bound_constraint(name, xmax, False))
        elif self.n_dim == 1:
            # Handle 1D case
            if hasattr(base_prior, "xmin"):
                xmin = getattr(base_prior, "xmin")
                self.constraints.append(_make_bound_constraint(self.parameter_names[0], xmin, True))
            if hasattr(base_prior, "xmax"):
                xmax = getattr(base_prior, "xmax")
                self.constraints.append(_make_bound_constraint(self.parameter_names[0], xmax, False))

    def eval_constraints(self, x: dict[str, Float]) -> Bool:
        return jnp.array([constraint(x) for constraint in self.constraints]).all()

    def log_prob(self, z: dict[str, Float]) -> Float:
        eval_result = self.eval_constraints(z)
        return jnp.where(eval_result, self.base_prior.log_prob(z), -jnp.inf)

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        rng_key, subkey = jax.random.split(rng_key)
        samples = self.base_prior.sample(subkey, n_samples)

        mask = jax.vmap(self.eval_constraints)(samples)
        valid_samples = jax.tree.map(lambda x: x[mask], samples)
        n_valid = mask.sum()

        while n_valid < n_samples:
            rng_key, subkey = jax.random.split(rng_key)
            new_samples = self.base_prior.sample(subkey, n_samples - n_valid)
            new_mask = jax.vmap(self.eval_constraints)(new_samples)
            valid_new_samples = jax.tree.map(lambda x: x[new_mask], new_samples)
            valid_samples = jax.tree.map(
                lambda x, y: jnp.concatenate([x, y], axis=0),
                valid_samples,
                valid_new_samples,
            )
            n_valid = valid_samples[list(valid_samples.keys())[0]].shape[0]

        return jax.tree.map(lambda x: x[:n_samples], valid_samples)


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
        assert self.n_dim == 1, "UniformPrior needs to be 1D distributions"
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
        assert self.n_dim == 1, "GaussianPrior needs to be 1D distributions"
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
        assert self.n_dim == 1, "SinePrior needs to be 1D distributions"
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
        assert self.n_dim == 1, "CosinePrior needs to be 1D distributions"
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
        assert self.n_dim == 1, "UniformSpherePrior only takes the name of the vector"
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
    Rayleigh distribution prior with scale parameter sigma.

    Attributes:
        sigma (float): Scale parameter of the Rayleigh distribution.
        parameter_names (list[str]): Name of the parameter.
    """

    xmin: float = 0.0
    sigma: float

    def __repr__(self):
        return f"RayleighPrior(parameter_names={self.parameter_names})"

    def __init__(
        self,
        sigma: float,
        parameter_names: list[str],
    ):
        self.parameter_names = parameter_names
        assert self.n_dim == 1, "RayleighPrior needs to be 1D distributions"
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
        assert self.n_dim == 1, "Power law needs to be 1D distributions"
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


# ====================== Things below may need rework ======================

# @jaxtyped(typechecker=typechecker)
# class Exponential(Prior):
#     """
#     A prior following the power-law with alpha in the range [xmin, xmax).
#     p(x) ~ exp(\alpha x)
#     """

#     xmin: float = 0.0
#     xmax: float = jnp.inf
#     alpha: float = -1.0
#     normalization: float = 1.0

#     def __repr__(self):
#         return f"Exponential(xmin={self.xmin}, xmax={self.xmax}, alpha={self.alpha}, naming={self.naming})"

#     def __init__(
#         self,
#         xmin: Float,
#         xmax: Float,
#         alpha: Union[Int, Float],
#         naming: list[str],
#         transforms: dict[str, tuple[str, Callable]] = {},
#         **kwargs,
#     ):
#         super().__init__(naming, transforms)
#         if alpha < 0.0:
#             assert xmin != -jnp.inf, "With negative alpha, xmin must finite"
#         if alpha > 0.0:
#             assert xmax != jnp.inf, "With positive alpha, xmax must finite"
#         assert not jnp.isclose(alpha, 0.0), "alpha=zero is given, use Uniform instead"
#         assert self.n_dim == 1, "Exponential needs to be 1D distributions"

#         self.xmax = xmax
#         self.xmin = xmin
#         self.alpha = alpha

#         self.normalization = self.alpha / (
#             jnp.exp(self.alpha * self.xmax) - jnp.exp(self.alpha * self.xmin)
#         )

#     def sample(
#         self, rng_key: PRNGKeyArray, n_samples: int
#     ) -> dict[str, Float[Array, " n_samples"]]:
#         """
#         Sample from a exponential distribution.

#         Parameters
#         ----------
#         rng_key : PRNGKeyArray
#             A random key to use for sampling.
#         n_samples : int
#             The number of samples to draw.

#         Returns
#         -------
#         samples : dict
#             Samples from the distribution. The keys are the names of the parameters.

#         """
#         q_samples = jax.random.uniform(rng_key, (n_samples,), minval=0.0, maxval=1.0)
#         samples = (
#             self.xmin
#             + jnp.log1p(
#                 q_samples * (jnp.exp(self.alpha * (self.xmax - self.xmin)) - 1.0)
#             )
#             / self.alpha
#         )
#         return self.add_name(samples[None])

#     def log_prob(self, x: dict[str, Float]) -> Float:
#         variable = x[self.naming[0]]
#         log_in_range = jnp.where(
#             (variable >= self.xmax) | (variable <= self.xmin),
#             jnp.zeros_like(variable) - jnp.inf,
#             jnp.zeros_like(variable),
#         )
#         log_p = self.alpha * variable + jnp.log(self.normalization)
#         return log_p + log_in_range
