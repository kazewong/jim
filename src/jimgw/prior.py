from dataclasses import field
from typing import Callable, Union

import jax
import jax.numpy as jnp
from astropy.time import Time
from beartype import beartype as typechecker
from flowMC.nfmodel.base import Distribution
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped

from jimgw.single_event.detector import GroundBased2G, detector_preset
from jimgw.single_event.utils import zenith_azimuth_to_ra_dec
from jimgw.transforms import Transform, Logit, Scale, Offset, ArcSine, ArcCosine


class Prior(Distribution):
    """
    A thin wrapper build on top of flowMC distributions to do book keeping.

    Should not be used directly since it does not implement any of the real method.

    The rationale behind this is to have a class that can be used to keep track of
    the names of the parameters and the transforms that are applied to them.
    """

    parameter_names: list[str]

    @property
    def n_dim(self):
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

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        raise NotImplementedError

    def log_prob(self, x: dict[str, Array]) -> Float:
        raise NotImplementedError


@jaxtyped(typechecker=typechecker)
class LogisticDistribution(Prior):

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

    def log_prob(self, x: dict[str, Float]) -> Float:
        variable = x[self.parameter_names[0]]
        return -variable - 2 * jnp.log(1 + jnp.exp(-variable))


@jaxtyped(typechecker=typechecker)
class StandardNormalDistribution(Prior):

    def __repr__(self):
        return f"StandardNormalDistribution(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str], **kwargs):
        super().__init__(parameter_names)
        assert self.n_dim == 1, "StandardNormalDistribution needs to be 1D distributions"

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

    def log_prob(self, x: dict[str, Float]) -> Float:
        variable = x[self.parameter_names[0]]
        return -0.5 * variable ** 2 - 0.5 * jnp.log(2 * jnp.pi)

class SequentialTransform(Prior):
    """
    Transform a prior distribution by applying a sequence of transforms.
    """

    base_prior: Prior
    transforms: list[Transform]

    def __repr__(self):
        return f"Sequential(priors={self.base_prior}, parameter_names={self.parameter_names})"

    def __init__(
        self,
        base_prior: Prior,
        transforms: list[Transform],
    ):

        self.base_prior = base_prior
        self.transforms = transforms
        self.parameter_names = base_prior.parameter_names
        for transform in transforms:
            self.parameter_names = transform.propagate_name(self.parameter_names)

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        output = self.base_prior.sample(rng_key, n_samples)
        return jax.vmap(self.transform)(output)

    def log_prob(self, x: dict[str, Float]) -> Float:
        """
        log_prob has to be evaluated in the space of the base_prior.
        """
        output = self.base_prior.log_prob(x)
        for transform in self.transforms:
            x, log_jacobian = transform.transform(x)
            output -= log_jacobian
        return output

    def sample_base(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        return self.base_prior.sample(rng_key, n_samples)

    def transform(self, x: dict[str, Float]) -> dict[str, Float]:
        for transform in self.transforms:
            x = transform.forward(x)
        return x


class Combine(Prior):
    """
    A prior class constructed by joinning multiple priors together to form a multivariate prior.
    This assumes the priors composing the Combine class are independent.
    """

    priors: list[Prior] = field(default_factory=list)

    def __repr__(self):
        return (
            f"Composite(priors={self.priors}, parameter_names={self.parameter_names})"
        )

    def __init__(
        self,
        priors: list[Prior],
    ):
        parameter_names = []
        for prior in priors:
            parameter_names += prior.parameter_names
        self.priors = priors
        self.parameter_names = parameter_names

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        output = {}
        for prior in self.priors:
            rng_key, subkey = jax.random.split(rng_key)
            output.update(prior.sample(subkey, n_samples))
        return output

    def log_prob(self, x: dict[str, Float]) -> Float:
        output = 0.0
        for prior in self.priors:
            output += prior.log_prob(x)
        return output


@jaxtyped(typechecker=typechecker)
class Uniform(SequentialTransform):
    xmin: float
    xmax: float

    def __repr__(self):
        return f"Uniform(xmin={self.xmin}, xmax={self.xmax}, parameter_names={self.parameter_names})"

    def __init__(
        self,
        xmin: float,
        xmax: float,
        parameter_names: list[str],
    ):
        self.parameter_names = parameter_names
        assert self.n_dim == 1, "Uniform needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin
        super().__init__(
            LogisticDistribution(self.parameter_names),
            [
                Logit((self.parameter_names, self.parameter_names)),
                Scale((self.parameter_names, self.parameter_names), xmax - xmin),
                Offset((self.parameter_names, self.parameter_names), xmin),
            ],
        )


@jaxtyped(typechecker=typechecker)
class Sine(SequentialTransform):
    """
    A prior distribution where the pdf is proportional to sin(x) in the range [0, pi].
    """

    def __repr__(self):
        return f"Sine(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str]):
        self.parameter_names = parameter_names
        assert self.n_dim == 1, "Sine needs to be 1D distributions"
        super().__init__(
            Uniform(-1.0, 1.0, f"cos_{self.parameter_names}"),
            [ArcCosine(([f"cos_{self.parameter_names}"], [self.parameter_names]))],
        )


@jaxtyped(typechecker=typechecker)
class Cosine(SequentialTransform):
    """
    A prior distribution where the pdf is proportional to cos(x) in the range [-pi/2, pi/2].
    """

    def __repr__(self):
        return f"Cosine(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str]):
        self.parameter_names = parameter_names
        assert self.n_dim == 1, "Cosine needs to be 1D distributions"
        super().__init__(
            Uniform(-1.0, 1.0, f"sin_{self.parameter_names}"),
            [ArcSine(([f"sin_{self.parameter_names}"], [self.parameter_names]))],
        )


@jaxtyped(typechecker=typechecker)
class UniformSphere(Combine):

    def __repr__(self):
        return f"UniformSphere(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str], **kwargs):
        assert (
            len(parameter_names) == 1
        ), "UniformSphere only takes the name of the vector"
        parameter_names = parameter_names[0]
        self.parameter_names = [
            f"{parameter_names}_mag",
            f"{parameter_names}_theta",
            f"{parameter_names}_phi",
        ]
        super().__init__(
            [
                Uniform(0.0, 1.0, [self.parameter_names[0]]),
                Sine([self.parameter_names[1]]),
                Uniform(0.0, 2 * jnp.pi, [self.parameter_names[2]]),
            ]
        )


# ====================== Things below may need rework ======================


@jaxtyped(typechecker=typechecker)
class AlignedSpin(Prior):
    """
    Prior distribution for the aligned (z) component of the spin.

    This assume the prior distribution on the spin magnitude to be uniform in [0, amax]
    with its orientation uniform on a sphere

    p(chi) = -log(|chi| / amax) / 2 / amax

    This is useful when comparing results between an aligned-spin run and
    a precessing spin run.

    See (A7) of https://arxiv.org/abs/1805.10457.
    """

    amax: Float = 0.99
    chi_axis: Array = field(default_factory=lambda: jnp.linspace(0, 1, num=1000))
    cdf_vals: Array = field(default_factory=lambda: jnp.linspace(0, 1, num=1000))

    def __repr__(self):
        return f"Alignedspin(amax={self.amax}, naming={self.naming})"

    def __init__(
        self,
        amax: Float,
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(naming, transforms)
        assert self.n_dim == 1, "Alignedspin needs to be 1D distributions"
        self.amax = amax

        # build the interpolation table for the ppf of the one-sided distribution
        chi_axis = jnp.linspace(1e-31, self.amax, num=1000)
        cdf_vals = -chi_axis * (jnp.log(chi_axis / self.amax) - 1.0) / self.amax
        self.chi_axis = chi_axis
        self.cdf_vals = cdf_vals

    @property
    def xmin(self):
        return -self.amax

    @property
    def xmax(self):
        return self.amax

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from the Alignedspin distribution.

        for chi > 0;
        p(chi) = -log(chi / amax) / amax  # halved normalization constant
        cdf(chi) = -chi * (log(chi / amax) - 1) / amax

        Since there is a pole at chi=0, we will sample with the following steps
        1. Map the samples with quantile > 0.5 to positive chi and negative otherwise
        2a. For negative chi, map the quantile back to [0, 1] via q -> 2(0.5 - q)
        2b. For positive chi, map the quantile back to [0, 1] via q -> 2(q - 0.5)
        3. Map the quantile to chi via the ppf by checking against the table
           built during the initialization
        4. add back the sign

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
        q_samples = jax.random.uniform(rng_key, (n_samples,), minval=0.0, maxval=1.0)
        # 1. calculate the sign of chi from the q_samples
        sign_samples = jnp.where(
            q_samples >= 0.5,
            jnp.zeros_like(q_samples) + 1.0,
            jnp.zeros_like(q_samples) - 1.0,
        )
        # 2. remap q_samples
        q_samples = jnp.where(
            q_samples >= 0.5,
            2 * (q_samples - 0.5),
            2 * (0.5 - q_samples),
        )
        # 3. map the quantile to chi via interpolation
        samples = jnp.interp(
            q_samples,
            self.cdf_vals,
            self.chi_axis,
        )
        # 4. add back the sign
        samples *= sign_samples

        return self.add_name(samples[None])

    def log_prob(self, x: dict[str, Float]) -> Float:
        variable = x[self.naming[0]]
        log_p = jnp.where(
            (variable >= self.amax) | (variable <= -self.amax),
            jnp.zeros_like(variable) - jnp.inf,
            jnp.log(-jnp.log(jnp.absolute(variable) / self.amax) / 2.0 / self.amax),
        )
        return log_p


@jaxtyped(typechecker=typechecker)
class EarthFrame(Prior):
    """
    Prior distribution for sky location in Earth frame.
    """

    ifos: list = field(default_factory=list)
    gmst: float = 0.0
    delta_x: Float[Array, " 3"] = field(default_factory=lambda: jnp.zeros(3))

    def __repr__(self):
        return f"EarthFrame(naming={self.naming})"

    def __init__(self, gps: Float, ifos: list, **kwargs):
        self.naming = ["zenith", "azimuth"]
        if len(ifos) < 2:
            return ValueError(
                "At least two detectors are needed to define the Earth frame"
            )
        elif isinstance(ifos[0], str):
            self.ifos = [detector_preset[ifos[0]], detector_preset[ifos[1]]]
        elif isinstance(ifos[0], GroundBased2G):
            self.ifos = ifos[:1]
        else:
            return ValueError(
                "ifos should be a list of detector names or GroundBased2G objects"
            )
        self.gmst = float(
            Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad
        )
        self.delta_x = self.ifos[1].vertex - self.ifos[0].vertex

        self.transforms = {
            "azimuth": (
                "ra",
                lambda params: zenith_azimuth_to_ra_dec(
                    params["zenith"],
                    params["azimuth"],
                    gmst=self.gmst,
                    delta_x=self.delta_x,
                )[0],
            ),
            "zenith": (
                "dec",
                lambda params: zenith_azimuth_to_ra_dec(
                    params["zenith"],
                    params["azimuth"],
                    gmst=self.gmst,
                    delta_x=self.delta_x,
                )[1],
            ),
        }

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        rng_keys = jax.random.split(rng_key, 2)
        zenith = jnp.arccos(
            jax.random.uniform(rng_keys[0], (n_samples,), minval=-1.0, maxval=1.0)
        )
        azimuth = jax.random.uniform(
            rng_keys[1], (n_samples,), minval=0, maxval=2 * jnp.pi
        )
        return self.add_name(jnp.stack([zenith, azimuth], axis=1).T)

    def log_prob(self, x: dict[str, Float]) -> Float:
        zenith = x["zenith"]
        azimuth = x["azimuth"]
        output = jnp.where(
            (zenith > jnp.pi) | (zenith < 0) | (azimuth > 2 * jnp.pi) | (azimuth < 0),
            jnp.zeros_like(0) - jnp.inf,
            jnp.zeros_like(0),
        )
        return output + jnp.log(jnp.sin(zenith))


@jaxtyped(typechecker=typechecker)
class PowerLaw(Prior):
    """
    A prior following the power-law with alpha in the range [xmin, xmax).
    p(x) ~ x^{\alpha}
    """

    xmin: float = 0.0
    xmax: float = 1.0
    alpha: float = 0.0
    normalization: float = 1.0

    def __repr__(self):
        return f"Powerlaw(xmin={self.xmin}, xmax={self.xmax}, alpha={self.alpha}, naming={self.naming})"

    def __init__(
        self,
        xmin: float,
        xmax: float,
        alpha: Union[Int, float],
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(naming, transforms)
        if alpha < 0.0:
            assert xmin > 0.0, "With negative alpha, xmin must > 0"
        assert self.n_dim == 1, "Powerlaw needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin
        self.alpha = alpha
        if alpha == -1:
            self.normalization = float(1.0 / jnp.log(self.xmax / self.xmin))
        else:
            self.normalization = (1 + self.alpha) / (
                self.xmax ** (1 + self.alpha) - self.xmin ** (1 + self.alpha)
            )

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from a power-law distribution.

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
        q_samples = jax.random.uniform(rng_key, (n_samples,), minval=0.0, maxval=1.0)
        if self.alpha == -1:
            samples = self.xmin * jnp.exp(q_samples * jnp.log(self.xmax / self.xmin))
        else:
            samples = (
                self.xmin ** (1.0 + self.alpha)
                + q_samples
                * (self.xmax ** (1.0 + self.alpha) - self.xmin ** (1.0 + self.alpha))
            ) ** (1.0 / (1.0 + self.alpha))
        return self.add_name(samples[None])

    def log_prob(self, x: dict[str, Float]) -> Float:
        variable = x[self.naming[0]]
        log_in_range = jnp.where(
            (variable >= self.xmax) | (variable <= self.xmin),
            jnp.zeros_like(variable) - jnp.inf,
            jnp.zeros_like(variable),
        )
        log_p = self.alpha * jnp.log(variable) + jnp.log(self.normalization)
        return log_p + log_in_range


@jaxtyped(typechecker=typechecker)
class Exponential(Prior):
    """
    A prior following the power-law with alpha in the range [xmin, xmax).
    p(x) ~ exp(\alpha x)
    """

    xmin: float = 0.0
    xmax: float = jnp.inf
    alpha: float = -1.0
    normalization: float = 1.0

    def __repr__(self):
        return f"Exponential(xmin={self.xmin}, xmax={self.xmax}, alpha={self.alpha}, naming={self.naming})"

    def __init__(
        self,
        xmin: Float,
        xmax: Float,
        alpha: Union[Int, Float],
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(naming, transforms)
        if alpha < 0.0:
            assert xmin != -jnp.inf, "With negative alpha, xmin must finite"
        if alpha > 0.0:
            assert xmax != jnp.inf, "With positive alpha, xmax must finite"
        assert not jnp.isclose(alpha, 0.0), "alpha=zero is given, use Uniform instead"
        assert self.n_dim == 1, "Exponential needs to be 1D distributions"

        self.xmax = xmax
        self.xmin = xmin
        self.alpha = alpha

        self.normalization = self.alpha / (
            jnp.exp(self.alpha * self.xmax) - jnp.exp(self.alpha * self.xmin)
        )

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from a exponential distribution.

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
        q_samples = jax.random.uniform(rng_key, (n_samples,), minval=0.0, maxval=1.0)
        samples = (
            self.xmin
            + jnp.log1p(
                q_samples * (jnp.exp(self.alpha * (self.xmax - self.xmin)) - 1.0)
            )
            / self.alpha
        )
        return self.add_name(samples[None])

    def log_prob(self, x: dict[str, Float]) -> Float:
        variable = x[self.naming[0]]
        log_in_range = jnp.where(
            (variable >= self.xmax) | (variable <= self.xmin),
            jnp.zeros_like(variable) - jnp.inf,
            jnp.zeros_like(variable),
        )
        log_p = self.alpha * variable + jnp.log(self.normalization)
        return log_p + log_in_range
