from dataclasses import field

import jax

from beartype import beartype as typechecker
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from jimgw.transforms import BijectiveTransform
from jimgw.prior import (
    Prior,
    UniformPrior,
    CombinePrior,
)
from jimgw.single_event.transforms import (
    UniformComponentMassSecondaryMassQuantileToSecondaryMassTransform,
)
from jimgw.single_event.utils import (
    Mc_q_to_m1_m2,
)


@jaxtyped(typechecker=typechecker)
class ChirpMassMassRatioBoundedUniformComponentPrior(CombinePrior):

    M_c_min: float = 5.0
    M_c_max: float = 15.0
    q_min: float = 0.125
    q_max: float = 1.0

    m_1_min: float = 6.0
    m_1_max: float = 53.0
    m_2_min: float = 3.0
    m_2_max: float = 17.0

    base_prior: list[Prior] = field(default_factory=list)
    transform: BijectiveTransform = (
        UniformComponentMassSecondaryMassQuantileToSecondaryMassTransform(
            q_min=q_min,
            q_max=q_max,
            M_c_min=M_c_min,
            M_c_max=M_c_max,
            m_1_min=m_1_min,
            m_1_max=m_1_max,
        )
    )

    def __init__(self, q_min: Float, q_max: Float, M_c_min: Float, M_c_max: Float):
        self.parameter_names = ["m_1", "m_2"]
        # calculate the respective range of m1 and m2 given the Mc-q range
        self.M_c_min = M_c_min
        self.M_c_max = M_c_max
        self.q_min = q_min
        self.q_max = q_max
        self.m_1_min = Mc_q_to_m1_m2(M_c_min, q_max)[0]
        self.m_1_max = Mc_q_to_m1_m2(M_c_max, q_min)[0]
        self.m_2_min = Mc_q_to_m1_m2(M_c_min, q_min)[1]
        self.m_2_max = Mc_q_to_m1_m2(M_c_max, q_max)[1]
        # define the prior on m1 and m2_quantile
        m1_prior = UniformPrior(self.m_1_min, self.m_1_max, parameter_names=["m_1"])
        m2q_prior = UniformPrior(0.0, 1.0, parameter_names=["m_2_quantile"])
        self.base_prior = [m1_prior, m2q_prior]
        self.transform = (
            UniformComponentMassSecondaryMassQuantileToSecondaryMassTransform(
                q_min=self.q_min,
                q_max=self.q_max,
                M_c_min=self.M_c_min,
                M_c_max=self.M_c_max,
                m_1_min=self.m_1_min,
                m_1_max=self.m_1_max,
            )
        )

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        output = {}
        for prior in self.base_prior:
            rng_key, subkey = jax.random.split(rng_key)
            output.update(prior.sample(subkey, n_samples))
        output = jax.vmap(self.transform.forward)(output)
        return output

    def log_prob(self, z: dict[str, Float]) -> Float:
        z, jacobian = self.transform.inverse(z)
        output = jacobian
        for prior in self.base_prior:
            output += prior.log_prob(z)
        return output


# ====================== Things below may need rework ======================


# @jaxtyped(typechecker=typechecker)
# class AlignedSpin(Prior):
#     """
#     Prior distribution for the aligned (z) component of the spin.

#     This assume the prior distribution on the spin magnitude to be uniform in [0, amax]
#     with its orientation uniform on a sphere

#     p(chi) = -log(|chi| / amax) / 2 / amax

#     This is useful when comparing results between an aligned-spin run and
#     a precessing spin run.

#     See (A7) of https://arxiv.org/abs/1805.10457.
#     """

#     amax: Float = 0.99
#     chi_axis: Array = field(default_factory=lambda: jnp.linspace(0, 1, num=1000))
#     cdf_vals: Array = field(default_factory=lambda: jnp.linspace(0, 1, num=1000))

#     def __repr__(self):
#         return f"Alignedspin(amax={self.amax}, naming={self.naming})"

#     def __init__(
#         self,
#         amax: Float,
#         naming: list[str],
#         transforms: dict[str, tuple[str, Callable]] = {},
#         **kwargs,
#     ):
#         super().__init__(naming, transforms)
#         assert self.n_dim == 1, "Alignedspin needs to be 1D distributions"
#         self.amax = amax

#         # build the interpolation table for the ppf of the one-sided distribution
#         chi_axis = jnp.linspace(1e-31, self.amax, num=1000)
#         cdf_vals = -chi_axis * (jnp.log(chi_axis / self.amax) - 1.0) / self.amax
#         self.chi_axis = chi_axis
#         self.cdf_vals = cdf_vals

#     @property
#     def xmin(self):
#         return -self.amax

#     @property
#     def xmax(self):
#         return self.amax

#     def sample(
#         self, rng_key: PRNGKeyArray, n_samples: int
#     ) -> dict[str, Float[Array, " n_samples"]]:
#         """
#         Sample from the Alignedspin distribution.

#         for chi > 0;
#         p(chi) = -log(chi / amax) / amax  # halved normalization constant
#         cdf(chi) = -chi * (log(chi / amax) - 1) / amax

#         Since there is a pole at chi=0, we will sample with the following steps
#         1. Map the samples with quantile > 0.5 to positive chi and negative otherwise
#         2a. For negative chi, map the quantile back to [0, 1] via q -> 2(0.5 - q)
#         2b. For positive chi, map the quantile back to [0, 1] via q -> 2(q - 0.5)
#         3. Map the quantile to chi via the ppf by checking against the table
#            built during the initialization
#         4. add back the sign

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
#         # 1. calculate the sign of chi from the q_samples
#         sign_samples = jnp.where(
#             q_samples >= 0.5,
#             jnp.zeros_like(q_samples) + 1.0,
#             jnp.zeros_like(q_samples) - 1.0,
#         )
#         # 2. remap q_samples
#         q_samples = jnp.where(
#             q_samples >= 0.5,
#             2 * (q_samples - 0.5),
#             2 * (0.5 - q_samples),
#         )
#         # 3. map the quantile to chi via interpolation
#         samples = jnp.interp(
#             q_samples,
#             self.cdf_vals,
#             self.chi_axis,
#         )
#         # 4. add back the sign
#         samples *= sign_samples

#         return self.add_name(samples[None])

#     def log_prob(self, x: dict[str, Float]) -> Float:
#         variable = x[self.naming[0]]
#         log_p = jnp.where(
#             (variable >= self.amax) | (variable <= -self.amax),
#             jnp.zeros_like(variable) - jnp.inf,
#             jnp.log(-jnp.log(jnp.absolute(variable) / self.amax) / 2.0 / self.amax),
#         )
#         return log_p
