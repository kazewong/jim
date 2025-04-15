import jax.numpy as jnp
from jax.scipy.special import i0e
from jaxtyping import Array, Float

from jimgw.prior import Prior

MAX_ATAN_TOL = 1.0e-15


def trace_prior_parent(prior: Prior, output: list[Prior] = []) -> list[Prior]:
    if prior.composite:
        if isinstance(prior.base_prior, list):
            for subprior in prior.base_prior:
                output = trace_prior_parent(subprior, output)
        elif isinstance(prior.base_prior, Prior):
            output = trace_prior_parent(prior.base_prior, output)
    else:
        output.append(prior)

    return output


def log_i0(x: Float[Array, " n"]) -> Float[Array, " n"]:
    """
    A numerically stable method to evaluate log of
    a modified Bessel function of order 0.
    It is used in the phase-marginalized likelihood.

    Parameters
    ==========
    x: array-like
        Value(s) at which to evaluate the function

    Returns
    =======
    array-like:
        The natural logarithm of the bessel function
    """
    return jnp.log(i0e(x)) + x


def safe_arctan2(
    y: Float[Array, " n"], x: Float[Array, " n"], default_value: float = 0.0
) -> Float[Array, " n"]:
    return jnp.where(
        (y < MAX_ATAN_TOL) and (x < MAX_ATAN_TOL),
        default_value * jnp.ones_like(x),
        jnp.atan2(y, x),
    )
