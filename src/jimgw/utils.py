import jax.numpy as jnp
from jax.scipy.special import i0e
from jaxtyping import Array, Float


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
