import jax.numpy as jnp
from jax.scipy.special import i0e
from jaxtyping import Array, Float

from jimgw.prior import Prior

EPS = 1e-15


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
    """
    A numerically stable method to evaluate arctan2 upon taking gradient.

    The gradient (jnp.jacfwd) of the default jnp.arctan2 is undefined 
    at (0, 0) and returns NaN. This function circumvent this issue by 
    specifying a default value at that point.

    Parameters
    ==========
    y: array-like
        y-coordinate of the point
    x: array-like
        x-coordinate of the point
    default_value: float
        arctan2 value to return at (0, 0), default is 0.0

    Returns
    =======
    array-like:
        The signed azimuthal angle, in radians, within [-π, π]
    """
    return jnp.where(
        (jnp.abs(x) < EPS) and (jnp.abs(y) < EPS),
        default_value * jnp.ones_like(x),
        jnp.atan2(y, x),
    )


def safe_polar_angle(
    x: Float[Array, " n"], y: Float[Array, " n"], z: Float[Array, " n"] 
) -> Float[Array, " n"]:
    """
    A numerically stable method to compute the polar angle upon taking gradient.

    The canonical computation is:
        theta = ArcCos[ z / Sqrt[x^2 + y^2 + z^2] ] .
    The gradient (jnp.jacfwd) of this method, however, 
    will return NaN when both x and y are 0.
    This function circumvent this issue by specifying computing the simplified 
    expression of the function at the troubled point.

    Parameters
    ==========
    x: array-like
        x-coordinate of the point
    y: array-like
        y-coordinate of the point
    z: array-like
        z-coordinate of the point

    Returns
    =======
    array-like:
        The polar angle, in radians, within [0, π]
    """
    return jnp.where(
        (jnp.abs(x) < EPS) and (jnp.abs(y) < EPS),
        jnp.arccos(jnp.sign(z)),
        jnp.arccos(z / jnp.sqrt(x**2 + y**2 + z**2)),
    )


def carte_to_spherical_angles(
    x: Float[Array, " n"], y: Float[Array, " n"], z: Float[Array, " n"],
    default_value: float = 0.0, 
) -> Float[Array, " n n"]:
    """
    A numerically stable method to compute the spherical angles upon taking gradient.

    For more details, see 
    * `safe_polar_angle` for the polar angle and 
    * `safe_arctan2` for the azimuthal angle.

    Parameters
    ==========
    x: array-like
        x-coordinate of the point
    y: array-like
        y-coordinate of the point
    z: array-like
        z-coordinate of the point
    default_value: float
        arctan2 value to return at (0, 0), default is 0.0

    Returns
    =======
    theta: array-like:
        The polar angle, in radians, within [0, π]
    phi: array-like:
        The signed azimuthal angle, in radians, within [-π, π]
    """
    align_condition = (jnp.absolute(x) < EPS) and (jnp.absolute(y) < EPS)
    theta = jnp.where(
        align_condition,
        jnp.arccos(jnp.sign(z)),
        jnp.arccos(z / jnp.sqrt(x**2 + y**2 + z**2)),
    )
    phi = jnp.where(
        align_condition,
        default_value * jnp.ones_like(x),
        jnp.atan2(y, x),
    )
    return theta, phi