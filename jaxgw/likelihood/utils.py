import jax.numpy as jnp
from jax import jit

@jit
def inner_product(h1, h2, frequency, PSD):
    """
    Do PSD interpolation outside the inner product loop to speed up the evaluation
    """
    #psd_interp = jnp.interp(frequency, PSD_frequency, PSD)
    df = frequency[1] - frequency[0]
    integrand = jnp.conj(h1)* h2 / PSD
    return 4. * jnp.real(jnp.sum(integrand)*df)
