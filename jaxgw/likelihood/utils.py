import jax.numpy as jnp

def inner_product(h1, h2, frequency, PSD, PSD_frequency):
	psd_interp = jnp.interp(frequency, PSD_frequency, PSD)
	df = frequency[1] - frequency[0]
	integrand = jnp.conj(h1)* h2 / psd_interp
	return 4. * jnp.real(jnp.sum(integrand)*df)
