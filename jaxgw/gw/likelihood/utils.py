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
	return 4. * jnp.real(jnp.trapz(integrand,dx=df))

@jit
def m1m2_to_Mq(m1,m2):
	"""
	Transforming the primary mass m1 and secondary mass m2 to the Total mass M
	and mass ratio q.

	Args:
		m1: Primary mass of the binary.
		m2: Secondary mass of the binary.

	Returns:
		A tuple containing both the total mass M and mass ratio q.
	"""
	M_tot = jnp.log(m1+m2)
	q = jnp.log(m2/m1)-jnp.log(1-m2/m1)
	return M_tot, q

@jit
def Mq_to_m1m2(trans_M_tot,trans_q):
	M_tot = jnp.exp(trans_M_tot)
	q = 1./(1+jnp.exp(-trans_q))
	m1 = M_tot/(1+q)
	m2 = m1*q
	return m1, m2

