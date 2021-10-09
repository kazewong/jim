import jax.numpy as jnp
from jaxgw.utils import *

def TaylorF2(f,params):
	local_m1 = params['mass_1']*Msun
	local_m2 = params['mass_2']*Msun
	local_d = params['luminosity_distance']*Mpc


	M_tot = local_m1+local_m2
	eta = local_m1*local_m2/(local_m1+local_m2)**2
	M_chirp = eta**(3./5)*M_tot
	PNcoef = (jnp.pi*M_tot*f)**(1./3)
	euler_gamma = 0.57721566490153286060	

	amplitude = M_chirp**(5./6)/local_d
	
	PN0 = 1.
	PN1 = (20./9) * (743./336 + 11./4*eta) * PNcoef**2
	PN1d5 = -16*jnp.pi*PNcoef**3
	PN2 = 10 * ((3058673./1016064)+ 5429./1008 *eta + 617./144 * eta**2) * PNcoef**4
	PN2d5 = jnp.pi*(38645./756-65./9*eta)*(1 + 3*jnp.log(6**(3./2)*jnp.pi*M_tot*f)) * PNcoef**5
#	PN3 = 11583231236531./4694215680 - 640./3 *jnp.pi**2 - 6868./21*(euler_gamma+jnp.log(4)

	phase = 2*jnp.pi*f*params['geocent_time'] - params['phase'] - jnp.pi/4 + 3./(128*eta*PNcoef**5) * \
			(PN0+PN1+PN1d5)#+PN2+PN2d5)

#	phase = - jnp.pi/4 + 3./(128*eta*PNcoef**5) * \
#			(PN0+PN1+PN1d5)#+PN2+PN2d5)



	totalh = jnp.sqrt(5./96)/jnp.pi**(2./3)*amplitude*f**(-7./6)*jnp.exp(1j*phase)
	hp = totalh * (1/2*(1+jnp.cos(params['theta_jn'])**2)*jnp.cos(2*params['psi']))
	hc = totalh * jnp.cos(params['theta_jn'])*jnp.sin(2*params['psi'])

	return {'plus':hp,'cross':hc}

def flso(M):
	return (6**3./2*jnp.pi*M)**-1
