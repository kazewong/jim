import jax.numpy as jnp
from jaxgw.gw.constants import *

def TaylorF2(f,params):
	local_m1 = params['mass_1']*Msun
	local_m2 = params['mass_2']*Msun
	local_d = params['luminosity_distance']*Mpc
	local_spin1 = params['spin_1']
	local_spin2 = params['spin_2']

	M_tot = local_m1+local_m2
	eta = local_m1*local_m2/(local_m1+local_m2)**2
	M_chirp = eta**(3./5)*M_tot
	chi_eff = (local_spin1*local_m1 + local_spin2*local_m2)/M_tot
	PNcoef = (jnp.pi*M_tot*f)**(1./3)

	# Flux coefficients
	FT_PN0 = 32.0 * eta*eta / 5.0
	# FT_PN1 = -(12.47/3.36 + 3.5/1.2 * eta)
	# FT_PN1d5 = 4 * jnp.pi
	# FT_PN2 = -(44.711/9.072 - 92.71/5.04 * eta - 6.5/1.8 * eta*eta)
	# FT_PN2d5 = -(81.91/6.72 + 58.3/2.4 * eta) * jnp.pi
	# FT_PN3 = (664.3739519/6.9854400 + 16.0/3.0 * jnp.pi*jnp.pi - 17.12/1.05 * euler_gamma - 17.12/1.05*jnp.log(4.) + (4.1/4.8 * jnp.pi*jnp.pi - 134.543/7.776) * eta - 94.403/3.024 * eta*eta - 7.75/3.24 * eta*eta*eta)
	# FT_PN3log = -17.12/1.05
	# FT_PN3d5 = -(162.85/5.04 - 214.745/1.728 * eta - 193.385/3.024 * eta*eta) * jnp.pi

	# Energy coefficients
	E_PN0 = 2. * -eta / 2.0
	# E_PN1 = 2. * -(0.75 + eta/12.0)
	# E_PN2 = 3. * -(27.0/8.0 - 19.0/8.0 * eta + 1./24.0 * eta*eta)
	# E_PN3 = 4. * -(67.5/6.4 - (344.45/5.76 - 20.5/9.6 * jnp.pi*jnp.pi) * eta + 15.5/9.6 * eta*eta + 3.5/518.4 * eta*eta*eta)

	

	amplitude = (-4. * local_m1 * local_m2 / local_d* jnp.sqrt(jnp.pi/12.))* jnp.sqrt(-(E_PN0*PNcoef)/(FT_PN0 * PNcoef**10)) * PNcoef

	Ph_PN0 = 1.
	Ph_PN1 = (20./9) * (743./336 + 11./4*eta) * PNcoef**2
	Ph_PN1d5 = -16*jnp.pi*PNcoef**3
	Ph_PN2 = (5.*(3058.673/7.056 + 5429./7.*eta+617.*eta*eta)/72.) * PNcoef**4
	Ph_PN2d5 = 5./9.*(772.9/8.4-13.*eta)*jnp.pi * PNcoef**5
	Ph_PN2d5_log = (5./3.*(772.9/8.4-13.*eta)*jnp.pi) * PNcoef**5 * jnp.log(PNcoef)
	Ph_PN3 = (11583.231236531/4.694215680 - 640./3.*jnp.pi*jnp.pi - 684.8/2.1*euler_gamma + eta*(-15737.765635/3.048192 + 225.5/1.2*jnp.pi*jnp.pi) + eta*eta*76.055/1.728 - eta*eta*eta*127.825/1.296) * PNcoef**6
	Ph_PN3_log = -684.8/2.1 * jnp.log(PNcoef) * PNcoef**6
	Ph_PN3d5 = jnp.pi*(770.96675/2.54016 + 378.515/1.512*eta - 740.45/7.56*eta*eta) * PNcoef**7

	phase = 2*jnp.pi*f*params['t_c'] - params['phase_c'] - jnp.pi/4 + 3./(128*eta*PNcoef**5) * \
			(Ph_PN0+Ph_PN1+Ph_PN1d5+Ph_PN2+Ph_PN2d5+Ph_PN2d5_log+Ph_PN3+Ph_PN3_log+Ph_PN3d5)
	

	totalh = amplitude * jnp.cos(phase) - amplitude * jnp.sin(phase) * 1j
	hp = totalh * (1/2*(1+jnp.cos(params['theta_jn'])**2)*jnp.cos(2*params['psi']))
	hc = totalh * jnp.cos(params['theta_jn'])*jnp.sin(2*params['psi'])

	return totalh#{'plus':hp,'cross':hc}

def flso(M):
	return (6**3./2*jnp.pi*M)**-1
