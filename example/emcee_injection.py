import numpy as np
import bilby
import jax
import jax.numpy as jnp
import time

from jax.config import config
config.update("jax_enable_x64", True)

from jaxgw.gw.likelihood.detector_projection import construct_arm, detector_tensor, antenna_response, get_detector_response

from jaxgw.gw.likelihood.utils import inner_product
from jaxgw.gw.likelihood.detector_preset import get_H1, get_L1
from jaxgw.gw.waveform.TaylorF2 import TaylorF2
from jaxgw.gw.waveform.IMRPhenomC import IMRPhenomC
from jax import random, grad, jit, vmap, jacfwd, jacrev, value_and_grad, pmap


true_m1 = 15.
true_m2 = 5.
true_ld = 600.
true_phase = 0.
true_gt = 0.

injection_parameters = dict(
	mass_1=true_m1, mass_2=true_m2, spin_1=0.0, spin_2=0.0, luminosity_distance=true_ld, theta_jn=0.4, psi=2.659,
	phase_c=true_phase, t_c=true_gt, ra=1.375, dec=-1.2108)


#guess_parameters = dict(m1=true_m1, m2=true_m2)

guess_parameters = dict(
	mass_1=true_m1, mass_2=true_m2, luminosity_distance=true_ld, theta_jn=0.4, psi=2.659,
	phase_c=true_phase, t_c=true_gt, ra=1.375, dec=-1.2108)



# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(['H1'])
ifos.set_strain_data_from_power_spectral_densities(
	sampling_frequency=2048, duration=1,
	start_time=- 3)

psd = ifos[0].power_spectral_density_array
psd_frequency = ifos[0].frequency_array

psd_frequency = psd_frequency[jnp.isfinite(psd)]
psd = psd[jnp.isfinite(psd)]

waveform = IMRPhenomC(psd_frequency, injection_parameters)
H1, H1_vertex = get_H1()
L1, L1_vertex = get_L1()
strain_H1 = get_detector_response(psd_frequency, waveform, injection_parameters, H1, H1_vertex)
strain_L1 = get_detector_response(psd_frequency, waveform, injection_parameters, L1, L1_vertex)

print('SNR of the event in H1: '+str(np.sqrt(inner_product(strain_H1,strain_H1,psd_frequency,psd))))
print('SNR of the event in L1: '+str(np.sqrt(inner_product(strain_L1,strain_L1,psd_frequency,psd))))

def single_detector_likelihood(params, data, data_f, PSD, detector, detector_vertex):
	waveform = IMRPhenomC(data_f, params)
#	waveform = TaylorF2(data_f, params)
	waveform = get_detector_response(data_f, waveform, params, detector, detector_vertex)
	match_filter_SNR = inner_product(waveform, data, data_f, PSD)
	optimal_SNR = inner_product(waveform, waveform, data_f, PSD)
	return (-2*match_filter_SNR + optimal_SNR)/2#, match_filter_SNR, optimal_SNR

#@jit
#def logprob_wrap(m1, m2):
#	params = dict(mass_1=m1, mass_2=m2, spin_1=0, spin_2=0, luminosity_distance=true_ld, phase_c=true_phase, t_c=true_gt, theta_jn=0.4, psi=2.659, ra=1.375, dec=-1.2108)
#	return single_detector_likelihood(params, strain_H1, psd_frequency, psd, H1, H1_vertex)+single_detector_likelihood(params, strain_L1, psd_frequency, psd, L1, L1_vertex)
#
def log_prob(params):
	if (params[0]<=0) or (params[1]<=0):
		return -jnp.inf
	params = dict(mass_1=params[0], mass_2=params[1], spin_1=0, spin_2=0, luminosity_distance=params[2], phase_c=params[3], t_c=params[4], theta_jn=params[5], psi=params[6], ra=params[7], dec=params[8])
	return single_detector_likelihood(params, strain_H1, psd_frequency, psd, H1, H1_vertex)+single_detector_likelihood(params, strain_L1, psd_frequency, psd, L1, L1_vertex)

################################################################
## BlackJax section
################################################################

import emcee 

nwalkers = 32
ndim = 9
p0 = np.random.rand(nwalkers, ndim) + list(guess_parameters.values())
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
state = sampler.run_mcmc(p0, 100)
sampler.reset()
sampler.run_mcmc(state, 5000)
