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


true_m1 = 5.
true_m2 = 5.
true_ld = 1000.
true_phase = 0.
true_gt = 0.

injection_parameters = dict(
	mass_1=true_m1, mass_2=true_m2, spin_1=0.0, spin_2=0.0, luminosity_distance=true_ld, theta_jn=0.4, psi=2.659,
	phase_c=true_phase, t_c=true_gt, ra=1.375, dec=-1.2108)


#guess_parameters = dict(m1=true_m1, m2=true_m2)

guess_parameters = dict(
	mass_1=true_m1*0.99, mass_2=true_m2*1.01, luminosity_distance=true_ld, theta_jn=0.4, psi=2.659,
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
#waveform = TaylorF2(psd_frequency, injection_parameters)
H1, H1_vertex = get_H1()
L1, L1_vertex = get_L1()
strain_H1 = get_detector_response(psd_frequency, waveform, injection_parameters, H1, H1_vertex)
strain_L1 = get_detector_response(psd_frequency, waveform, injection_parameters, L1, L1_vertex)

print('SNR of the event in H1: '+str(np.sqrt(inner_product(strain_H1,strain_H1,psd_frequency,psd))))
print('SNR of the event in L1: '+str(np.sqrt(inner_product(strain_L1,strain_L1,psd_frequency,psd))))

@jit
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
@jit
def logprob_wrap(mass_1, mass_2, luminosity_distance, phase_c, t_c, theta_jn, psi, ra, dec):
	params = dict(mass_1=mass_1, mass_2=mass_2, spin_1=0, spin_2=0, luminosity_distance=luminosity_distance, phase_c=phase_c, t_c=t_c, theta_jn=theta_jn, psi=psi, ra=ra, dec=dec)
	return single_detector_likelihood(params, strain_H1, psd_frequency, psd, H1, H1_vertex)+single_detector_likelihood(params, strain_L1, psd_frequency, psd, L1, L1_vertex)

log_prob = lambda x: logprob_wrap(**x)
log_prob = jit(log_prob)

################################################################
## BlackJax section
################################################################

import blackjax.hmc as hmc
import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup

rng_key = jax.random.PRNGKey(0)
key, subkey = random.split(rng_key)

initial_state = hmc.new_state(guess_parameters, log_prob)
print('Finding step size and mass matrix')

time1 = time.time()
kernel_generator = lambda step_size, inverse_mass_matrix: hmc.kernel(
	log_prob, step_size, inverse_mass_matrix, 5
)

final_state, (step_size, inverse_mass_matrix), info = stan_warmup.run(
	key,
	kernel_generator,
	initial_state,
	100,
	#is_mass_matrix_diagonal=False
)

print("Finding inverse mass matrix takes: "+str(time.time()-time1)+" seconds")
print("Stepsize: "+str(step_size))
print("Inverse mass matrix: "+str(inverse_mass_matrix))
num_integration_steps = 5

hmc_kernel = hmc.kernel(log_prob, step_size, inverse_mass_matrix, num_integration_steps)
hmc_kernel = jit(hmc_kernel)

test_likelihood = hmc_kernel(subkey, initial_state)
print("Energy of the first step is: "+str(test_likelihood[1].energy))

def inference_loop(rng_key, kernel, initial_state, num_samples):
	def one_step(state, rng_key):
		state, _ = kernel(rng_key, state)
		return state, state

	keys = jax.random.split(rng_key, num_samples)
	_, states = jax.lax.scan(one_step, initial_state, keys)

	return states

print("Start sampling")
time1 = time.time()
states = inference_loop(subkey, hmc_kernel, initial_state, 1000)
print("Sampling takes: "+str(time.time()-time1)+" seconds")
