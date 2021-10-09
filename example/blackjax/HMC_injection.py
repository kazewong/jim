import numpy as np
import bilby
import jax
import jax.numpy as jnp
import time

from jax.config import config
config.update("jax_enable_x64", True)

from jaxgw.likelihood.detector_projection import construct_arm, detector_tensor, antenna_response, get_detector_response

from jaxgw.likelihood.utils import inner_product
from jaxgw.waveform.TaylorF2 import TaylorF2
from jaxgw.waveform.IMRPhenomC import IMRPhenomC
from jax import random, grad, jit, vmap, jacfwd, jacrev, value_and_grad

true_m1 = 3.
true_m2 = 3.
true_ld = 1000


injection_parameters = dict(
    mass_1=true_m1, mass_2=true_m2, a_1=0.0, a_2=0.0, luminosity_distance=true_ld, theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

guess_parameters = dict(
    mass_1=true_m1, mass_2=true_m2)#, luminosity_distance=400.)#, theta_jn=0.4, psi=2.659, ra=1.375, dec=-1.2108)

#injection_parameters = dict(mass_1=36., mass_2=29., luminosity_distance=410.)
#
#guess_parameters = dict(mass_1=40., mass_2=25., luminosity_distance=400.)



# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(['H1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=2048, duration=16,
    start_time=- 3)

psd = ifos[0].power_spectral_density_array
psd_frequency = ifos[0].frequency_array

psd_frequency = psd_frequency[jnp.isfinite(psd)]
psd = psd[jnp.isfinite(psd)]

waveform = IMRPhenomC(psd_frequency, injection_parameters)
#waveform = TaylorF2(psd_frequency, injection_parameters)
H1_lat = 46 + 27. / 60 + 18.528 / 3600
H1_long = -(119 + 24. / 60 + 27.5657 / 3600)
H1_xarm_azimuth = 125.9994
H1_yarm_azimuth = 215.9994
H1_xarm_tilt = -6.195e-4
H1_yarm_tilt = 1.25e-5

H1_arm1 = construct_arm(H1_long, H1_lat, H1_xarm_tilt, H1_xarm_azimuth)
H1_arm2 = construct_arm(H1_long, H1_lat, H1_yarm_tilt, H1_yarm_azimuth)

H1 = detector_tensor(H1_arm1, H1_arm2)
strain = waveform#get_detector_response(waveform,injection_parameters,H1)
#strain = get_detector_response(waveform,injection_parameters,H1)

print('SNR of the event: '+str(np.sqrt(inner_product(strain,strain,psd_frequency,psd))))

@jit
def jax_likelihood(params, data, data_f, PSD):
	waveform = IMRPhenomC(data_f, params)
#	waveform = TaylorF2(data_f, params)
#	waveform = get_detector_response(waveform, params, H1)
	match_filter_SNR = inner_product(waveform, data, data_f, PSD)
	optimal_SNR = inner_product(waveform, waveform, data_f, PSD)
	return (-2*match_filter_SNR + optimal_SNR)/2#, match_filter_SNR, optimal_SNR

@jit
def Mq_to_m1m2(M_tot,q):
	m1 = M_tot/(1+q)
	m2 = m1*q
	inv_1pq = 1./(1+q)
	Jac_det =  inv_1pq*(M*(inv_1pq-q/inv_1pq**2))+(q*inv_1pq**3)*M
	return m1, m2, Jac_det

def logprob_wrap(mass_1, mass_2):#, luminosity_distance):#, theta_jn, psi, ra, dec):
#	m1, m2, Jac_det = Mq_to_m1m2(M_tot, q)

	params = dict(mass_1=mass_1, mass_2=mass_2, a_1=0, a_2=0, luminosity_distance=true_ld, theta_jn=0.4, psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)
	return jax_likelihood(params, strain, psd_frequency, psd)

log_prob = lambda x: logprob_wrap(**x)

#def log_prob(param):
#	if (param[0]<=0) or (param[1]<=0):
#		return -jnp.inf
#	params = dict(mass_1=param[0], mass_2=param[1], a_1=0, a_2=0, luminosity_distance=true_ld, theta_jn=0.4, psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)
#	return jax_likelihood(params, strain, psd_frequency, psd)

################################################################
# Test with Emcee to make sure likelihood looks fine
################################################################

#import emcee 
#
#nwalkers = 32
#ndim = 2
#p0 = np.random.rand(nwalkers, ndim) + [true_m1,true_m2]
#sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
#state = sampler.run_mcmc(p0, 100)
#sampler.reset()
#sampler.run_mcmc(state, 10000)

###############################################################
# BlackJax section
###############################################################

import blackjax.hmc as hmc
import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup

rng_key = jax.random.PRNGKey(0)
key, subkey = random.split(rng_key)

initial_state = hmc.new_state(guess_parameters, log_prob)
print('Finding step size and mass matrix')

time1 = time.time()
kernel_generator = lambda step_size, inverse_mass_matrix: hmc.kernel(
    log_prob, step_size, inverse_mass_matrix, 60
)

final_state, (step_size, inverse_mass_matrix), info = stan_warmup.run(
    key,
    kernel_generator,
    initial_state,
    1000,
    is_mass_matrix_diagonal=False
)

print("Finding inverse mass matrix takes: "+str(time.time()-time1)+" seconds")
print("Stepsize: "+str(step_size))
print("Mass_matrix: "+str(inverse_mass_matrix))
num_integration_steps = 60

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
