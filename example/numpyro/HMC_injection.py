import numpy as np
import bilby
import jax.numpy as jnp

from jax.config import config
from jax import grad, jit
config.update("jax_enable_x64", True)

from jaxgw.likelihood.utils import inner_product
from jaxgw.waveform.TaylorF2 import TaylorF2
from jaxgw.waveform.IMRPhenomC import IMRPhenomC
from jax import random, grad, jit, vmap, jacfwd, jacrev, value_and_grad



injection_parameters = dict(
    mass_1=36., mass_2=29., luminosity_distance=40., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(['H1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=2048, duration=4,
    start_time=- 3)

psd = ifos[0].power_spectral_density_array
psd_frequency = ifos[0].frequency_array

psd_frequency = psd_frequency[jnp.isfinite(psd)]
psd = psd[jnp.isfinite(psd)]

waveform = TaylorF2(psd_frequency, injection_parameters)
strain = waveform['plus']#get_detector_response(waveform,injection_parameters,H1).T[0]

@jit
def jax_likelihood(params, data, data_f, PSD):
	waveform = TaylorF2(data_f, params)['plus']
#	waveform = get_detector_response(waveform, params, H1).T[0]
	match_filter_SNR = inner_product(waveform, data, data_f, PSD)
	optimal_SNR = inner_product(waveform, waveform, data_f, PSD)
	return -(2*match_filter_SNR - optimal_SNR)/2

def jax_posterior(params,data,data_f,PSD):
	if params['mass_1'] < 0:
		params['mass_1'] = 0
	if params['mass_2'] < 0:
		params['mass_2'] = 0
	if (params['a_1'] < 0):
		params['a_1'] = 0
	if (params['a_2'] < 0):
		params['a_2'] = 0
	return jax_likelihood(params,data,data_f,PSD)


from numpyro.infer import MCMC, NUTS

nuts_kernel = NUTS(jax_likelihood)

mcmc = MCMC(nuts_kernel, num_warmup=5, num_samples=10)

rng_key = random.PRNGKey(0)

mcmc.run(rng_key, injection_parameters, strain, psd_frequency, psd, extra_fields=('potential_energy',))
