import numpy as np
import bilby
import jax
import jax.numpy as jnp
import time

from jax.config import config
config.update("jax_enable_x64", True)

from jaxgw.gw.likelihood.detector_projection import construct_arm, detector_tensor, antenna_response, get_detector_response

from jaxgw.gw.likelihood.utils import inner_product
from jaxgw.gw.likelihood.detector_preset import get_H1
from jaxgw.gw.waveform.IMRPhenomC import IMRPhenomC, IMRPhenomC_dict2list
from jax import random, grad, jit, vmap, jacfwd, jacrev, value_and_grad, pmap


true_m1 = 3.
true_m2 = 2.
true_ld = 150.
true_phase = 0.
true_gt = 0.

injection_parameters = dict(mass_1=true_m1, mass_2=true_m2, spin_1=0.0, spin_2=0.0, luminosity_distance=true_ld, theta_jn=0.4, psi=2.659, phase_c=true_phase, t_c=true_gt, ra=1.375, dec=-1.2108)

guess_parameters = IMRPhenomC_dict2list(dict(mass_1=true_m1, mass_2=true_m2, spin_1=0.1, spin_2=0.0, luminosity_distance=true_ld, theta_jn=0.4, psi=2.659, phase_c=true_phase, t_c=true_gt, ra=1.375, dec=-1.2108))

#injection_parameters = IMRPhenomC_dict2list(injection_parameters)


# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(['H1'])
ifos.set_strain_data_from_power_spectral_densities(
	sampling_frequency=2048, duration=32,
	start_time=- 3)

psd = ifos[0].power_spectral_density_array
psd_frequency = ifos[0].frequency_array

psd_frequency = psd_frequency[jnp.isfinite(psd)]
psd = psd[jnp.isfinite(psd)]

waveform = IMRPhenomC(psd_frequency, injection_parameters)
H1 = get_H1()
strain = get_detector_response(waveform,injection_parameters,H1)

print('SNR of the event: '+str(np.sqrt(inner_product(strain,strain,psd_frequency,psd))))

@jit
def jax_likelihood(params, data, data_f, PSD):
	waveform = IMRPhenomC(data_f, params)
	waveform = get_detector_response(waveform, params, H1)
	match_filter_SNR = inner_product(waveform, data, data_f, PSD)
	optimal_SNR = inner_product(waveform, waveform, data_f, PSD)
	return (-2*match_filter_SNR + optimal_SNR)/2

@jit
def logprob_wrap(params):
	parameters = dict(mass_1=params[0], mass_2=params[1], spin_1=params[2], spin_2=params[3], luminosity_distance=params[4], phase_c=params[5], t_c=params[6], theta_jn=params[7], psi=params[8], ra=params[9], dec=params[10])
	return jax_likelihood(parameters, strain, psd_frequency, psd)

logL = logprob_wrap(guess_parameters)
logL_jacobian = jacfwd(logprob_wrap)(guess_parameters)
logL_hessian = jacfwd(jacrev(logprob_wrap))(guess_parameters)


