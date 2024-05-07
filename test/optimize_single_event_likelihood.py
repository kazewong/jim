import bilby
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.config import config

config.update("jax_enable_x64", True)

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 4.
sampling_frequency = 2048.
minimum_frequency = 20

# Specify the output directory and the name of the simulation.
outdir = 'outdir'
label = 'fast_tutorial'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
injection_parameters = dict(
    mass_1=36., mass_2=29., a_1=0., a_2=0., tilt_1=0., tilt_2=0.,
    phi_12=0., phi_jl=0., luminosity_distance=40., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

# Fixed arguments passed into the source model
waveform_arguments = dict(waveform_approximant='IMRPhenomC',
                          reference_frequency=50.,
                          minimum_frequency=minimum_frequency)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(['H1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)
ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters)

##############################################
# Jax section
##############################################


waveform = waveform_generator.frequency_domain_strain()
waveform_frequency = waveform_generator.frequency_array

psd = ifos[0].power_spectral_density_array
psd_frequency = ifos[0].frequency_array

waveform_frequency = waveform_frequency[jnp.isfinite(psd)]
psd_frequency = psd_frequency[jnp.isfinite(psd)]
psd = psd[jnp.isfinite(psd)]

from jax import grad, jacfwd, jacrev, jit, random, value_and_grad, vmap
from jax.experimental.optimizers import adam, sgd
from jaxgw.likelihood.detector_projection import (antenna_response,
                                                  construct_arm,
                                                  detector_tensor,
                                                  get_detector_response)
from jaxgw.likelihood.utils import inner_product
from jaxgw.waveform.IMRPhenomB import IMRPhenomB, Lorentzian, getPhenomCoef
from jaxgw.waveform.IMRPhenomC import IMRPhenomC
from jaxgw.waveform.TaylorF2 import TaylorF2

waveform = TaylorF2(waveform_frequency, injection_parameters)
H1_lat = 46 + 27. / 60 + 18.528 / 3600
H1_long = -(119 + 24. / 60 + 27.5657 / 3600)
H1_xarm_azimuth = 125.9994
H1_yarm_azimuth = 215.9994
H1_xarm_tilt = -6.195e-4
H1_yarm_tilt = 1.25e-5

H1_arm1 = construct_arm(H1_long, H1_lat, H1_xarm_tilt, H1_xarm_azimuth)
H1_arm2 = construct_arm(H1_long, H1_lat, H1_yarm_tilt, H1_yarm_azimuth)

H1 = detector_tensor(H1_arm1, H1_arm2)

psd_interp = jnp.interp(waveform_frequency, psd_frequency, psd)
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

guess_parameters = dict(
    mass_1=36., mass_2=29.9, a_1=0., a_2=0., luminosity_distance=40., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

print("True likelihood"+str(jax_likelihood(injection_parameters,strain, waveform_frequency,psd_interp)))
print("Guess likelihood"+str(jax_likelihood(guess_parameters,strain, waveform_frequency,psd_interp)))


learning_rate = 1e-4
opt_init, opt_update, get_params = adam(learning_rate)
opt_state = opt_init(guess_parameters)

def step(step, opt_state):
    params = get_params(opt_state)
    value, grads = value_and_grad(jax_likelihood,argnums=(0))(params, strain, waveform_frequency, psd_interp)
    opt_state = opt_update(step, grads, opt_state)
    return value, opt_state

for i in range(10000):
    value, opt_state = step(i, opt_state)
    if jnp.isnan(value):
        break
    if i%10 == 0:
    	print(value,get_params(opt_state))

