import numpy as np
import bilby
import jax.numpy as jnp

from jax.config import config
from jax import grad, jit
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
    phi_12=0., phi_jl=0., luminosity_distance=410., theta_jn=0.4, psi=2.659,
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

from jaxgw.likelihood.detector_projection import construct_arm, detector_tensor, antenna_response, get_detector_response
from jaxgw.likelihood.utils import inner_product
from jaxgw.waveform.TaylorF2 import TaylorF2
from jaxgw.waveform.IMRPhenomB import IMRPhenomB, getPhenomCoef, Lorentzian
from jaxgw.waveform.IMRPhenomC import IMRPhenomC


waveform = IMRPhenomC(waveform_frequency, injection_parameters)
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
strain = get_detector_response(waveform,injection_parameters,H1).T[0]
jaxgw_snr = inner_product(strain, strain, waveform_frequency, psd_interp)
d_jaxgw_snr = grad(inner_product)(strain, strain, waveform_frequency, psd_interp)

@jit
def jax_likelihood(params, data, data_f, PSD):
	waveform = IMRPhenomC(data_f, params)
	waveform = get_detector_response(waveform, params, H1).T[0]
	output = inner_product(waveform, data, data_f, PSD)
	return output


dlikelihood = grad(jax_likelihood)(injection_parameters, strain, waveform_frequency, psd_interp)


