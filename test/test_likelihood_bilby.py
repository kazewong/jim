from bilby.gw.detector import psd
import numpy as np
import bilby
import jax
import jax.numpy as jnp

from jax.config import config

from jaxgw.sampler.NF_proposal import nf_metropolis_kernel, nf_metropolis_sampler
config.update("jax_enable_x64", True)

from jaxgw.gw.likelihood.detector_projection import construct_arm, detector_tensor, antenna_response, get_detector_response

from jaxgw.gw.likelihood.utils import inner_product
from jaxgw.gw.likelihood.detector_preset import get_H1, get_L1
from jaxgw.gw.waveform.TaylorF2 import TaylorF2
from jaxgw.gw.waveform.IMRPhenomC import IMRPhenomC
from jax import random, grad, jit, vmap, jacfwd, jacrev, value_and_grad, pmap

from jaxgw.sampler.Gaussian_random_walk import rw_metropolis_sampler
from jaxgw.sampler.maf import MaskedAutoregressiveFlow
from jaxgw.sampler.realNVP import RealNVP
from jax.scipy.stats import multivariate_normal
from flax.training import train_state  # Useful dataclass to keep train state
import optax                           # Optimizers


"""
This tutorial includes advanced specifications
for analysing binary neutron star event data.
Here GW170817 is used as an example.
"""
import bilby
from gwpy.timeseries import TimeSeries
from bilby.gw.utils import greenwich_mean_sidereal_time
import lalsimulation as lalsim
from lal import MSUN_SI, PC_SI, MTSUN_SI

logger = bilby.core.utils.logger

outdir = 'outdir'
label = 'bns_example'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)

# We are going to inject a binary neutron star waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# aligned spins of both black holes (chi_1, chi_2), etc.
injection_parameters = dict(
    mass_1=1.5, mass_2=1.3, chi_1=0.0, chi_2=0.0, luminosity_distance=50.,
    theta_jn=0.4, psi=2.659, phase=1.3, geocent_time=1126259642.413,
    ra=1.375, dec=-1.2108, lambda_1=0, lambda_2=0)


# Set the duration and sampling frequency of the data segment that we're going
# to inject the signal into. For the
# TaylorF2 waveform, we cut the signal close to the isco frequency
duration = 32
sampling_frequency = 2 * 1024
start_time = injection_parameters['geocent_time'] + 2 - duration

jaxgw_params = dict(mass_1=1.5, mass_2=1.3, spin_1=0.0, spin_2=0.0, luminosity_distance=50, phase_c=1.3, t_c=0,\
        theta_jn=0.4, psi=2.659, ra=1.375, dec=-1.2108,\
        f_ref=50., geocent_time = start_time, start_time=start_time,
        greenwich_mean_sidereal_time=greenwich_mean_sidereal_time(start_time))


# Fixed arguments passed into the source model. The analysis starts at 40 Hz.
waveform_arguments = dict(waveform_approximant='TaylorF2',
                          reference_frequency=50., minimum_frequency=40.0)

# Create the waveform_generator using a LAL Binary Neutron Star source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments)

# Set up interferometers.  In this case we'll use three interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1), and Virgo (V1)).
# These default to their design sensitivity and start at 40 Hz.
interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1'])
for interferometer in interferometers:
    interferometer.minimum_frequency = 40
interferometers.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=start_time)
interferometers.inject_signal(parameters=injection_parameters,
                              waveform_generator=waveform_generator)

# Load the default prior for binary neutron stars.
# We're going to sample in chirp_mass, symmetric_mass_ratio, lambda_tilde, and
# delta_lambda rather than mass_1, mass_2, lambda_1, and lambda_2.
# BNS have aligned spins by default, if you want to allow precessing spins
# pass aligned_spin=False to the BNSPriorDict
priors = bilby.gw.prior.BNSPriorDict()
for key in ['psi', 'geocent_time', 'ra', 'dec', 'chi_1', 'chi_2',
            'theta_jn', 'luminosity_distance', 'phase']:
    priors[key] = injection_parameters[key]
priors.pop('mass_ratio')
priors.pop('lambda_1')
priors.pop('lambda_2')
priors['chirp_mass'] = bilby.core.prior.Gaussian(
    1.215, 0.1, name='chirp_mass', unit='$M_{\\odot}$')
priors['symmetric_mass_ratio'] = bilby.core.prior.Uniform(
    0.1, 0.25, name='symmetric_mass_ratio')
priors['lambda_tilde'] = bilby.core.prior.Uniform(0, 5000, name='lambda_tilde')
priors['delta_lambda'] = bilby.core.prior.Uniform(
    -5000, 5000, name='delta_lambda')

# Initialise the likelihood by passing in the interferometer data (IFOs)
# and the waveform generator
bilby_likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=interferometers, waveform_generator=waveform_generator,
    time_marginalization=False, phase_marginalization=False,
    distance_marginalization=False, priors=priors)

psd_frequency = interferometers[0].frequency_array[1:]
H1, H1_vertex = get_H1()
L1, L1_vertex = get_L1()

strain_H1 = get_detector_response(psd_frequency,TaylorF2(psd_frequency,jaxgw_params), jaxgw_params,H1,H1_vertex)#interferometers[0].frequency_domain_strain[1:]
strain_L1 = get_detector_response(psd_frequency,TaylorF2(psd_frequency,jaxgw_params), jaxgw_params,L1,L1_vertex)#interferometers[1].frequency_domain_strain[1:]
psd_H1 = interferometers[0].power_spectral_density_array[1:]
psd_L1 = interferometers[1].power_spectral_density_array[1:]

duration = waveform_generator.duration

print('SNR of the event in H1: '+str(np.sqrt(inner_product(strain_H1,strain_H1,psd_frequency,psd_H1))))
print('SNR of the event in L1: '+str(np.sqrt(inner_product(strain_L1,strain_L1,psd_frequency,psd_L1))))

@jit
def single_detector_likelihood(params, data, data_f, PSD, detector, detector_vertex):
#	waveform = IMRPhenomC(data_f, params)
    waveform = TaylorF2(data_f, params)
    waveform = get_detector_response(data_f, waveform, params, detector, detector_vertex)
#    waveform *= mask
    match_filter_SNR = inner_product(waveform, data, data_f, PSD)
    optimal_SNR = inner_product(waveform, waveform, data_f, PSD)
    return  match_filter_SNR, optimal_SNR

@jit
def single_detector_likelihood_bilby(params, data, data_f, PSD, detector, detector_vertex):
#	waveform = IMRPhenomC(data_f, params)
	waveform = TaylorF2(data_f, params)
	waveform = get_detector_response(data_f, waveform, params, detector, detector_vertex)
	log_l = -2 / duration * jnp.vdot(data-waveform, (data-waveform)/PSD)
	return log_l.real

@jit
def logprob_wrap(mass_1, mass_2, luminosity_distance, phase_c, t_c,  theta_jn, psi, ra, dec):
	params = dict(mass_1=mass_1, mass_2=mass_2, spin_1=0, spin_2=0, luminosity_distance=luminosity_distance, phase_c=phase_c%(2*jnp.pi), t_c=t_c,\
                theta_jn=theta_jn%(jnp.pi), psi=psi%(jnp.pi), ra=ra%(2*jnp.pi), dec=dec%(jnp.pi),\
                f_ref=50., geocent_time = interferometers[0].strain_data.start_time+t_c, start_time=interferometers[0].strain_data.start_time,
				greenwich_mean_sidereal_time=greenwich_mean_sidereal_time(interferometers[0].strain_data.start_time))
	H1_SNR = single_detector_likelihood(params, strain_H1, psd_frequency, psd_H1, H1, H1_vertex)
	L1_SNR = single_detector_likelihood(params, strain_L1, psd_frequency, psd_L1, L1, L1_vertex)
	match_filter_SNR = H1_SNR[0] + L1_SNR[0]
	optimal_SNR = H1_SNR[1] + L1_SNR[1]
	return match_filter_SNR - optimal_SNR/2


likelihood = lambda x: logprob_wrap(*x)
likelihood = jit(likelihood)
d_likelihood = jit(grad(likelihood))
para_logp = jit(jax.vmap(likelihood))

result = bilby.result.read_in_result(filename='/mnt/home/wwong/GWProject/Tutorial/bilby_tutorial/outdir/bns_example_result.json')

for i in result.posterior.keys():
    bilby_likelihood.parameters[i] = result.posterior[i].values[-1]

print("Section where we use bilby waveform generator.")

waveform = bilby_likelihood.waveform_generator.frequency_domain_strain(bilby_likelihood.parameters)
params = {}
params['mass_1'] = bilby_likelihood.parameters['mass_1']
params['mass_2'] = bilby_likelihood.parameters['mass_2']
params['spin_1'] = 0.0#bilby_likelihood.parameters['a_1']
params['spin_2'] = 0.0#bilby_likelihood.parameters['a_2']
params['luminosity_distance'] = bilby_likelihood.parameters['luminosity_distance']
params['phase_c'] = bilby_likelihood.parameters['phase']
params['t_c'] = 0#bilby_likelihood.parameters['geocent_time']
params['theta_jn'] = bilby_likelihood.parameters['theta_jn']
params['psi'] = bilby_likelihood.parameters['psi']
params['ra'] = bilby_likelihood.parameters['ra']
params['dec'] = bilby_likelihood.parameters['dec']
params['f_ref'] = bilby_likelihood.parameters['reference_frequency']
params['start_time'] = interferometers[0].strain_data.start_time
params['geocent_time'] = bilby_likelihood.parameters['geocent_time']
params['greenwich_mean_sidereal_time'] = greenwich_mean_sidereal_time(bilby_likelihood.parameters['geocent_time'])

mask = np.ones(psd_frequency.shape)
mask[psd_frequency<bilby_likelihood.parameters['minimum_frequency']] = 0

jaxgw_projection = get_detector_response(psd_frequency, waveform ,params,H1,H1_vertex)
bilby_projection = interferometers[0].get_detector_response(waveform,bilby_likelihood.parameters)

print("Difference in projections: "+str(np.linalg.norm(jaxgw_projection-bilby_projection)))

bilby_H1_snr = bilby_likelihood.calculate_snrs(waveform,interferometers[0])
bilby_L1_snr = bilby_likelihood.calculate_snrs(waveform,interferometers[1])

jax_H1_match_filter_snr = inner_product(get_detector_response(psd_frequency, waveform ,params,H1,H1_vertex), strain_H1, psd_frequency, psd_H1)
jax_L1_match_filter_snr = inner_product(get_detector_response(psd_frequency, waveform ,params,L1,L1_vertex), strain_L1, psd_frequency, psd_L1)
jax_H1_optimal_snr = inner_product(get_detector_response(psd_frequency, waveform ,params,H1,H1_vertex), get_detector_response(psd_frequency, waveform ,params,H1,H1_vertex), psd_frequency, psd_H1)
jax_L1_optimal_snr = inner_product(get_detector_response(psd_frequency, waveform ,params,L1,L1_vertex), get_detector_response(psd_frequency, waveform ,params,L1,L1_vertex), psd_frequency, psd_L1)

print("Bilby log likelihood: "+str(bilby_likelihood.log_likelihood_ratio()))
print("Jax log likelihood: "+str((jax_H1_match_filter_snr+jax_L1_match_filter_snr)-(jax_H1_optimal_snr+jax_L1_optimal_snr)/2))
