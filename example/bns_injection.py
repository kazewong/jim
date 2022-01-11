import numpy as np
import bilby
import jax
import jax.numpy as jnp

from jax.config import config

from jaxgw.sampler.NF_proposal import nf_metropolis_kernel, nf_metropolis_sampler
config.update("jax_enable_x64", True)

from jaxgw.gw.likelihood.detector_projection import construct_arm, detector_tensor, antenna_response, get_detector_response

from jaxgw.gw.likelihood.utils import Mc_q_to_m1m2, inner_product
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
    mass_1=1.5, mass_2=1.3, chi_1=0.0, chi_2=0.0, luminosity_distance=200.,
    theta_jn=0.4, psi=2.659, phase=1.3, geocent_time=1126259642.413,
    ra=1.375, dec=-1.2108, lambda_1=0, lambda_2=0, spin_1 = 0.0, spin_2 = 0.0,
    f_ref=50., t_c = 0, phase_c = 1.3,
    greenwich_mean_sidereal_time=greenwich_mean_sidereal_time(1126259642.413),
    start_time=1126259642.413)

# Set the duration and sampling frequency of the data segment that we're going
# to inject the signal into. For the
# TaylorF2 waveform, we cut the signal close to the isco frequency
duration = 4
sampling_frequency = 2 * 1024
start_time = injection_parameters['geocent_time'] + 2 - duration

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
# interferometers.inject_signal(parameters=injection_parameters,
#                               waveform_generator=waveform_generator)

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

psd_frequency = interferometers[0].frequency_array
true_signal = TaylorF2(psd_frequency,injection_parameters)
true_signal['plus'] = np.array(true_signal['plus'])
true_signal['cross'] = np.array(true_signal['cross'])
true_signal['plus'][0] = 0
true_signal['cross'][0] = 0
for interferometer in interferometers:
    interferometer.inject_signal_from_waveform_polarizations(injection_parameters,true_signal)

strain_H1 = interferometers[0].frequency_domain_strain[1:]
strain_L1 = interferometers[1].frequency_domain_strain[1:]

psd_H1 = interferometers[0].power_spectral_density_array[1:]
psd_L1 = interferometers[1].power_spectral_density_array[1:]

H1, H1_vertex = get_H1()
L1, L1_vertex = get_L1()

duration = waveform_generator.duration

print('SNR of the event in H1: '+str(np.sqrt(inner_product(strain_H1,strain_H1,psd_frequency,psd_H1))))
print('SNR of the event in L1: '+str(np.sqrt(inner_product(strain_L1,strain_L1,psd_frequency,psd_L1))))

mask = np.ones(psd_frequency[1:].shape)
mask[psd_frequency[1:]<40] = 0

@jit
def single_detector_likelihood(params, data, data_f, PSD, detector, detector_vertex):
#	waveform = IMRPhenomC(data_f, params)
    waveform = TaylorF2(data_f, params)
    waveform = get_detector_response(data_f, waveform, params, detector, detector_vertex)
    waveform *= mask
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


def logprob_wrap(mass_1, mass_2, luminosity_distance, phase_c, t_c,  theta_jn, psi, ra, dec):
    params = dict(mass_1=mass_1, mass_2=mass_2, spin_1=0, spin_2=0, luminosity_distance=luminosity_distance, phase_c=phase_c%(2*jnp.pi), t_c=t_c,\
                theta_jn=theta_jn%(jnp.pi), psi=psi%(jnp.pi), ra=ra%(2*jnp.pi), dec=dec%(jnp.pi),\
                f_ref=50., geocent_time = interferometers[0].strain_data.start_time+t_c, start_time=interferometers[0].strain_data.start_time,
				greenwich_mean_sidereal_time=greenwich_mean_sidereal_time(interferometers[0].strain_data.start_time))
    H1_SNR = single_detector_likelihood(params, strain_H1, psd_frequency[1:], psd_H1, H1, H1_vertex)
    L1_SNR = single_detector_likelihood(params, strain_L1, psd_frequency[1:], psd_L1, L1, L1_vertex)
    match_filter_SNR = H1_SNR[0] + L1_SNR[0]
    optimal_SNR = H1_SNR[1] + L1_SNR[1]
    return match_filter_SNR - optimal_SNR/2

# def logprob_wrap(Mc, q, luminosity_distance, phase_c, t_c,  theta_jn, psi, ra, dec):
#     mass_1, mass_2 = Mc_q_to_m1m2(Mc, q)
#     params = dict(mass_1=mass_1, mass_2=mass_2, spin_1=0, spin_2=0, luminosity_distance=10**luminosity_distance, phase_c=phase_c%(2*jnp.pi), t_c=0,\
#                 theta_jn=theta_jn%(jnp.pi), psi=psi%(jnp.pi), ra=ra%(2*jnp.pi), dec=dec%(jnp.pi),\
#                 f_ref=50., geocent_time = interferometers[0].strain_data.start_time, start_time=interferometers[0].strain_data.start_time,
# 				greenwich_mean_sidereal_time=greenwich_mean_sidereal_time(interferometers[0].strain_data.start_time))
#     H1_SNR = single_detector_likelihood(params, strain_H1, psd_frequency[1:], psd_H1, H1, H1_vertex)
#     L1_SNR = single_detector_likelihood(params, strain_L1, psd_frequency[1:], psd_L1, L1, L1_vertex)
#     match_filter_SNR = H1_SNR[0] + L1_SNR[0]
#     optimal_SNR = H1_SNR[1] + L1_SNR[1]
#     return match_filter_SNR - optimal_SNR/2

likelihood = lambda x: logprob_wrap(*x)
likelihood = jit(likelihood)
d_likelihood = jit(grad(likelihood))
para_logp = jit(jax.vmap(likelihood))

#### Sampling ####

def train_step(model, state, batch):
	def loss(params):
		y, log_det = model.apply({'params': params},batch)
		mean = jnp.zeros((batch.shape[0],model.n_features))
		cov = jnp.repeat(jnp.eye(model.n_features)[None,:],batch.shape[0],axis=0)
		log_det = log_det + multivariate_normal.logpdf(y,mean,cov)
		return -jnp.mean(log_det)
	grad_fn = jax.value_and_grad(loss)
	value, grad = grad_fn(state.params)
	state = state.apply_gradients(grads=grad)
	return value,state

train_step = jax.jit(train_step,static_argnums=(0,))

def train_flow(rng, model, state, data):

    def train_epoch(state, train_ds, batch_size, epoch, rng):
        """Train for a single epoch."""
        train_ds_size = len(train_ds)
        steps_per_epoch = train_ds_size // batch_size

        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        for perm in perms:
            batch = train_ds[perm, ...]
            value, state = train_step(model, state, batch)

        return value, state

    for epoch in range(1, num_epochs + 1):
        print('Epoch %d' % epoch)
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        value, state = train_epoch(state, data, batch_size, epoch, input_rng)
        print('Train loss: %.3f' % value)

    return rng, state

def sample_nf(model, param, rng_key,n_sample):
    rng_key, subkey = random.split(rng_key)
    samples = model.apply({'params': param}, subkey, n_sample,param, method=model.sample)
    return rng_key,samples

n_dim = 9
n_samples = 100
nf_samples = 10
n_chains = 20
learning_rate = 0.01
momentum = 0.9
num_epochs = 300
batch_size = 10000
look_back_epoch = 10
start_train_epoch = 100
train_epoch = 100
nf_sample_epoch = 25
total_epoch = 100
precompiled = False

print("Preparing RNG keys")
rng_key = jax.random.PRNGKey(42)
rng_key_ic, rng_key_mcmc, rng_key_nf = jax.random.split(rng_key,3)

rng_keys_mcmc = jax.random.split(rng_key_mcmc, n_chains)  # (nchains,)
rng_keys_nf, init_rng_keys_nf = jax.random.split(rng_key_nf,2)

print("Finding initial position for chains")

prior_range = []
prior_range.append([1.0,2.0])
prior_range.append([1.0,2.0])
prior_range.append([100.0,300.0])
# prior_range.append([0.5,3.0])
# prior_range.append([0.1,1.0])
# prior_range.append([np.log10(100.0),np.log10(300.0)])

prior_range.append([0.,2*jnp.pi])
prior_range.append([0,0.1])
prior_range.append([0.,jnp.pi])
prior_range.append([0.,jnp.pi])
prior_range.append([0.,2*jnp.pi])
prior_range.append([0.,jnp.pi])
prior_range = jnp.array(prior_range)

initial_guess = jax.random.uniform(rng_key_ic,(n_chains,n_dim)) #(n_dim, n_chains)
initial_guess = (initial_guess*(prior_range[:,1]-prior_range[:,0])+prior_range[:,0])

from scipy.optimize import minimize

loss = lambda x: -likelihood(x)
loss = jit(loss)

initial_position = []
for i in range(n_chains):
	res = minimize(loss,initial_guess[i,:],method='Nelder-Mead')
	initial_position.append(res.x)

initial_position = jnp.array(initial_position).T


#initial_position = (jax.random.normal(rng_key_ic,(n_chains,n_dim))*0.5 + jnp.array(list(guess_parameters.values()))).T #(n_dim, n_chains)

print("Initializing MCMC model and normalizing flow model.")

#model = MaskedAutoregressiveFlow(n_dim,64,4)
model = RealNVP(10,n_dim,64, 1)
params = model.init(init_rng_keys_nf, jnp.ones((1,n_dim)))['params']

run_mcmc = jax.vmap(rw_metropolis_sampler, in_axes=(0, None, None, 1, None),
                    out_axes=0)

tx = optax.adam(learning_rate, momentum)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def sample(rng_key, params):
	return model.apply({'params': params}, rng_key, nf_samples*n_chains, params, method=model.sample)[0]

def log_prob_nf_function(params, location):
	return model.apply({'params': params}, location, method=model.log_prob)

sample = jax.jit(sample)
log_prob_nf_function = jax.jit(log_prob_nf_function)

print("Starting sampling")

trained = False
last_step = initial_position
chains = []
for i in range(total_epoch):
	rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples, likelihood, last_step, 0.1)
	positions = positions.at[:,:,3].set(positions[:,:,3]%(2*jnp.pi))
	positions = positions.at[:,:,5].set(positions[:,:,5]%(jnp.pi))
	positions = positions.at[:,:,6].set(positions[:,:,6]%(jnp.pi))
	positions = positions.at[:,:,7].set(positions[:,:,7]%(2*jnp.pi))
	positions = positions.at[:,:,8].set(positions[:,:,8]%(jnp.pi))
	last_step = positions[:,-1].T
	# if (i > start_train_epoch) and (i%train_epoch == train_epoch-1):
	# 	train_sample = np.concatenate(chains[-look_back_epoch:],axis=1).reshape(-1,n_dim)
	# 	rng_keys_nf, state = train_flow(rng_key_nf, model, state, train_sample)
	# 	trained = True
	# if i%nf_sample_epoch == 0 and trained == True:
	# 	rng_keys_nf, nf_chain, log_prob, log_prob_nf = nf_metropolis_sampler(rng_keys_nf, sample, log_prob_nf_function, state.params , para_logp, positions[:,-1])
	# 	positions = jnp.concatenate((positions,nf_chain),axis=1)
	chains.append(positions)

chains = np.concatenate(chains,axis=1)
nf_samples = sample_nf(model, state.params, rng_keys_nf, 10000)
