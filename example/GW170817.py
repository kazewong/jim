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

logger = bilby.core.utils.logger


outdir = 'outdir'
data_dir = '/mnt/home/wwong/ceph/GWProject/GWTC/individual_events/'
label = 'GW170817'
time_of_event = bilby.gw.utils.get_event_time(label)
bilby.core.utils.setup_logger(outdir=outdir, label=label)
# GET DATA FROM INTERFEROMETER
# include 'V1' for appropriate O2 events
interferometer_names = ['H1', 'L1', 'V1']
duration = 32
roll_off = 0.2  # how smooth is the transition from no signal
# to max signal in a Tukey Window.
psd_offset = -512  # PSD is estimated using data from
# `center_time+psd_offset` to `center_time+psd_offset + psd_duration`
# This determines the time window used to fetch open data.
psd_duration = 1024
coherence_test = False  # coherence between detectors
filter_freq = None  # low pass filter frequency to cut signal content above
# Nyquist frequency. The condition is 2 * filter_freq >= sampling_frequency
end_time = time_of_event + duration/2
start_time = end_time - duration

psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time


ifo_list = bilby.gw.detector.InterferometerList([])
for det in ["H1", "L1", "V1"]:
	try:
		logger.info("Loading signal data for detector %s", det)
		data = TimeSeries.read(data_dir+label+'/'+det+'_signal.hdf5')
	except:
		logger.info("Downloading signal data for ifo {}".format(det))
		data = TimeSeries.fetch_open_data(det, start_time, end_time)
		data.write(data_dir+label+'/'+det+'_signal.hdf5')

	ifo = bilby.gw.detector.get_empty_interferometer(det)
	ifo.strain_data.set_from_gwpy_timeseries(data)


	try:
		logger.info("Loading psd data for detector %s", det)
		psd_data = TimeSeries.read(data_dir+label+'/'+det+'_psd.hdf5')
	except:
		logger.info("Downloading psd data for ifo {}".format(det))
		psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)
		psd_data.write(data_dir+label+'/'+det+'_psd.hdf5')
	psd_alpha = 2 * roll_off / duration
	psd = psd_data.psd(
		fftlength=duration,
		overlap=0,
		window=("tukey", psd_alpha),
		method="median"
	)
	ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
		frequency_array=psd.frequencies.value, psd_array=psd.value)
	ifo_list.append(ifo)

logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)

# CHOOSE PRIOR FILE
prior = bilby.gw.prior.BNSPriorDict(filename='GW170817.prior')
deltaT = 0.1
prior['geocent_time'] = bilby.core.prior.Uniform(
    minimum=time_of_event - deltaT / 2,
    maximum=time_of_event + deltaT / 2,
    name='geocent_time',
    latex_label='$t_c$',
    unit='$s$')
# GENERATE WAVEFORM
# OVERVIEW OF APPROXIMANTS:
# https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/Waveforms/Overview
duration = None  # duration and sampling frequency will be overwritten
# to match the ones in interferometers.
sampling_frequency = 4096
start_time = 0  # set the starting time of the time array
waveform_arguments = {
    'waveform_approximant': 'IMRPhenomPv2_NRTidal', 'reference_frequency': 20}

source_model = bilby.gw.source.lal_binary_neutron_star
convert_bns = bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    start_time=start_time,
    frequency_domain_source_model=source_model,
    parameter_conversion=convert_bns,
    waveform_arguments=waveform_arguments,)

# CHOOSE LIKELIHOOD FUNCTION
# Time marginalisation uses FFT.
# Distance marginalisation uses a look up table calculated at run time.
# Phase marginalisation is done analytically using a Bessel function.
bilby_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list,
    waveform_generator,
    time_marginalization=False,
    distance_marginalization=False,
    phase_marginalization=False,)

strain_H1 = ifo_list[0].frequency_domain_strain[1:]
strain_L1 = ifo_list[1].frequency_domain_strain[1:]
psd_frequency = ifo_list[0].frequency_array[1:]
psd_H1 = ifo_list[0].power_spectral_density_array[1:]
psd_L1 = ifo_list[1].power_spectral_density_array[1:]

H1, H1_vertex = get_H1()
L1, L1_vertex = get_L1()

duration = waveform_generator.duration

print('SNR of the event in H1: '+str(np.sqrt(inner_product(strain_H1,strain_H1,psd_frequency,psd_H1))))
print('SNR of the event in L1: '+str(np.sqrt(inner_product(strain_L1,strain_L1,psd_frequency,psd_L1))))

@jit
def single_detector_likelihood(params, data, data_f, PSD, detector, detector_vertex):
#	waveform = IMRPhenomC(data_f, params)
	waveform = TaylorF2(data_f, params)
	waveform = get_detector_response(data_f, waveform, params, detector, detector_vertex)
	match_filter_SNR = inner_product(waveform, data, data_f, PSD)
	optimal_SNR = inner_product(waveform, waveform, data_f, PSD)
	return (-2*match_filter_SNR + optimal_SNR)/2#, match_filter_SNR, optimal_SNR

@jit
def single_detector_likelihood_bilby(params, data, data_f, PSD, detector, detector_vertex):
#	waveform = IMRPhenomC(data_f, params)
	waveform = TaylorF2(data_f, params)
	waveform = get_detector_response(data_f, waveform, params, detector, detector_vertex)
	log_l = -2 / duration * jnp.vdot(data-waveform, (data-waveform)/PSD)
	return log_l.real

@jit
def logprob_wrap(mass_1, mass_2, luminosity_distance, phase_c, t_c,  theta_jn, psi, ra, dec):
	params = dict(mass_1=mass_1, mass_2=mass_2, spin_1=0, spin_2=0, luminosity_distance=10**luminosity_distance, phase_c=phase_c, t_c=10**t_c, theta_jn=theta_jn, psi=psi, ra=ra, dec=dec, f_ref=50)
#	params = dict(mass_1=mass_1, mass_2=mass_2, spin_1=0, spin_2=0, luminosity_distance=true_ld, theta_jn=0.4, psi=2.659, phase_c=true_phase, t_c=true_gt, ra=1.375, dec=-1.2108)
	return single_detector_likelihood_bilby(params, strain_H1, psd_frequency, psd_H1, H1, H1_vertex)+single_detector_likelihood_bilby(params, strain_L1, psd_frequency, psd_L1, L1, L1_vertex)



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
n_chains = 100
learning_rate = 0.01
momentum = 0.9
num_epochs = 300
batch_size = 10000
look_back_epoch = 10
train_epoch = 25
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
prior_range.append([1.6093862655801942,1.6093862655801943 ])
prior_range.append([1.1616754457131563,1.1616754457131564])
prior_range.append([np.log10(0.1),np.log10(3000.0)])
prior_range.append([0.,2*jnp.pi])
prior_range.append([np.log10(greenwich_mean_sidereal_time(time_of_event)),np.log10(greenwich_mean_sidereal_time(time_of_event)+1)])
prior_range.append([0.,jnp.pi])
prior_range.append([0.,jnp.pi])
prior_range.append([0.,2*jnp.pi])
prior_range.append([0.,jnp.pi])
prior_range = jnp.array(prior_range)

initial_guess = jax.random.uniform(rng_key_ic,(n_chains,n_dim)) #(n_dim, n_chains)
initial_guess = (initial_guess*(prior_range[:,1]-prior_range[:,0])+prior_range[:,0])

from scipy.optimize import minimize

loss = lambda x: -likelihood(x)

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
	rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples, likelihood, last_step, 0.01)
	last_step = positions[:,-1].T
	if i%train_epoch == train_epoch-1:
		train_sample = np.concatenate(chains[-look_back_epoch:],axis=1).reshape(-1,n_dim)
		rng_keys_nf, state = train_flow(rng_key_nf, model, state, train_sample)
		trained = True
	if i%nf_sample_epoch == 0 and trained == True:
		rng_keys_nf, nf_chain, log_prob, log_prob_nf = nf_metropolis_sampler(rng_keys_nf, sample, log_prob_nf_function, state.params , para_logp, positions[:,-1])
		positions = jnp.concatenate((positions,nf_chain),axis=1)
	chains.append(positions)

chains = np.concatenate(chains,axis=1)
nf_samples = sample_nf(model, state.params, rng_keys_nf, 10000)
