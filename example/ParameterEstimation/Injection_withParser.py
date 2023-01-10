# Import packages
import lalsimulation as lalsim 
import numpy as np
import jax.numpy as jnp
import jax
from lal import GreenwichMeanSiderealTime


# from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from ripple import ms_to_Mc_eta
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from jaxgw.PE.detector_preset import * 
from jaxgw.PE.heterodyneLikelihood import make_heterodyne_likelihood_mutliple_detector
from jaxgw.PE.detector_projection import make_detector_response
from jaxgw.PE.generate_noise import generate_noise

from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.MALA import MALA, mala_sampler_autotune
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *

import argparse
import yaml

from tqdm import tqdm
from functools import partialmethod

import sys
sys.path.append('/mnt/home/wwong/GWProject/JaxGW')

parser = argparse.ArgumentParser(description='Injection test')

parser.add_argument('--config', type=str, default='config.yaml', help='config file')

# Add noise parameters to parser
parser.add_argument('--seed', type=int, default=None, help='seed for random number generator')
parser.add_argument('--f_sampling', type=int, default=None, help='sampling frequency')
parser.add_argument('--duration', type=int, default=None, help='duration of the data')
parser.add_argument('--fmin', type=float, default=None, help='minimum frequency')
parser.add_argument('--ifos', nargs='+', default=None, help='list of detectors')

# Add injection parameters to parser
parser.add_argument('--m1', type=float, default=None, help='mass of the first component')
parser.add_argument('--m2', type=float, default=None, help='mass of the second component')
parser.add_argument('--chi1', type=float, default=None, help='dimensionless spin of the first component')
parser.add_argument('--chi2', type=float, default=None, help='dimensionless spin of the second component')
parser.add_argument('--dist_mpc', type=float, default=None, help='distance in megaparsecs')
parser.add_argument('--tc', type=float, default=None, help='coalescence time')
parser.add_argument('--phic', type=float, default=None, help='phase of coalescence')
parser.add_argument('--inclination', type=float, default=None, help='inclination angle')
parser.add_argument('--polarization_angle', type=float, default=None, help='polarization angle')
parser.add_argument('--ra', type=float, default=None, help='right ascension')
parser.add_argument('--dec', type=float, default=None, help='declination')
parser.add_argument('--heterodyne_bins', type=int, default=101, help='number of bins for heterodyne likelihood')

# Add sampler parameters to parser

parser.add_argument('--n_dim', type=int, default=None, help='number of parameters')
parser.add_argument('--n_chains', type=int, default=None, help='number of chains')
parser.add_argument('--n_loop_training', type=int, default=None, help='number of training loops')
parser.add_argument('--n_loop_production', type=int, default=None, help='number of production loops')
parser.add_argument('--n_local_steps', type=int, default=None, help='number of local steps')
parser.add_argument('--n_global_steps', type=int, default=None, help='number of global steps')
parser.add_argument('--learning_rate', type=float, default=None, help='learning rate')
parser.add_argument('--max_samples', type=int, default=None, help='maximum number of samples')
parser.add_argument('--momentum', type=float, default=None, help='momentum during training')
parser.add_argument('--num_epochs', type=int, default=None, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=None, help='batch size')
parser.add_argument('--stepsize', type=float, default=None, help='stepsize for Local sampler')

# Add output parameters to parser

parser.add_argument('--output_path', type=str, default=None, help='output file path')
parser.add_argument('--downsample_factor', type=int, default=1, help='downsample factor')

# parser

args = parser.parse_args()
opt = vars(args)
args = yaml.load(open(opt['config'], 'r'), Loader=yaml.FullLoader)
opt.update(args)
args = opt

# Fetch noise parameters 

print("Constructing detectors")
print("Making noises")

seed = args['seed']
f_sampling = args['f_sampling']
duration = args['duration']
fmin = args['fmin']
ifos = args['ifos']


freqs, psd_dict, noise_dict = generate_noise(seed+1234, f_sampling, duration, fmin, ifos)


# Fetch injection parameters and inject signal

print("Injection signals")

m1 = args['m1']
m2 = args['m2']
chi1 = args['chi1']
chi2 = args['chi2']
dist_mpc = args['dist_mpc']
tc = args['tc']
phic = args['phic']
inclination = args['inclination']
polarization_angle = args['polarization_angle']
ra = args['ra']
dec = args['dec']

Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))

heterodyne_bins = args['heterodyne_bins']

H1 = get_H1()
H1_response = make_detector_response(H1[0], H1[1])
L1 = get_L1()
L1_response = make_detector_response(L1[0], L1[1])
V1 = get_V1()
V1_response = make_detector_response(V1[0], V1[1])

f_ref = 30.0
trigger_time = 1126259462.4
post_trigger_duration = 2
epoch = duration - post_trigger_duration
gmst = GreenwichMeanSiderealTime(trigger_time)


def gen_waveform_H1(f, theta):
    theta_waveform = theta[:8]
    theta_waveform = theta_waveform.at[5].set(0)
    ra = theta[9]
    dec = theta[10]
    hp, hc = gen_IMRPhenomD_polar(f, theta_waveform, f_ref)
    return H1_response(f, hp, hc, ra, dec, gmst , theta[8]) * jnp.exp(-1j*2*jnp.pi*f*(epoch+theta[5]))

def gen_waveform_L1(f, theta):
    theta_waveform = theta[:8]
    theta_waveform = theta_waveform.at[5].set(0)
    ra = theta[9]
    dec = theta[10]
    hp, hc = gen_IMRPhenomD_polar(f, theta_waveform, f_ref)
    return L1_response(f, hp, hc, ra, dec, gmst, theta[8]) * jnp.exp(-1j*2*jnp.pi*f*(epoch+theta[5]))

def gen_waveform_V1(f, theta):
    theta_waveform = theta[:8]
    theta_waveform = theta_waveform.at[5].set(0)
    ra = theta[9]
    dec = theta[10]
    hp, hc = gen_IMRPhenomD_polar(f, theta_waveform, f_ref)
    return V1_response(f, hp, hc, ra, dec, gmst, theta[8]) * jnp.exp(-1j*2*jnp.pi*f*(epoch+theta[5]))

true_param = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle, ra, dec])

from scipy.interpolate import interp1d
q_axis = np.linspace(0.1, 1.0, 10000)
eta_axis = q_axis/(1+q_axis)**2
true_q = interp1d(eta_axis, q_axis)(eta)
cos_inclination = np.cos(inclination)
sin_dec = np.sin(dec)
true_param_trans = jnp.array([Mc, true_q, chi1, chi2, dist_mpc, tc, phic, cos_inclination, polarization_angle, ra, sin_dec])

f_list = freqs[freqs>fmin]
H1_signal = gen_waveform_H1(f_list, true_param)
H1_noise_psd = noise_dict['H1'][freqs>fmin]
H1_psd = psd_dict['H1'][freqs>fmin]
H1_data = H1_noise_psd + H1_signal

L1_signal = gen_waveform_L1(f_list, true_param)
L1_noise_psd = noise_dict['L1'][freqs>fmin]
L1_psd = psd_dict['L1'][freqs>fmin]
L1_data = L1_noise_psd + L1_signal

V1_signal = gen_waveform_V1(f_list, true_param)
V1_noise_psd = noise_dict['V1'][freqs>fmin]
V1_psd = psd_dict['V1'][freqs>fmin]
V1_data = V1_noise_psd + V1_signal

ref_param = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle, ra, dec])

data_list = [H1_data, L1_data, V1_data]
psd_list = [H1_psd, L1_psd, V1_psd]
response_list = [H1_response, L1_response, V1_response]

def LogLikelihood(theta):
    theta = jnp.array(theta)
    # theta = theta.at[1].set(theta[1]/(1+theta[1])**2) # convert q to eta
    # theta = theta.at[7].set(jnp.arccos(theta[7])) # convert cos iota to iota
    # theta = theta.at[10].set(jnp.arcsin(theta[10])) # convert cos dec to dec
    theta_waveform = theta[:8]
    theta_waveform = theta_waveform.at[5].set(0)
    ra = theta[9]
    dec = theta[10]
    hp_test, hc_test = gen_IMRPhenomD_polar(f_list, theta_waveform, f_ref)
    align_time = jnp.exp(-1j*2*jnp.pi*f_list*(epoch+theta[5]))
    h_test_H1 = H1_response(f_list, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    h_test_L1 = L1_response(f_list, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    h_test_V1 = V1_response(f_list, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    df = f_list[1] - f_list[0]
    match_filter_SNR_H1 = 4*jnp.sum((jnp.conj(h_test_H1)*H1_data)/H1_psd*df).real
    match_filter_SNR_L1 = 4*jnp.sum((jnp.conj(h_test_L1)*L1_data)/L1_psd*df).real
    match_filter_SNR_V1 = 4*jnp.sum((jnp.conj(h_test_V1)*V1_data)/V1_psd*df).real
    optimal_SNR_H1 = 4*jnp.sum((jnp.conj(h_test_H1)*h_test_H1)/H1_psd*df).real
    optimal_SNR_L1 = 4*jnp.sum((jnp.conj(h_test_L1)*h_test_L1)/L1_psd*df).real
    optimal_SNR_V1 = 4*jnp.sum((jnp.conj(h_test_V1)*h_test_V1)/V1_psd*df).real

    return (match_filter_SNR_H1-optimal_SNR_H1/2) + (match_filter_SNR_L1-optimal_SNR_L1/2) + (match_filter_SNR_V1-optimal_SNR_V1/2)


logL = make_heterodyne_likelihood_mutliple_detector(data_list, psd_list, response_list, gen_IMRPhenomD_polar, ref_param, f_list, gmst, epoch, f_ref, heterodyne_bins)

# Fetch sampler parameters, construct sampler and initial guess

print("Making sampler")

n_dim = args['n_dim']
n_chains = args['n_chains']
n_loop_training = args['n_loop_training']
n_loop_production = args['n_loop_production']
n_local_steps = args['n_local_steps']
n_global_steps = args['n_global_steps']
learning_rate = args['learning_rate']
max_samples = args['max_samples']
momentum = args['momentum']
num_epochs = args['num_epochs']
batch_size = args['batch_size']
stepsize = args['stepsize']


guess_param = np.array(jnp.repeat(true_param_trans[None,:],int(n_chains),axis=0)*(1+0.1*jax.random.normal(jax.random.PRNGKey(seed+98127),shape=(int(n_chains),n_dim))))
guess_param[guess_param[:,1]>1,1] = 1

print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains, seed=seed)

print("Initializing MCMC model and normalizing flow model.")

prior_range = jnp.array([[10,50],[0.5,1.0],[-0.5,0.5],[-0.5,0.5],[300,2000],[-0.5,0.5],[0,2*np.pi],[-1,1],[0,np.pi],[0,2*np.pi],[-1,1]])


initial_position = jax.random.uniform(rng_key_set[0], shape=(int(n_chains), n_dim)) * 1
for i in range(n_dim):
    initial_position = initial_position.at[:,i].set(initial_position[:,i]*(prior_range[i,1]-prior_range[i,0])+prior_range[i,0])

from ripple import Mc_eta_to_ms
m1,m2 = jax.vmap(Mc_eta_to_ms)(guess_param[:,:2])
q = m2/m1
initial_position = initial_position.at[:,0].set(guess_param[:,0])
initial_position = initial_position.at[:,5].set(guess_param[:,5])

from astropy.cosmology import Planck18 as cosmo

z = np.linspace(0.01,0.4,10000)
dL = cosmo.luminosity_distance(z).value
dVdz = cosmo.differential_comoving_volume(z).value

def top_hat(x):
    output = 0.
    for i in range(n_dim):
        output = jax.lax.cond(x[i]>=prior_range[i,0], lambda: output, lambda: -jnp.inf)
        output = jax.lax.cond(x[i]<=prior_range[i,1], lambda: output, lambda: -jnp.inf)
    return output#+jnp.log(jnp.interp(x[4],dL,dVdz))

def posterior(theta):
    q = theta[1]
    iota = jnp.arccos(theta[7])
    dec = jnp.arcsin(theta[10])
    prior = top_hat(theta)
    theta = theta.at[1].set(q/(1+q)**2) # convert q to eta
    theta = theta.at[7].set(iota) # convert cos iota to iota
    theta = theta.at[10].set(dec) # convert cos dec to dec
    return logL(theta) + prior


model = RQSpline(n_dim, 10, [128,128], 8)


print("Initializing sampler class")

posterior = posterior
dposterior = jax.grad(posterior)


mass_matrix = np.eye(n_dim)
mass_matrix = np.abs(1./(jax.grad(logL)(true_param)+jax.grad(top_hat)(true_param)))*mass_matrix
mass_matrix = jnp.array(mass_matrix)

local_sampler = MALA(posterior, True, {"step_size": mass_matrix*3e-3})
print("Running sampler")

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    local_sampler,
    posterior,
    model,
    n_loop_training=n_loop_training,
    n_loop_production = n_loop_production,
    n_local_steps=n_local_steps,
    n_global_steps=n_global_steps,
    n_chains=n_chains,
    n_epochs=num_epochs,
    learning_rate=learning_rate,
    momentum=momentum,
    batch_size=batch_size,
    use_global=True,
    keep_quantile=0.,
    train_thinning = 40,
    local_autotune=mala_sampler_autotune
)

nf_sampler.sample(initial_position)

labels = ['Mc', 'eta', 'chi1', 'chi2', 'dist_mpc', 'tc', 'phic', 'cos_inclination', 'polarization_angle', 'ra', 'sin_dec']

print("Saving to output")

chains, log_prob, local_accs, global_accs, loss_vals = nf_sampler.get_sampler_state(training=True).values()
chains, log_prob, local_accs, global_accs = nf_sampler.get_sampler_state().values()

# Fetch output parameters

output_path = args['output_path']
downsample_factor = args['downsample_factor']

np.savez(args['output_path'], chains=chains[:,::downsample_factor], log_prob=log_prob[:,::downsample_factor], local_accs=local_accs[:,::downsample_factor], global_accs=global_accs[:,::downsample_factor], loss_vals=loss_vals, labels=labels, true_param=true_param, true_log_prob=LogLikelihood(true_param))
