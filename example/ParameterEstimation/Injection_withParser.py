# Import packages
import lalsimulation as lalsim 
import numpy as np
import jax.numpy as jnp
import jax

# from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from ripple import ms_to_Mc_eta
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from jaxgw.PE.detector_preset import * 
from jaxgw.PE.heterodyneLikelihood import make_heterodyne_likelihood_mutliple_detector
from jaxgw.PE.detector_projection import make_detector_response
from jaxgw.PE.generate_noise import generate_noise

from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.MALA import make_mala_sampler
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *

import argparse
import yaml

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
parser.add_argument('--n_loop', type=int, default=None, help='number of loops')
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


def gen_waveform_H1(f, theta):
    theta_waveform = theta[:9]
    ra = theta[9]
    dec = theta[10]
    hp, hc = gen_IMRPhenomD_polar(f, theta_waveform)
    return H1_response(f, hp, hc, ra, dec, theta[5], theta[8])

def gen_waveform_L1(f, theta):
    theta_waveform = theta[:9]
    ra = theta[9]
    dec = theta[10]
    hp, hc = gen_IMRPhenomD_polar(f, theta_waveform)
    return L1_response(f, hp, hc, ra, dec, theta[5], theta[8])

true_param = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle, ra, dec])


f_list = freqs[freqs>fmin]
H1_signal = gen_waveform_H1(f_list, true_param)
H1_noise_psd = noise_dict['H1'][freqs>fmin]
H1_data = H1_noise_psd + H1_signal

L1_signal = gen_waveform_L1(f_list, true_param)
L1_noise_psd = noise_dict['L1'][freqs>fmin]
L1_data = L1_noise_psd + L1_signal

ref_param = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle, ra, dec])

data_list = [H1_data, L1_data]
psd_list = [psd_dict['H1'], psd_dict['L1']]
response_list = [H1_response, L1_response]

logL = make_heterodyne_likelihood_mutliple_detector(data_list, psd_list, response_list, gen_IMRPhenomD_polar, ref_param, f_list, heterodyne_bins)

# Fetch sampler parameters, construct sampler and initial guess

print("Making sampler")

n_dim = args['n_dim']
n_chains = args['n_chains']
n_loop = args['n_loop']
n_local_steps = args['n_local_steps']
n_global_steps = args['n_global_steps']
learning_rate = args['learning_rate']
max_samples = args['max_samples']
momentum = args['momentum']
num_epochs = args['num_epochs']
batch_size = args['batch_size']
stepsize = args['stepsize']


guess_param = np.array(jnp.repeat(true_param[None,:],int(n_chains),axis=0)*(1+0.1*jax.random.normal(jax.random.PRNGKey(seed+98127),shape=(int(n_chains),n_dim))))
guess_param[guess_param[:,1]>0.25,1] = 0.249
guess_param[:,6] = (guess_param[:,6]+np.pi/2)%(np.pi)-np.pi/2
guess_param[:,7] = (guess_param[:,7]+np.pi/2)%(np.pi)-np.pi/2
guess_param[:,8] = (guess_param[:,8]%(2*np.pi))
guess_param[:,9] = (guess_param[:,9]%(2*np.pi))
guess_param[:,10] = (guess_param[:,10]%(np.pi))

print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains, seed=seed)

print("Initializing MCMC model and normalizing flow model.")

prior_range = jnp.array([[10,70],[0.0,0.25],[-1,1],[-1,1],[0,2000],[-5,5],[-np.pi/2,np.pi/2],[-np.pi/2,np.pi/2],[0,2*np.pi],[0,2*np.pi],[0,np.pi]])

initial_position = jax.random.uniform(rng_key_set[0], shape=(int(n_chains), n_dim)) * 1
for i in range(n_dim):
    initial_position = initial_position.at[:,i].set(initial_position[:,i]*(prior_range[i,1]-prior_range[i,0])+prior_range[i,0])

initial_position = initial_position.at[:,0].set(guess_param[:,0])
initial_position = initial_position.at[:,1].set(guess_param[:,1])
initial_position = initial_position.at[:,5].set(guess_param[:,5])

def top_hat(x):
    output = 0.
    for i in range(n_dim):
        output = jax.lax.cond(x[i]>=prior_range[i,0], lambda: output, lambda: -jnp.inf)
        output = jax.lax.cond(x[i]<=prior_range[i,1], lambda: output, lambda: -jnp.inf)
    return output

def posterior(theta):
    prior = top_hat(theta)
    return logL(theta) + prior


model = RQSpline(n_dim, 10, [128,128], 8)


print("Initializing sampler class")

posterior = posterior
dposterior = jax.grad(posterior)

mass_matrix = jnp.eye(n_dim)
mass_matrix = mass_matrix.at[1,1].set(1e-3)
mass_matrix = mass_matrix.at[5,5].set(1e-3)

local_sampler,updater, kernel, logp, dlogp = make_mala_sampler(posterior, dposterior,2e-3, jit=True, M=mass_matrix)

print("Running sampler")

nf_sampler = Sampler(n_dim, rng_key_set, model, local_sampler,
                    posterior,
                    d_likelihood=dposterior,
                    n_loop=n_loop,
                    n_local_steps=n_local_steps,
                    n_global_steps=n_global_steps,
                    n_chains=n_chains,
                    stepsize=stepsize,
                    n_nf_samples=100,
                    learning_rate=learning_rate,
                    n_epochs= num_epochs,
                    max_samples = max_samples,
                    momentum=momentum,
                    batch_size=batch_size,
                    use_global=True,
                    keep_quantile=0.5)

# nf_sampler.sample(initial_position)

# labels = ['Mc', 'eta', 'chi1', 'chi2', 'dist_mpc', 'tc', 'phic', 'inclination', 'polarization_angle', 'ra', 'dec']

# print("Saving to output")

# chains, log_prob, local_accs, global_accs, loss_vals = nf_sampler.get_sampler_state()

# # Fetch output parameters

# output_path = args['output_path']
# downsample_factor = args['downsample_factor']

# np.savez(args['output_path'], chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs, loss_vals=loss_vals, labels=labels)
