import numpy as np
import jax.numpy as jnp
import jax

from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from jaxgw.PE.detector_preset import * 
from jaxgw.PE.heterodyneLikelihood import make_heterodyne_likelihood
from jaxgw.PE.detector_projection import make_detector_response

from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.MALA import make_mala_sampler
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *

data = np.load('./data/GW150914_data.npz',allow_pickle=True)

minimum_frequency = data['minimum_frequency']

H1_frequency = data['frequency'].tolist()['H1']
H1_data = data['data'].tolist()['H1'][H1_frequency>minimum_frequency]
H1_psd = data['psd'].tolist()['H1'][H1_frequency>minimum_frequency]
H1_frequency = H1_frequency[H1_frequency>minimum_frequency]

L1_frequency = data['frequency'].tolist()['L1']
L1_data = data['data'].tolist()['L1'][L1_frequency>minimum_frequency]
L1_psd = data['psd'].tolist()['L1'][L1_frequency>minimum_frequency]
L1_frequency = L1_frequency[L1_frequency>minimum_frequency]

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

def H1_LogLikelihood(theta):
    h_test = gen_waveform_H1(H1_frequency,theta)
    df = H1_frequency[1] - H1_frequency[0] 
    match_filter_SNR = 4*jnp.sum((jnp.conj(h_test)*H1_data)/H1_psd*df).real
    optimal_SNR = 4*jnp.sum((jnp.conj(h_test)*h_test)/H1_psd*df).real
    return (match_filter_SNR-optimal_SNR/2)

def L1_LogLikelihood(theta):
    h_test = gen_waveform_L1(L1_frequency,theta)
    df = L1_frequency[1] - L1_frequency[0] 
    match_filter_SNR = 4*jnp.sum((jnp.conj(h_test)*L1_data)/L1_psd*df).real
    optimal_SNR = 4*jnp.sum((jnp.conj(h_test)*h_test)/L1_psd*df).real
    return (match_filter_SNR-optimal_SNR/2)

ref_param = jnp.array([ 3.41096639e+01,  2.42240502e-01,  7.03845904e-02,
              1.45055597e-01,  4.00156164e+02, -1.97202379e+00,
              1.08177416e+00, -6.94499550e-02,  1.95503312e+00,
              8.60901399e-01,  2.89425087e+00])

ref_param = ref_param.at[-1].set(ref_param[-1]%(jnp.pi))
ref_param = ref_param.at[6].set((ref_param[6]+jnp.pi/2)%(jnp.pi)-jnp.pi/2)

H1_logL = make_heterodyne_likelihood(H1_data, gen_waveform_H1, ref_param, H1_psd, H1_frequency, 101)
L1_logL = make_heterodyne_likelihood(L1_data, gen_waveform_L1, ref_param, L1_psd, L1_frequency, 101)

from jaxgw.PE.heterodyneLikelihood import make_heterodyne_likelihood_mutliple_detector

data_list = [H1_data, L1_data]
psd_list = [H1_psd, L1_psd]
response_list = [H1_response, L1_response]

logL = make_heterodyne_likelihood_mutliple_detector(data_list, psd_list, response_list, gen_IMRPhenomD_polar, ref_param, H1_frequency, 101)


n_dim = 11
n_chains = 1000
n_loop = 10
n_local_steps = 1000
n_global_steps = 1000
learning_rate = 0.001
max_samples = 50000
momentum = 0.9
num_epochs = 300
batch_size = 50000
stepsize = 0.01

guess_param = ref_param

guess_param = np.array(jnp.repeat(guess_param[None,:],int(n_chains),axis=0)*np.random.normal(loc=1,scale=0.1,size=(int(n_chains),n_dim)))
guess_param[guess_param[:,1]>0.25,1] = 0.249
guess_param[:,6] = (guess_param[:,6]+np.pi/2)%(np.pi)-np.pi/2
guess_param[:,7] = (guess_param[:,7]+np.pi/2)%(np.pi)-np.pi/2


print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains, seed=42)

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
    return H1_logL(theta) + L1_logL(theta) + prior

model = RQSpline(n_dim, 10, [128,128], 8)

print("Initializing sampler class")

posterior = posterior
dposterior = jax.grad(posterior)

mass_matrix = jnp.eye(n_dim)
mass_matrix = mass_matrix.at[1,1].set(1e-3)
mass_matrix = mass_matrix.at[5,5].set(1e-2)

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
                    keep_quantile=0.)

nf_sampler.sample(initial_position)
