import numpy as np
import jax.numpy as jnp
import jax

from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from jaxgw.PE.detector_preset import * 
from jaxgw.PE.heterodyneLikelihood import make_heterodyne_likelihood
from jaxgw.PE.detector_projection import make_detector_response

from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.MALA import make_mala_sampler, mala_sampler_autotune
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

ref_param = jnp.array([ 3.16158455e+01,  2.49059583e-01,  1.62840606e-02,  1.74049295e-02,
        4.43372352e+02,  2.01511565e+00,  1.64269305e+00,  6.83397730e-01,
        5.40140193e-01,  1.32292500e+00, -1.18695991e+00])


from jaxgw.PE.heterodyneLikelihood import make_heterodyne_likelihood_mutliple_detector

data_list = [H1_data, L1_data]
psd_list = [H1_psd, L1_psd]
response_list = [H1_response, L1_response]

logL = make_heterodyne_likelihood_mutliple_detector(data_list, psd_list, response_list, gen_IMRPhenomD_polar, ref_param, H1_frequency, 101)


n_dim = 11
n_chains = 1000
n_loop_training = 20
n_loop_production = 10
n_local_steps = 1000
n_global_steps = 1000
learning_rate = 0.001
max_samples = 50000
momentum = 0.9
num_epochs = 60
batch_size = 50000

guess_param = ref_param

guess_param = np.array(jnp.repeat(guess_param[None,:],int(n_chains),axis=0)*np.random.normal(loc=1,scale=0.1,size=(int(n_chains),n_dim)))
guess_param[guess_param[:,1]>0.25,1] = 0.249
guess_param[:,6] = (guess_param[:,6]%(2*jnp.pi))
guess_param[:,7] = (guess_param[:,7]%(jnp.pi))
guess_param[:,8] = (guess_param[:,8]%(jnp.pi))
guess_param[:,9] = (guess_param[:,9]%(2*jnp.pi))


print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains, seed=42)

print("Initializing MCMC model and normalizing flow model.")

prior_range = jnp.array([[10,80],[0.0,0.25],[0,1],[0,1],[0,2000],[1.9,2.1],[0,2*np.pi],[0,np.pi],[0,np.pi],[0,2*np.pi],[-jnp.pi/2,jnp.pi/2]])

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

mass_matrix = jnp.eye(n_dim)
mass_matrix = mass_matrix.at[1,1].set(1e-3)
mass_matrix = mass_matrix.at[5,5].set(1e-3)

local_sampler_caller = lambda x: make_mala_sampler(x, jit=True)
sampler_params = {'dt':mass_matrix*3e-3}
print("Running sampler")

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    local_sampler_caller,
    sampler_params,
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
)

nf_sampler.sample(initial_position)
