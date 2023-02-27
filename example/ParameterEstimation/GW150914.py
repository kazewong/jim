import numpy as np
import jax.numpy as jnp
import jax
from lal import GreenwichMeanSiderealTime

from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from jimgw.PE.detector_preset import * 
from jimgw.PE.heterodyneLikelihood import make_heterodyne_likelihood
from jimgw.PE.detector_projection import make_detector_response

from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.Sampler import Sampler
from flowMC.sampler.MALA import MALA
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

trigger_time = 1126259462.4
duration = 4 
post_trigger_duration = 2
epoch = duration - post_trigger_duration
gmst = GreenwichMeanSiderealTime(trigger_time)
f_ref = 20

def gen_waveform_H1(f, theta, epoch, gmst, f_ref):
    theta_waveform = theta[:8]
    theta_waveform = theta_waveform.at[5].set(0)
    ra = theta[9]
    dec = theta[10]
    hp, hc = gen_IMRPhenomD_polar(f, theta_waveform, f_ref)
    return H1_response(f, hp, hc, ra, dec, gmst , theta[8]) * jnp.exp(-1j*2*jnp.pi*f*(epoch+theta[5]))

def gen_waveform_L1(f, theta, epoch, gmst, f_ref):
    theta_waveform = theta[:8]
    theta_waveform = theta_waveform.at[5].set(0)
    ra = theta[9]
    dec = theta[10]
    hp, hc = gen_IMRPhenomD_polar(f, theta_waveform, f_ref)
    return L1_response(f, hp, hc, ra, dec, gmst, theta[8]) * jnp.exp(-1j*2*jnp.pi*f*(epoch+theta[5]))

def H1_LogLikelihood(theta):
    h_test = gen_waveform_H1(H1_frequency, theta, epoch, gmst, f_ref)
    df = H1_frequency[1] - H1_frequency[0] 
    match_filter_SNR = 4*jnp.sum((jnp.conj(h_test)*H1_data)/H1_psd*df).real
    optimal_SNR = 4*jnp.sum((jnp.conj(h_test)*h_test)/H1_psd*df).real
    return (match_filter_SNR-optimal_SNR/2)

def L1_LogLikelihood(theta):
    h_test = gen_waveform_L1(L1_frequency, theta, epoch, gmst, f_ref)
    df = L1_frequency[1] - L1_frequency[0] 
    match_filter_SNR = 4*jnp.sum((jnp.conj(h_test)*L1_data)/L1_psd*df).real
    optimal_SNR = 4*jnp.sum((jnp.conj(h_test)*h_test)/L1_psd*df).real
    return (match_filter_SNR-optimal_SNR/2)

ref_param = jnp.array([ 3.13857132e+01,  2.49301122e-01,  1.31593299e-02,  2.61342217e-03,
        5.37766606e+02,  1.18679090e-02,  1.26153956e+00,  2.61240760e+00,
        1.33131339e+00,  2.33978644e+00, -1.20993116e+00])


from jimgw.PE.heterodyneLikelihood import make_heterodyne_likelihood_mutliple_detector

data_list = [H1_data, L1_data]
psd_list = [H1_psd, L1_psd]
response_list = [H1_response, L1_response]

logL = make_heterodyne_likelihood_mutliple_detector(data_list, psd_list, response_list, gen_IMRPhenomD_polar, ref_param, H1_frequency, gmst, epoch, f_ref, 301)


n_dim = 11
n_chains = 1000
n_loop_training = 20
n_loop_production = 10
n_local_steps = 200
n_global_steps = 200
learning_rate = 0.001
max_samples = 100000
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

prior_range = jnp.array([[10,80],[0.125,1.0],[-1,1],[-1,1],[0,2000],[-0.1,0.1],[0,2*np.pi],[-1,1],[0,np.pi],[0,2*np.pi],[-1,1]])


initial_position = jax.random.uniform(rng_key_set[0], shape=(int(n_chains), n_dim)) * 1
for i in range(n_dim):
    initial_position = initial_position.at[:,i].set(initial_position[:,i]*(prior_range[i,1]-prior_range[i,0])+prior_range[i,0])

from ripple import Mc_eta_to_ms
m1,m2 = jax.vmap(Mc_eta_to_ms)(guess_param[:,:2])
q = m2/m1

initial_position = initial_position.at[:,0].set(guess_param[:,0])

from astropy.cosmology import Planck18 as cosmo

z = np.linspace(0.002,3,10000)
dL = cosmo.luminosity_distance(z).value
dVdz = cosmo.differential_comoving_volume(z).value

def top_hat(x):
    output = 0.
    for i in range(n_dim):
        output = jax.lax.cond(x[i]>=prior_range[i,0], lambda: output, lambda: -jnp.inf)
        output = jax.lax.cond(x[i]<=prior_range[i,1], lambda: output, lambda: -jnp.inf)
    return output+jnp.log(jnp.interp(x[4],dL,dVdz))

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

mass_matrix = jnp.eye(n_dim)
mass_matrix = mass_matrix.at[1,1].set(1e-3)
mass_matrix = mass_matrix.at[5,5].set(1e-3)

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
    train_thinning = 40
)

nf_sampler.sample(initial_position)
chains, log_prob, local_accs, global_accs = nf_sampler.get_sampler_state().values()
np.savez('/mnt/home/wwong/ceph/GWProject/JaxGW/RealtimePE/GW150914.npz', chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)