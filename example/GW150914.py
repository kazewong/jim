import numpy as np
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp
import jax

from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from jimgw.PE.detector_preset import * 
from jimgw.PE.heterodyneLikelihood import make_heterodyne_likelihood_mutliple_detector
from jimgw.PE.detector_projection import make_detector_response

from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.sampler.Sampler import Sampler
from flowMC.sampler.MALA import MALA
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *
from flowMC.utils.EvolutionaryOptimizer import EvolutionaryOptimizer

from astropy.time import Time

# We only use this to grab the data
from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
start = gps - 2
end = gps + 2
fmin = 20
fmax = 1024

ifos = ['H1', 'L1']

print("Fetching data...")
data_td_dict = {ifo: TimeSeries.fetch_open_data(ifo, start, end) for ifo in ifos}
print("Finished fetching data.")

# GWpy normalizes the FFT like an instrumentalist would, which is not what we 
# want for the likelihoood, so fix this manually
n = len(data_td_dict[ifos[0]])
delta_t = data_td_dict[ifos[0]].dt.value

print("Computing the FFTs...")
# For BNS 0.00625 is a good choice for the tukey window
# For BBH 0.2 is a good choice for the tukey window
data_fd_dict = {i: np.fft.rfft(np.array(d)*tukey(n, 0.2))*delta_t 
                   for i, d in data_td_dict.items()}

freq = np.fft.rfftfreq(n, delta_t)

# # We take a bit of extra data to compute PSDs
start_psd = int(gps) - 16
end_psd = int(gps) + 16

print("Fetching PSD data...")
psd_data_td_dict = {ifo: TimeSeries.fetch_open_data(ifo, start_psd, end_psd) for ifo in ifos}
psd_dict = {i: d.psd(fftlength=4) for i, d in psd_data_td_dict.items()}
print("Finished generating data.")

H1_frequency = np.array(freq[(freq>fmin)&(freq<fmax)])
H1_data = np.array(data_fd_dict['H1'].data)[(freq>fmin)&(freq<fmax)]
H1_psd = np.array(psd_dict['H1'].data)[(freq>fmin)&(freq<fmax)]

L1_frequency = np.array(freq[(freq>fmin)&(freq<fmax)])
L1_data = np.array(data_fd_dict['L1'].data)[(freq>fmin)&(freq<fmax)]
L1_psd = np.array(psd_dict['L1'].data)[(freq>fmin)&(freq<fmax)]


###########################################
######## Set up the likelihood ############
###########################################

H1 = get_H1()
H1_response = make_detector_response(H1[0], H1[1])
L1 = get_L1()
L1_response = make_detector_response(L1[0], L1[1])

trigger_time = 1126259462.4
duration = 4 
post_trigger_duration = 2
epoch = duration - post_trigger_duration
gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
f_ref = 20

def LogLikelihood(theta):
    theta_waveform = theta[:8]
    theta_waveform = theta_waveform.at[5].set(0)
    ra = theta[9]
    dec = theta[10]
    hp_test, hc_test = gen_IMRPhenomD_polar(H1_frequency, theta_waveform, f_ref)
    align_time = jnp.exp(-1j*2*jnp.pi*H1_frequency*(epoch+theta[5]))
    h_test_H1 = H1_response(H1_frequency, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    h_test_L1 = L1_response(L1_frequency, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    df = H1_frequency[1] - H1_frequency[0]
    match_filter_SNR_H1 = 4*jnp.sum((jnp.conj(h_test_H1)*H1_data)/H1_psd*df).real
    match_filter_SNR_L1 = 4*jnp.sum((jnp.conj(h_test_L1)*L1_data)/L1_psd*df).real
    optimal_SNR_H1 = 4*jnp.sum((jnp.conj(h_test_H1)*h_test_H1)/H1_psd*df).real
    optimal_SNR_L1 = 4*jnp.sum((jnp.conj(h_test_L1)*h_test_L1)/L1_psd*df).real

    return (match_filter_SNR_H1-optimal_SNR_H1/2) + (match_filter_SNR_L1-optimal_SNR_L1/2)

# prior on the waveform parameters
# these are Mc, eta, s1, s2, dist, tc, phic, ra, dec, psi
prior_range = jnp.array([[20.,50.],[0.20,0.25],[-0.9,0.9],
                         [-0.9,0.9],[100,3000],[-1.0,1.0],
                         [0,2*np.pi],[0.001,np.pi],[0.001,np.pi],
                         [0.001,2*np.pi],[-jnp.pi/2,jnp.pi/2]])


###########################################
##### Optimize to find high L point #######
###########################################

set_nwalkers = 100
initial_guess = jax.random.uniform(jax.random.PRNGKey(42), (set_nwalkers,11,),
                                    minval=prior_range[:,0], maxval=prior_range[:,1])

y = lambda x: -LogLikelihood(x)
y = jax.jit(jax.vmap(y))
print("Compiling likelihood function")
y(initial_guess)
print("Done compiling")

print("Starting the optimizer")
optimizer = EvolutionaryOptimizer(11, verbose = True)
state = optimizer.optimize(y, prior_range, n_loops=2000)
best_fit = optimizer.get_result()[0]

print(best_fit)

data_list = [H1_data, L1_data]
psd_list = [H1_psd, L1_psd]
response_list = [H1_response, L1_response]


print("Constructing the heterodyned likelihood function")
logL = make_heterodyne_likelihood_mutliple_detector(data_list, psd_list, response_list, gen_IMRPhenomD_polar,
                                                    best_fit, H1_frequency, gmst, epoch, f_ref, 301)


###########################################
####### Finally, we can sample! ###########
###########################################

n_dim = 11
n_chains = 500
n_loop_training = 15
n_loop_production = 10
n_local_steps = 100
n_global_steps = 100
learning_rate = 0.001
max_samples = 100000
momentum = 0.9
num_epochs = 200
batch_size = 50000

guess_param = best_fit

guess_param = np.array(jnp.repeat(guess_param[None,:],int(n_chains),axis=0)*np.random.normal(loc=1,scale=0.1,size=(int(n_chains),n_dim)))
guess_param[guess_param[:,1]>0.25,1] = 0.249
guess_param[:,6] = (guess_param[:,6]%(2*jnp.pi))
guess_param[:,7] = (guess_param[:,7]%(jnp.pi))
guess_param[:,8] = (guess_param[:,8]%(jnp.pi))
guess_param[:,9] = (guess_param[:,9]%(2*jnp.pi))


print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains, seed=42)

print("Initializing MCMC model and normalizing flow model.")

prior_range = jnp.array([[10,80],[0.125,1.0],[-1,1],[-1,1],
                        [0,2000],[-0.05,0.05],[0,2*np.pi],[-1,1],
                        [0,np.pi],[0,2*np.pi],[-1,1]])


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
    theta = theta.at[7].set(jnp.arcsin(jnp.sin(theta[7]/2*jnp.pi))*2/jnp.pi)
    theta = theta.at[10].set(jnp.arcsin(jnp.sin(theta[10]/2*jnp.pi))*2/jnp.pi)
    iota = jnp.arccos(theta[7])
    dec = jnp.arcsin(theta[10])
    prior = top_hat(theta)
    theta = theta.at[1].set(q/(1+q)**2) # convert q to eta
    theta = theta.at[7].set(iota) # convert cos iota to iota
    theta = theta.at[10].set(dec) # convert cos dec to dec
    return logL(theta) + prior

posterior_new = lambda theta, data: posterior(theta)

model = MaskedCouplingRQSpline(n_dim, 10, [128,128], 8, jax.random.PRNGKey(10))

print("Initializing sampler class")

mass_matrix = jnp.eye(n_dim)
mass_matrix = mass_matrix.at[1,1].set(1e-3)
mass_matrix = mass_matrix.at[5,5].set(1e-3)

local_sampler = MALA(posterior_new, True, {"step_size": mass_matrix*3e-3})
print("Running sampler")

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    None,
    local_sampler,
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
)

nf_sampler.sample(initial_position, None)
chains, log_prob, local_accs, global_accs = nf_sampler.get_sampler_state().values()
print("Script complete and took: {} minutes".format((time.time()-total_time_start)/60))
# np.savez('GW150914.npz', chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)