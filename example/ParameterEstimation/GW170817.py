import numpy as np
import jax.numpy as jnp
import jax

from lal import GreenwichMeanSiderealTime
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d


from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from jaxgw.PE.detector_preset import * 
from jaxgw.PE.heterodyneLikelihood import make_heterodyne_likelihood_mutliple_detector
from jaxgw.PE.detector_projection import make_detector_response

from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.MALA import make_mala_sampler
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *

minimum_frequency = 23
maximum_frequency = 700

trigger_time = event_gps("GW170817")
duration = 128
post_trigger_duration = 32
epoch = duration - post_trigger_duration
gmst = GreenwichMeanSiderealTime(trigger_time)
f_ref = minimum_frequency

# H1_data = TimeSeries.read('/mnt/home/misi/projects/cbc_birefringence/GW170817/raw_data/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf','H1:LOSC-STRAIN')
# H1_data = H1_data[(H1_data.times.value >= (trigger_time-epoch)) & (H1_data.times.value <= (trigger_time+post_trigger_duration))]
# n = len(H1_data)
# dt = H1_data.dt.value
# H1_data = np.fft.rfft(H1_data.value*tukey(n, 0.2))/4096
# H1_frequency = np.fft.rfftfreq(n, dt)
# H1_psd = np.genfromtxt('/mnt/home/misi/projects/cbc_birefringence/GW170817/psd_data/h1_psd.txt')
# H1_psd = interp1d(H1_psd[:,0], H1_psd[:,1], fill_value=np.inf,bounds_error=False)(H1_frequency[H1_frequency>minimum_frequency])
# H1_data = H1_data[H1_frequency>minimum_frequency]
# H1_frequency = H1_frequency[H1_frequency>minimum_frequency]

# L1_data = TimeSeries.read('/mnt/home/misi/projects/cbc_birefringence/GW170817/raw_data/L-L1_LOSC_CLN_4_V1-1187007040-2048.gwf','L1:LOSC-STRAIN')
# L1_data = L1_data[(L1_data.times.value >= (trigger_time-epoch)) & (L1_data.times.value <= (trigger_time+post_trigger_duration))]
# n = len(L1_data)
# dt = L1_data.dt.value
# L1_data = np.fft.rfft(L1_data.value*tukey(n, 0.2))/4096
# L1_frequency = np.fft.rfftfreq(n, dt)
# L1_psd = np.genfromtxt('/mnt/home/misi/projects/cbc_birefringence/GW170817/psd_data/l1_psd.txt')
# L1_psd = interp1d(L1_psd[:,0], L1_psd[:,1], fill_value=np.inf,bounds_error=False)(L1_frequency[L1_frequency>minimum_frequency])
# L1_data = L1_data[L1_frequency>minimum_frequency] 
# L1_frequency = L1_frequency[L1_frequency>minimum_frequency]

# V1_data = TimeSeries.read('/mnt/home/misi/projects/cbc_birefringence/GW170817/raw_data/V-V1_LOSC_CLN_4_V1-1187007040-2048.gwf','V1:LOSC-STRAIN')
# V1_data = V1_data[(V1_data.times.value >= (trigger_time-epoch)) & (V1_data.times.value <= (trigger_time+post_trigger_duration))]
# n = len(V1_data)
# dt = V1_data.dt.value
# V1_data = np.fft.rfft(V1_data.value*tukey(n, 0.2))/4096
# V1_frequency = np.fft.rfftfreq(n, dt)
# V1_psd = np.genfromtxt('/mnt/home/misi/projects/cbc_birefringence/GW170817/psd_data/v1_psd.txt')
# V1_psd = interp1d(V1_psd[:,0], V1_psd[:,1], fill_value=np.inf,bounds_error=False)(V1_frequency[V1_frequency>minimum_frequency])
# V1_data = V1_data[V1_frequency>minimum_frequency]
# V1_frequency = V1_frequency[V1_frequency>minimum_frequency]

data = np.load('./data/GW170817_data.npz',allow_pickle=True)


H1_frequency = data['frequency']
H1_data = data['data_dict'].tolist()['H1'][(H1_frequency>minimum_frequency)*(H1_frequency<maximum_frequency)]
H1_psd = data['psd_dict'].tolist()['H1'][(H1_frequency>minimum_frequency)*(H1_frequency<maximum_frequency)]
H1_frequency = H1_frequency[(H1_frequency>minimum_frequency)*(H1_frequency<maximum_frequency)]

L1_frequency = data['frequency']
L1_data = data['data_dict'].tolist()['L1'][(L1_frequency>minimum_frequency)*(L1_frequency<maximum_frequency)]
L1_psd = data['psd_dict'].tolist()['L1'][(L1_frequency>minimum_frequency)*(L1_frequency<maximum_frequency)]
L1_frequency = L1_frequency[(L1_frequency>minimum_frequency)*(L1_frequency<maximum_frequency)]

V1_frequency = data['frequency']
V1_data = data['data_dict'].tolist()['V1'][(V1_frequency>minimum_frequency)*(V1_frequency<maximum_frequency)]
V1_psd = data['psd_dict'].tolist()['V1'][(V1_frequency>minimum_frequency)*(V1_frequency<maximum_frequency)]
V1_frequency = V1_frequency[(V1_frequency>minimum_frequency)*(V1_frequency<maximum_frequency)]

H1 = get_H1()
H1_response = make_detector_response(H1[0], H1[1])
L1 = get_L1()
L1_response = make_detector_response(L1[0], L1[1])
V1 = get_V1()
V1_response = make_detector_response(V1[0], V1[1])

def genWaveform(theta):
    theta_waveform = theta[:8]
    theta_waveform = theta_waveform.at[5].set(0)
    ra = theta[9]
    dec = theta[10]
    hp_test, hc_test = gen_IMRPhenomD_polar(H1_frequency, theta_waveform, f_ref)
    align_time = jnp.exp(-1j*2*jnp.pi*H1_frequency*(epoch+theta[5]))
    h_test_H1 = H1_response(H1_frequency, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    return h_test_H1

def calculate_match_filter_SNR(theta):
    theta_waveform = theta[:8]
    theta_waveform = theta_waveform.at[5].set(0)
    ra = theta[9]
    dec = theta[10]
    hp_test, hc_test = gen_IMRPhenomD_polar(H1_frequency, theta_waveform, f_ref)
    align_time = jnp.exp(-1j*2*jnp.pi*H1_frequency*(epoch+theta[5]))
    h_test_H1 = H1_response(H1_frequency, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    h_test_L1 = L1_response(L1_frequency, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    h_test_V1 = V1_response(V1_frequency, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    df = H1_frequency[1] - H1_frequency[0]
    match_filter_SNR_H1 = 4*jnp.sum((jnp.conj(H1_data)*h_test_H1)/H1_psd*df).real
    match_filter_SNR_L1 = 4*jnp.sum((jnp.conj(L1_data)*h_test_L1)/L1_psd*df).real
    match_filter_SNR_V1 = 4*jnp.sum((jnp.conj(V1_data)*h_test_V1)/V1_psd*df).real
    optimal_SNR_H1 = 4*jnp.sum((jnp.conj(h_test_H1)*h_test_H1)/H1_psd*df).real
    optimal_SNR_L1 = 4*jnp.sum((jnp.conj(h_test_L1)*h_test_L1)/L1_psd*df).real
    optimal_SNR_V1 = 4*jnp.sum((jnp.conj(h_test_V1)*h_test_V1)/V1_psd*df).real
    return match_filter_SNR_H1, match_filter_SNR_L1, match_filter_SNR_V1, optimal_SNR_H1, optimal_SNR_L1, optimal_SNR_V1


def LogLikelihood(theta):
    theta = theta.at[1].set(theta[1]/(1+theta[1])**2) # convert q to eta
    theta = theta.at[7].set(jnp.arccos(theta[7])) # convert cos iota to iota
    theta = theta.at[10].set(jnp.arcsin(theta[10])) # convert cos dec to dec
    theta_waveform = theta[:8]
    theta_waveform = theta_waveform.at[5].set(0)
    ra = theta[9]
    dec = theta[10]
    hp_test, hc_test = gen_IMRPhenomD_polar(H1_frequency, theta_waveform, f_ref)
    align_time = jnp.exp(-1j*2*jnp.pi*H1_frequency*(epoch+theta[5]))
    h_test_H1 = H1_response(H1_frequency, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    h_test_L1 = L1_response(L1_frequency, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    h_test_V1 = V1_response(V1_frequency, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    df = H1_frequency[1] - H1_frequency[0]
    match_filter_SNR_H1 = 4*jnp.sum((jnp.conj(h_test_H1)*H1_data)/H1_psd*df).real
    match_filter_SNR_L1 = 4*jnp.sum((jnp.conj(h_test_L1)*L1_data)/L1_psd*df).real
    match_filter_SNR_V1 = 4*jnp.sum((jnp.conj(h_test_V1)*V1_data)/V1_psd*df).real
    optimal_SNR_H1 = 4*jnp.sum((jnp.conj(h_test_H1)*h_test_H1)/H1_psd*df).real
    optimal_SNR_L1 = 4*jnp.sum((jnp.conj(h_test_L1)*h_test_L1)/L1_psd*df).real
    optimal_SNR_V1 = 4*jnp.sum((jnp.conj(h_test_V1)*h_test_V1)/V1_psd*df).real

    return (match_filter_SNR_H1-optimal_SNR_H1/2) + (match_filter_SNR_L1-optimal_SNR_L1/2) + (match_filter_SNR_V1-optimal_SNR_V1/2)

ref_param = jnp.array([ 1.19736744e+00,  0.24985044062115083, -1.18532170e-01,
        1.02293135e-01,  3.35316272e+01,  3.03494379e-02,
        1.86495116e+00, 2.37025514,  2.06376050e+00,
        3.42839234e+00, -0.37789968])

data_list = [H1_data, L1_data, V1_data]
psd_list = [H1_psd, L1_psd, V1_psd]
response_list = [H1_response, L1_response, V1_response]

logL = make_heterodyne_likelihood_mutliple_detector(data_list, psd_list, response_list, gen_IMRPhenomD_polar, ref_param, H1_frequency, gmst, epoch, f_ref, 301)

n_dim = 11
n_chains = 1000
n_loop_training = 20
n_loop_production = 10
n_local_steps = 200
n_global_steps = 200
learning_rate = 0.001
max_samples = 50000
momentum = 0.9
num_epochs = 60
batch_size = 50000

guess_param = ref_param

guess_param = np.array(jnp.repeat(guess_param[None,:],int(n_chains),axis=0)*np.random.normal(loc=1,scale=0.00001,size=(int(n_chains),n_dim)))
guess_param[guess_param[:,1]>0.25,1] = 0.249


print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains, seed=42)

print("Initializing MCMC model and normalizing flow model.")

prior_range = jnp.array([[1.18,1.21],[0.125,1],[-0.3,0.3],[-0.3,0.3],[1,75],[-0.1,0.1],[0,2*np.pi],[-1,1],[0,np.pi],[0,2*np.pi],[-1,1]])
# prior_range = jnp.array([[1.18,1.21],[0.2,0.25],[0.0,0.3],[0.0,0.3],[1,75],[-0.1,0.1],[0,2*np.pi],[0,np.pi],[0,np.pi],[0,2*np.pi],[-np.pi/2,np.pi/2]])

initial_position = jax.random.uniform(rng_key_set[0], shape=(int(n_chains), n_dim)) * 1
for i in range(n_dim):
    initial_position = initial_position.at[:,i].set(initial_position[:,i]*(prior_range[i,1]-prior_range[i,0])+prior_range[i,0])

from ripple import Mc_eta_to_ms
m1,m2 = jax.vmap(Mc_eta_to_ms)(guess_param[:,:2])
q = m2/m1

# initial_position = initial_position.at[:,0].set(guess_param[:,0])
# initial_position = initial_position.at[:,1].set(q)
# initial_position = initial_position.at[:,2].set(guess_param[:,2])
# initial_position = initial_position.at[:,3].set(guess_param[:,3])
# initial_position = initial_position.at[:,4].set(guess_param[:,4])
initial_position = initial_position.at[:,5].set(guess_param[:,5])
# initial_position = initial_position.at[:,6].set(guess_param[:,6])
# initial_position = initial_position.at[:,7].set(jnp.cos(guess_param[:,7]))
# initial_position = initial_position.at[:,8].set(guess_param[:,8])
# initial_position = initial_position.at[:,9].set(guess_param[:,9])
# initial_position = initial_position.at[:,10].set(jnp.cos(guess_param[:,10]))

from astropy.cosmology import Planck18 as cosmo

z = np.linspace(0.0002,0.03,10000)
dL = cosmo.luminosity_distance(z).value
dVdz = cosmo.differential_comoving_volume(z).value

def top_hat(x):
    output = 0.
    for i in range(n_dim):
        output = jax.lax.cond(x[i]>=prior_range[i,0], lambda: output, lambda: -jnp.inf)
        output = jax.lax.cond(x[i]<=prior_range[i,1], lambda: output, lambda: -jnp.inf)
    return output+jnp.log(jnp.interp(x[4],dL,dVdz))

def log_likelihood(theta):
    theta = theta.at[1].set(theta[1]/(1+theta[1])**2) # convert q to eta
    theta = theta.at[7].set(jnp.arccos(theta[7])) # convert cos iota to iota
    theta = theta.at[10].set(jnp.arcsin(theta[10])) # convert cos dec to dec
    return logL(theta)

def posterior(theta):
    q = theta[1]
    prior = top_hat(theta)
    theta = theta.at[1].set(q/(1+q)**2) # convert q to eta
    theta = theta.at[7].set(jnp.arccos(theta[7])) # convert cos iota to iota
    theta = theta.at[10].set(jnp.arcsin(theta[10])) # convert cos dec to dec

    return logL(theta) + prior

model = RQSpline(n_dim, 10, [128,128], 8)

print("Initializing sampler class")

posterior = posterior

mass_matrix = jnp.eye(n_dim)
mass_matrix = mass_matrix.at[0,0].set(1e-5)
mass_matrix = mass_matrix.at[1,1].set(1e-4)
mass_matrix = mass_matrix.at[2,2].set(1e-3)
mass_matrix = mass_matrix.at[3,3].set(1e-3)
mass_matrix = mass_matrix.at[5,5].set(1e-5)
mass_matrix = mass_matrix.at[9,9].set(1e-2)
mass_matrix = mass_matrix.at[10,10].set(1e-2)

local_sampler_caller = lambda x: make_mala_sampler(x, jit=True)
sampler_params = {'dt':mass_matrix*3e-2}
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
    train_thinning = 40,
)

nf_sampler.sample(initial_position)
