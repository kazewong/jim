import numpy as np
import jax.numpy as jnp
import jax

from lal import GreenwichMeanSiderealTime
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d


from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from jimgw.PE.detector_preset import * 
from jimgw.PE.heterodyneLikelihood import make_heterodyne_likelihood_mutliple_detector
from jimgw.PE.detector_projection import make_detector_response

from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.MALA import MALA
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

data = np.load('/mnt/home/wwong/ceph/GWProject/JaxGW/RealtimePE/GW170817_data.npz',allow_pickle=True)


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
    print("Compiling")
    # theta = theta.at[1].set(theta[1]/(1+theta[1])**2) # convert q to eta
    # theta = theta.at[7].set(jnp.arccos(theta[7])) # convert cos iota to iota
    # theta = theta.at[10].set(jnp.arcsin(theta[10])) # convert cos dec to dec
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


def ridge_reg_objective(params):
    print("Compiling")
    residuals = params
    return jnp.mean(residuals ** 2)

prior_range = jnp.array([[1.1,1.5],[0.20,0.25],[-0.2,0.2],[-0.1,0.1],[10,200],[-0.029,0.031],[0,2*np.pi],[0.001,np.pi],[0.001,np.pi],[0.001,2*np.pi],[-jnp.pi/2,jnp.pi/2]])


initial_guess = jax.random.uniform(jax.random.PRNGKey(42), (20,11,), minval=prior_range[:,0], maxval=prior_range[:,1])
ref_param = jnp.array([ 1.19736744e+00,  0.24985044062115083, -1.18532170e-01,
        1.02293135e-01,  3.35316272e+01,  3.03494379e-02,
        1.86495116e+00, 2.37025514,  2.06376050e+00,
        3.42839234e+00, -0.37789968])

y = lambda x: -LogLikelihood(x)
y = jax.jit(jax.vmap(y))
print("Compiling the function")
y(initial_guess)
print("Done compiling the function")

import jax
from evosax import CMA_ES

# Instantiate the search strategy
rng = jax.random.PRNGKey(0)
strategy = CMA_ES(popsize=100, num_dims=11, elite_ratio=0.5)
es_params = strategy.default_params
es_params = es_params.replace(clip_min=0, clip_max=1)
state = strategy.initialize(rng, es_params)

# Run ask-eval-tell loop - NOTE: By default minimization!
for t in range(1000):
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    x, state = strategy.ask(rng_gen, state, es_params)
    theta = x*(prior_range[:,1]-prior_range[:,0]) + prior_range[:,0]
    fitness = y(theta)
    state = strategy.tell(x, fitness.astype(jnp.float32), state, es_params)
    if t % 10 == 0:
        print(f"Generation {t}, best fitness: {state.best_fitness}")

# Get best overall population member & its fitness
state.best_member, state.best_fitness