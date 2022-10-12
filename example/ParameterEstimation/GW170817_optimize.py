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
maximum_frequency = 2048

trigger_time = event_gps("GW170817")
duration = 128
post_trigger_duration = 2
epoch = duration - post_trigger_duration
gmst = GreenwichMeanSiderealTime(trigger_time)
f_ref = minimum_frequency

H1_data = TimeSeries.read('/mnt/home/misi/projects/cbc_birefringence/GW170817/raw_data/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf','H1:LOSC-STRAIN')
H1_data = H1_data[(H1_data.times.value >= (trigger_time-epoch)) & (H1_data.times.value <= (trigger_time+post_trigger_duration))]
n = len(H1_data)
H1_data = np.fft.rfft(H1_data.value*tukey(n, 0.00625))/4096.
H1_frequency = np.fft.rfftfreq(n, 1/4096.)
H1_psd = np.genfromtxt('/mnt/home/misi/projects/cbc_birefringence/GW170817/psd_data/h1_psd.txt')
H1_psd = interp1d(H1_psd[:,0], H1_psd[:,1], fill_value=np.inf,bounds_error=False)(H1_frequency[H1_frequency>minimum_frequency])
H1_data = H1_data[H1_frequency>minimum_frequency]
H1_frequency = H1_frequency[H1_frequency>minimum_frequency]

L1_data = TimeSeries.read('/mnt/home/misi/projects/cbc_birefringence/GW170817/raw_data/L-L1_LOSC_CLN_4_V1-1187007040-2048.gwf','L1:LOSC-STRAIN')
L1_data = L1_data[(L1_data.times.value >= (trigger_time-epoch)) & (L1_data.times.value <= (trigger_time+post_trigger_duration))]
n = len(L1_data)
L1_data = np.fft.rfft(L1_data.value*tukey(n, 0.00625))/4096.
L1_frequency = np.fft.rfftfreq(n, 1/4096.)
L1_psd = np.genfromtxt('/mnt/home/misi/projects/cbc_birefringence/GW170817/psd_data/l1_psd.txt')
L1_psd = interp1d(L1_psd[:,0], L1_psd[:,1], fill_value=np.inf,bounds_error=False)(L1_frequency[L1_frequency>minimum_frequency])
L1_data = L1_data[L1_frequency>minimum_frequency] 
L1_frequency = L1_frequency[L1_frequency>minimum_frequency]

V1_data = TimeSeries.read('/mnt/home/misi/projects/cbc_birefringence/GW170817/raw_data/V-V1_LOSC_CLN_4_V1-1187007040-2048.gwf','V1:LOSC-STRAIN')
V1_data = V1_data[(V1_data.times.value >= (trigger_time-epoch)) & (V1_data.times.value <= (trigger_time+post_trigger_duration))]
n = len(V1_data)
V1_data = np.fft.rfft(V1_data.value*tukey(n, 0.00625))/4096.
V1_frequency = np.fft.rfftfreq(n, 1/4096.)
V1_psd = np.genfromtxt('/mnt/home/misi/projects/cbc_birefringence/GW170817/psd_data/v1_psd.txt')
V1_psd = interp1d(V1_psd[:,0], V1_psd[:,1], fill_value=np.inf,bounds_error=False)(V1_frequency[V1_frequency>minimum_frequency])
V1_data = V1_data[V1_frequency>minimum_frequency]
V1_frequency = V1_frequency[V1_frequency>minimum_frequency]

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

prior_range = jnp.array([[1.1,1.5],[0.20,0.25],[-0.1,0.1],[-0.1,0.1],[0,200],[-0.1,0.1],[0,2*np.pi],[0,np.pi],[0,np.pi],[0,2*np.pi],[-jnp.pi/2,jnp.pi/2]])

from scipy.optimize import differential_evolution

f = jax.jit(LogLikelihood)
y = lambda x: -f(x)

result = differential_evolution(y, prior_range, maxiter=2000)