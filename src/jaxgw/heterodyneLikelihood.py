import wave
import numpy as np
from scipy.interpolate import interp1d

import jax.numpy as jnp

def max_phase_diff(f, f_low, f_high, chi=1):
    gamma = np.arange(-5,6,1)/3.
    f = np.repeat(f[:,None],len(gamma),axis=1)
    f_star = np.repeat(f_low, len(gamma))
    f_star[gamma >= 0] = f_high
    return 2*np.pi*chi*np.sum((f/f_star)**gamma*np.sign(gamma),axis=1)


def make_binning_scheme(freqs, n_bins, chi=1):
    phase_diff_array = max_phase_diff(freqs,freqs[0],freqs[-1],chi=1)
    bin_f = interp1d(phase_diff_array, freqs)
    f_bins = np.array([])
    for i in np.linspace(phase_diff_array[0], phase_diff_array[-1], n_bins):
        f_bins = np.append(f_bins,bin_f(i))
    f_bins_center = (f_bins[:-1] + f_bins[1:])/2
    return f_bins, f_bins_center

def compute_coefficients(data, h_ref, psd, freqs, f_bins, f_bins_center):
    A0_array = []
    A1_array = []
    B0_array = []
    B1_array = []

    df = freqs[1] - freqs[0]
    data_prod = np.array(data*h_ref.conj())
    self_prod = np.array(h_ref*h_ref.conj())
    for i in range(len(f_bins)-1):
        f_index = np.where((freqs >= f_bins[i]) & (freqs < f_bins[i+1]))[0]
        A0_array.append(4*np.sum(data_prod[f_index]/psd[f_index])*df)
        A1_array.append(4*np.sum(data_prod[f_index]/psd[f_index]*(freqs[f_index]-f_bins_center[i]))*df)
        B0_array.append(4*np.sum(self_prod[f_index]/psd[f_index])*df)
        B1_array.append(4*np.sum(self_prod[f_index]/psd[f_index]*(freqs[f_index]-f_bins_center[i]))*df)

    A0_array = jnp.array(A0_array)
    A1_array = jnp.array(A1_array)
    B0_array = jnp.array(B0_array)
    B1_array = jnp.array(B1_array)
    return A0_array, A1_array, B0_array, B1_array

def make_heterodyned_likelihood_multiple_detectors(data_list, psd_list,
    response_list, h_function, ref_theta, freqs, gmst, epoch, f_ref, n_bins=101):

    num_detector = len(data_list)
    theta_waveform = ref_theta
    theta_waveform = theta_waveform.at[5].set(0)
    raw_hp, raw_hc = h_function(freqs, theta_waveform, f_ref)
    index = jnp.where((jnp.abs(raw_hc)+jnp.abs(raw_hp)) > 0)
    freqs = freqs[index]
    raw_hp = raw_hp[index]
    raw_hc = raw_hc[index]
    for i in range(num_detector):
        data_list[i] = data_list[i][index]
        psd_list[i] = psd_list[i][index]

    f_bins, f_bins_center = make_binning_scheme(freqs, n_bins)
    ra, dec = ref_theta[9], ref_theta[10]
    h_ref = []
    h_ref_low = []
    h_ref_bincenter = []
    raw_hp_bin, raw_hc_bin = h_function(f_bins[:-1], theta_waveform, f_ref)
    raw_hp_bincenter, raw_hc_bincenter = h_function(f_bins_center, theta_waveform, f_ref)
    for i in range(num_detector):
        h_ref.append(response_list[i](freqs, raw_hp, raw_hc, ra, dec, gmst, ref_theta[8])*jnp.exp(-1j*2*jnp.pi*freqs*(epoch+ref_theta[5])))
        h_ref_low.append(response_list[i](f_bins[:-1], raw_hp_bin, raw_hc_bin, ra, dec, gmst, ref_theta[8])*jnp.exp(-1j*2*jnp.pi*f_bins[:-1]*(epoch+ref_theta[5])))
        h_ref_bincenter.append(response_list[i](f_bins_center, raw_hp_bincenter, raw_hc_bincenter, ra, dec, gmst, ref_theta[8])*jnp.exp(-1j*2*jnp.pi*f_bins_center*(epoch+ref_theta[5])))
    
    A0_array = []
    A1_array = []
    B0_array = []
    B1_array = []

    for i in range(num_detector):
        A0, A1, B0, B1 = compute_coefficients(data_list[i], h_ref[i], psd_list[i], freqs, f_bins, f_bins_center)
        A0_array.append(A0)
        A1_array.append(A1)
        B0_array.append(B0)
        B1_array.append(B1)
        
        
    def heterodyned_likelihood(params):
        theta_waveform = params
        theta_waveform = theta_waveform.at[5].set(0)
        ra, dec = params[9], params[10]

        output_SNR = 0

        raw_hp_edge, raw_hc_edge = h_function(f_bins[:-1], theta_waveform, f_ref)
        raw_hp_center, raw_hc_center = h_function(f_bins_center, theta_waveform, f_ref)

        for i in range(num_detector):
            waveform_low = response_list[i](f_bins[:-1], raw_hp_edge, raw_hc_edge, ra, dec, gmst, params[8])*jnp.exp(-1j*2*jnp.pi*f_bins[:-1]*(epoch+params[5]))
            waveform_center = response_list[i](f_bins_center, raw_hp_center, raw_hc_center, ra, dec, gmst, params[8])*jnp.exp(-1j*2*jnp.pi*f_bins_center*(epoch+params[5]))

            r0 = waveform_center/h_ref_bincenter[i]
            r1 = (waveform_low/h_ref_low[i] - r0)/(f_bins[:-1]-f_bins_center)
            match_filter_SNR = jnp.sum(A0_array[i]*r0.conj() + A1_array[i]*r1.conj())
            optimal_SNR = jnp.sum(B0_array[i]*jnp.abs(r0)**2 + 2*B1_array[i]*(r0*r1.conj()).real)

            output_SNR += (match_filter_SNR - optimal_SNR/2).real
        
        return output_SNR
    
    return heterodyned_likelihood
