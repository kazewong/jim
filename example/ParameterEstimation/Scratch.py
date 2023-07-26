from typing import Callable, Iterator, Tuple
import chex, ripple, jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd
from scipy import signal, interpolate
import sxs
import glob
from tqdm import tqdm

from ripple.waveforms.IMRPhenomD_utils import get_coeffs
from math import pi, log
from ripple.typing import Array
from scipy.optimize import minimize, minimize_scalar

from numpy import random, abs
from ripple.waveforms import IMRPhenomD, IMRPhenomD_utils
from ripple.waveforms.IMRPhenomD import *
from ripple import ms_to_Mc_eta, Mc_eta_to_ms
from jax import grad, vmap, scipy
from functools import partial
import time

"""
Various constants, all in SI units.
"""

EulerGamma = 0.577215664901532860606512090082402431

MSUN = 1.9884099021470415e30  # kg
"""Solar mass"""

G = 6.67430e-11  # m^3 / kg / s^2
"""Newton's gravitational constant"""

C = 299792458.0  # m / s
"""Speed of light"""

gt = G * MSUN / (C ** 3.0)
"""
G MSUN / C^3 in seconds
"""

m_per_Mpc = 3.085677581491367278913937957796471611e22
"""
Meters per Mpc.
"""

catalog_list = []
for i in range(1):
    for j in range(0, 1):
        for k in range(0, 10): # FIX ME: CHANGE THIS LATER
            for l in range(1, 10):
                catalog_list.append(str(i)+str(j)+str(k)+str(l))

"""
Might want to change theta_intrinsic_list to a dict
"""
catalog_list = [i for i in catalog_list if int(i[2:])>65]

theta_extrinsic = jnp.array([440.0, 0.0, 0.0])
theta_intrinsic_list = []
x_pos_eta = []
x_pos_q = []
y_pos = []

for catalog_number in catalog_list:
    metadata = sxs.load("SXS:BBH:"+str(catalog_number)+"/Lev/metadata.json", download=None)
    #time.sleep(1)
    q = round(metadata.reference_mass_ratio * 1000) / 1000
    M = 50.0
    chi1 = metadata.reference_dimensionless_spin1[2]
    chi2 = metadata.reference_dimensionless_spin2[2]
    
    eta = q / (1 + q) ** 2.
    chi_eff = (q * chi1 + chi2) / (1 + q)
    chi_PN = chi_eff - 38 * eta * (chi1 + chi2) / 113
    chi_hat = chi_PN / (1 - 76 * eta / 113)
    
    theta_intrinsic = jnp.array([M * q / (1 + q), M * 1 / (1 + q), chi1, chi2])
    theta_intrinsic_list.append(theta_intrinsic)
    x_pos_eta.append(eta)
    x_pos_q.append(q)
    y_pos.append(chi_eff)

df = pd.DataFrame(theta_intrinsic_list)
df.index = catalog_list
df.columns = ['m1', 'm2', 'chi1', 'chi2']


with tqdm(total=len(catalog_list)) as pbar:
    for i, catalog_number in enumerate(catalog_list):
        # Initialize frequency list and NR waveforms 
        f_start = 8
        f_end = 1024
        delta_f = 0.001
        fs = jnp.arange(f_start, f_end, delta_f)
        f_sep = 100

        waveform = sxs.load("SXS:BBH:"+str(catalog_number)+"/Lev/rhOverM", extrapolation_order=4, download=None)
        waveform_l2_m2 = waveform.copy()
        waveform_l2_m2.data[...] = 0.0
        waveform_l2_m2.data[:, waveform_l2_m2.index(2, 2)] = waveform.data[:, waveform.index(2, 2)]
        waveform_l2_m2.data[:, waveform_l2_m2.index(2, -2)] = waveform.data[:, waveform.index(2, -2)]
        t_start = 0.0
        t_end = waveform_l2_m2.t[-1]
        dt = np.min(np.diff(waveform_l2_m2.t))
        t_uniform = np.arange(t_start, t_end, dt)
        h = waveform_l2_m2.interpolate(t_uniform)

        theta, phi = 0, 0
        h = h.evaluate(theta, phi)
        h = h.real

        h_scaled = (M / (theta_extrinsic[0] * m_per_Mpc)) * gt * C * h 
        t_scaled = t_uniform * M * gt
        dt = t_scaled[1] - t_scaled[0]

        index = int(np.where(np.abs(h_scaled) == np.max(np.abs(h_scaled)))[0])
        ringdown_time = len(h_scaled) - index
        # len_new_t = (int(len(h_scaled) / float(ringdown_time)) - 1) * ringdown_time
        start_time = int(len(h_scaled) / 50)
        t_scaled = t_scaled[start_time:]
        h_scaled = h_scaled[start_time:]
        alpha = (4 * ringdown_time) / (2 * len(h_scaled))
        # alpha = 0.20
        print(alpha)

        window = signal.windows.tukey(h_scaled.shape[0], alpha=alpha , sym=True) 
        h_scaled = h_scaled * window

        h_tilde = np.fft.rfft(h_scaled) * dt
        freq = np.fft.rfftfreq(h_scaled.size, dt)

        tck_amp = interpolate.splrep(freq, np.abs(h_tilde), s=0)
        NR_amp = interpolate.splev(fs, tck_amp)
        tck_phase = interpolate.splrep(freq, -np.unwrap(np.angle(h_tilde)), s=0)
        NR_phase = interpolate.splev(fs, tck_phase)

        NR = NR_amp * np.exp(-1j * NR_phase)
        IMR = IMRPhenomD._gen_IMRPhenomD(fs, theta_intrinsic_list[i],theta_extrinsic, 10, get_coeffs(theta_intrinsic_list[i]))
        NR_phase = -np.unwrap(np.angle(NR))
        NR_diff = (NR_phase[2:] - NR_phase[:-2]) / (2 * delta_f)

        tck_phase_diff = interpolate.splrep(fs[1:-1], NR_diff, s=0.1)
        NR_phase_diff = interpolate.splev(fs[1:-1], tck_phase_diff)

        for x in range(len(fs) - 3):
            if (NR_phase_diff[x+1] - NR_phase_diff[x]) < 0 and fs[x] * M * gt > 0.025:
                index = x
                break

        index = 36000
        start_index = round(0.05 * index)
        end_index = round(1.2 * index)
        f_uniform = fs[start_index:end_index]
        NR_waveform = NR[start_index:end_index]
        IMR_waveform = IMR[start_index:end_index]
        
        NR_phase = -jnp.unwrap(jnp.angle(NR_waveform))
        IMR_phase = -jnp.unwrap(jnp.angle(IMR_waveform))
        phase_diff = NR_phase - IMR_phase
    
        A = jnp.vstack([f_uniform, jnp.ones(len(f_uniform))]).T
        two_pi_t0, phi0 = jnp.linalg.lstsq(A, phase_diff, rcond=None)[0]

        NR_waveform = NR_waveform * jnp.exp(1j * (two_pi_t0 * f_uniform + phi0))
        
        # print(loss(PhenomD_coeff_table, theta_intrinsic_list[i], theta_extrinsic, f_uniform, NR_waveform))

#         plt.loglog(freq, jnp.abs(h_tilde), label='NR')
#         plt.loglog(f_uniform, jnp.abs(IMR_waveform), label='IMR')
#         plt.xlim(10, 500)
#         plt.ylim(1e-25, 2e-22)
#         plt.title('%s' % catalog_number)
#         plt.show()
        # plt.savefig('test.pdf')
        # plt.close()

        # plt.plot(f_uniform * M * gt, -jnp.unwrap(jnp.angle(NR_waveform)), label='NR')
        # plt.plot(f_uniform * M * gt, -jnp.unwrap(jnp.angle(IMR_waveform)), label='IMR')
        # plt.legend()
        # plt.show()

        df = pd.DataFrame([])
        df.insert(len(df.columns), "frequency", f_uniform)
        df.insert(len(df.columns), "NR_real", np.real(NR_waveform))
        df.insert(len(df.columns), "NR_imag", np.imag(NR_waveform))
        #np.savetxt('/mnt/home/averhaeghe/ceph/NR_waveform/NR_'+str(catalog_number)+'.txt', df.values)
        
        pbar.update(1)
        print('Run finished')

