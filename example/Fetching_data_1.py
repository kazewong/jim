import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import signal, interpolate
import sxs
import json 

from tqdm import tqdm
from math import pi, log

import matplotlib.pyplot as plt
from numpy import abs

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
        for k in range(0, 10): 
            for l in range(1, 10):
                catalog_list.append(str(i)+str(j)+str(k)+str(l))


def save_dict_to_txt(data, filename):
    with open(filename, 'w') as f:
        for key, value in data.items():
            f.write(f"{key}: {value}\n")

theta_intrinsic_list = []
theta_extrinsic = [440, 0, 0]

M = 60.0
 
debug_path = '/mnt/home/averhaeghe/ceph/debugging/'

with tqdm(total=len(catalog_list)) as pbar:
    for i, catalog_number in enumerate(catalog_list):
        # Initialize frequency list and NR waveforms 
        f_start = 8
        f_end = 1536
        delta_f = 0.05


        waveform = sxs.load("SXS:BBH:"+str(catalog_number)+"/Lev/rhOverM", extrapolation_order=2, download=True)
        metadata = sxs.load("SXS:BBH:"+str(catalog_number)+"/Lev/metadata.json", download=None)
        waveform_l2_m2 = waveform.copy()
        waveform_l2_m2.data[...] = 0.0
        waveform_l2_m2.data[:, waveform_l2_m2.index(2, 2)] = waveform.data[:, waveform.index(2, 2)]
        waveform_l2_m2.data[:, waveform_l2_m2.index(2, -2)] = waveform.data[:, waveform.index(2, -2)]

        t_start = metadata.reference_time
        t_end = waveform_l2_m2.t[-1]
        dt = 1/4096
        t_uniform = np.arange(t_start, t_end, dt)
        h = waveform_l2_m2.interpolate(t_uniform)



        theta, phi = 0, 0
        h = h.evaluate(theta, phi)
        h = h.real



        h_scaled = (M / (theta_extrinsic[0] * m_per_Mpc)) * gt * C * h 
        t_scaled = t_uniform * gt * M
        dt = t_scaled[1] - t_scaled[0]


        ringdown_time = len(h_scaled) - int(np.where(np.abs(h_scaled) == np.max(np.abs(h_scaled)))[0])
        t_scaled -= t_scaled[len(h_scaled) - ringdown_time] 

        alpha = (2 * ringdown_time) / len(h_scaled)

        window = signal.windows.tukey(h_scaled.shape[0], alpha=alpha , sym=True) 
        h_scaled = h_scaled * window

        fig, ax = plt.subplots(1,1, dpi=300)
        #write code to extend t_scaled from -4 to 0, by adding 0.
        t = np.concatenate((np.arange(-2+t_scaled[0], t_scaled[0], dt), t_scaled)) 
        t = np.concatenate((t, np.arange(t_scaled[-1], 2+t_scaled[-1], dt))) 

        #do the same for h_scaled
        h = np.concatenate((np.zeros(len(np.arange(-2+t_scaled[0], t_scaled[0], dt))), h_scaled))
        h = np.concatenate((h, np.zeros(len(np.arange(t_scaled[-1], 2+t_scaled[-1], dt)))))
        t -= t[int(np.where(np.abs(h_scaled) == np.max(np.abs(h_scaled)))[0])] 
        t = t - 2
        h,t = np.array(h), np.array(t)
        h[:np.where(t<-2)[0][-1]] = 0
        h[np.where(t>0)[0][0]:] = 0
        h = h[:np.where(t>0)[0][0]]
        t = t[:np.where(t>0)[0][0]]
        ax.plot(t, h.real, label='real')
        plt.savefig(debug_path+'waveform_'+str(catalog_number)+'.png')

        t_scaled, h_scaled = t,h
        h_tilde = np.fft.rfft(h_scaled) * dt * np.exp(-2j * np.pi * t_scaled[0])
        freq = np.fft.rfftfreq(h_scaled.size, dt)

        NR = h_tilde
        if freq[-1]>1024:
            freq = freq[:np.where(freq>1024)[0][0]]
            NR = NR[:int(len(freq))]

        index = int(len(freq))
        start_index = round(0.02 * index)
        end_index = round(0.98 * index)

        f_uniform = freq[start_index:end_index]
        #print(freq, np.min(np.diff(freq)))
        NR_waveform = NR[start_index:end_index]
    
        plt.close()
        fig, ax = plt.subplots(1,4)
        ax[0].loglog(freq, NR.real, label='p')
        ax[1].loglog(freq, -NR.imag, label='c')
        ax[2].loglog(freq, np.abs(NR), label='abs')
        ax[3].loglog(freq, np.unwrap(np.angle(NR)), label='angle')
        plt.savefig(debug_path+'waveform_fd_'+str(catalog_number)+'.png')
        plt.close()


        q = metadata.reference_mass_ratio
        chi1 = metadata.reference_dimensionless_spin1[2]
        chi2 = metadata.reference_dimensionless_spin2[2]

        
        df = pd.DataFrame([])
        df.insert(len(df.columns), "frequency", f_uniform)
        df.insert(len(df.columns), "NR_p", np.real(NR_waveform))
        df.insert(len(df.columns), "NR_c", -np.imag(NR_waveform))

        param = {}
        param["eta"] = q / (1 + q)**2
        param["chirp_mass"] = M * param["eta"]**(3/5)
        param["chi1"] = chi1
        param["chi2"] = chi2
        
        
        save_dict_to_txt(param, '/mnt/home/averhaeghe/ceph/NR_waveforms/param_'+str(catalog_number)+'.txt')
        np.savetxt('/mnt/home/averhaeghe/ceph/NR_waveforms/NR_'+str(catalog_number)+'.txt', df.values)
        #np.savetxt('/mnt/home/averhaeghe/ceph/NR_waveforms/param_'+str(catalog_number)+'.txt', param.values)


        pbar.update(1)
