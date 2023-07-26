import jax.numpy as jnp 
import numpy as np
import jax
import ripple
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD
from jimgw.generate_noise_mod import *

theta_extrinsic = jnp.array([440.0,0,0])
theta_intrinsic = jnp.array([0.5,0.5,9e-10,1.4e-10])

path = '/mnt/home/averhaeghe/ceph/NR_waveform/NR_0002.txt'
data = np.loadtxt(path)
freq = data[:,0]
NR = data[:,1] + 1j*data[:,2]

noise = generate_noise_freq(0,freqs = freq, f_min = freq[0])
noise_fd_H1 = noise[2]['H1']
signal_fd = NR + noise_fd_H1