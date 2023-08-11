import jax.numpy as jnp 
import numpy as np
import jax


from jimgw.generate_noise_mod import generate_noise_freq as noise_gen
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD
from jimgw.generate_noise_mod import *
from jimgw.jim import Jim
from jimgw.likelihood import TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform
from jimgw.detector import H1, L1

#   ASSERT THAT FREQ IS BIGGER THAN 30
theta_extrinsic = jnp.array([440.0,0,0])
theta_intrinsic = jnp.array([0.5,0.5,9e-10,1.4e-10])
ifos = 'H1'

path = '/mnt/home/averhaeghe/ceph/NR_waveforms/NR_0002.txt'
data = np.genfromtxt(path)
freq = data[:,0]
NR = data[:,1] + 1j*data[:,2]
NR = {}
NR['p'] = data[:,1]
NR['c'] = data[:,2]
param = {'ra':0,'dec':0, 'psi':0, 'gmst': 1126259462}

# Time to do some likelihood estimation

#total_time_start = time.time()
f = jax.jit(H1.fd_response)
#noise_gen = jax.jit(generate_noise_freq)
noise = noise_gen(0, freqs = freq)
noise_fd_H1 = noise[2]['H1']

signal_fd_H1 = f(freq, NR, param) + noise_fd_H1

