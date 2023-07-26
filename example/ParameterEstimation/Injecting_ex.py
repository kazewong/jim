import jax.numpy as jnp 
import numpy as np
import jax
import ripple
import time

from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD
from jimgw.generate_noise_mod import *
from jimgw.jim import Jim
from jimgw.detector import H1, L1
from jimgw.likelihood import TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform


theta_extrinsic = jnp.array([440.0,0,0])
theta_intrinsic = jnp.array([0.5,0.5,9e-10,1.4e-10])

path = '/mnt/home/averhaeghe/ceph/NR_waveform/NR_0002.txt'
data = np.loadtxt(path)
freq = data[:,0]
NR = data[:,1] + 1j*data[:,2]

noise = generate_noise_freq(0,freqs = freq, f_min = freq[0])
noise_fd_H1 = noise[2]['H1']
signal_fd_H1 = NR + noise_fd_H1

# Time to do some likelihood estimation


###########################################
########## First we grab data #############
###########################################

#total_time_start = time.time()


ifos = ['H1', 'L1']

H1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

# Note that the likelihood is independentn of the last 3 arguments
likelihood = TransientLikelihoodFD([H1, L1], RippleIMRPhenomD(), gps, 4, 2)
prior = Uniform(
    xmin = [10, 0.125, -1., -1., 0., -0.05, 0., -1, 0., 0.,-1.],
    xmax = [80., 1., 1., 1., 2000., 0.05, 2*jnp.pi, 1., jnp.pi, 2*jnp.pi, 1.],
    naming = ["M_c", "q", "s1_z", "s2_z", "d_L", "t_c", "phase_c", "iota", "psi", "ra", "dec"],
    transforms = {"q": lambda q: q/(1+q)**2,
                 "iota": lambda iota: jnp.arccos(iota),
                 "dec": lambda dec: jnp.arcsin(dec)}
)

mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[1,1].set(1e-3)
mass_matrix = mass_matrix.at[5,5].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix*3e-3}

jim = Jim(likelihood, 
        prior,
        n_loop_training=10,
        n_loop_production = 10,
        n_local_steps=300,
        n_global_steps=300,
        n_chains=500,
        n_epochs=300,
        learning_rate = 0.001,
        momentum = 0.9,
        batch_size = 50000,
        use_global=True,
        keep_quantile=0.,
        train_thinning = 40,
        local_sampler_arg = local_sampler_arg,
        )

jim.maximize_likelihood([prior.xmin, prior.xmax])
jim.sample(jax.random.PRNGKey(42))