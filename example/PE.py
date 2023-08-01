import time
from jimgw.jim import Jim
from jimgw.detector import GroundBased2G
from jimgw.likelihood import TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform
import jax.numpy as jnp
import jax
import numpy as np

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

DEG_TO_RAD = jnp.pi / 180.

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
start = gps - 2
end = gps + 2
fmin = 20.
fmax = 1024.

ifos = ['H1', 'L1']

path = "mnt/home/averhaeghe/ceph/NR_waveforms_noisy/NR_0001.txt"
data = np.genfromtxt(path)
freqs = data[:,0]
signal = data[:,1]


H1 = GroundBased2G('H1', freqs, signal,
latitude = (46 + 27. / 60 + 18.528 / 3600) * DEG_TO_RAD,
longitude = -(119 + 24. / 60 + 27.5657 / 3600) * DEG_TO_RAD,
xarm_azimuth = 125.9994 * DEG_TO_RAD,
yarm_azimuth = 215.9994 * DEG_TO_RAD,
xarm_tilt = -6.195e-4,
yarm_tilt = 1.25e-5,
elevation = 142.554,
mode='pc')

likelihood = TransientLikelihoodFD([H1], RippleIMRPhenomD(), gps, 4, 2)
prior = Uniform(
    xmin = [10, 0.125, -1., -1., 0., -0.05, 0., -1, 0., 0.,-1.],
    xmax = [80., 1., 1., 1., 2000., 0.05, 2*jnp.pi, 1., jnp.pi, 2*jnp.pi, 1.],
    naming = ["M_c", "q", "s1_z", "s2_z", "d_L", "t_c", "phase_c", "cos_iota", "psi", "ra", "sin_dec"],
    transforms = {"q": lambda q: q/(1+q)**2,
                 "iota": lambda iota: jnp.arccos(jnp.arcsin(jnp.sin(iota/2*jnp.pi))*2/jnp.pi),
                 "dec": lambda dec: jnp.arcsin(jnp.arcsin(jnp.sin(dec/2*jnp.pi))*2/jnp.pi)} # sin and arcsin are periodize cos_iota and sin_dec
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



