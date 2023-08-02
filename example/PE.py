import time
from jimgw.jim import Jim
from jimgw.detector import GroundBased2G
from jimgw.detector import H1 as H1_class
from jimgw.likelihood import TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform
import jax.numpy as jnp
import jax
import numpy as np
from scipy.interpolate import interp1d
jax.config.update('jax_enable_x64', True)
import corner
import matplotlib.pyplot as plt
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
key = 0

ifos = ['H1', 'L1']

# Now we load the data
def read_complex_matrix_from_file(file_path):
    complex_matrix = []

    with open(file_path, 'r') as file:
        for line in file:
            row_elements = line.strip().split()  # Split the line by whitespace
            complex_row = [complex(element.replace('(', '').replace(')', '')) for element in row_elements]
            complex_matrix.append(complex_row)

    return complex_matrix

path = "/mnt/home/averhaeghe/ceph/NR_waveforms_noisy/NR_0001.txt"
data = read_complex_matrix_from_file(path)
freqs = [item[0].real for item in data]
signal = [item[1] for item in data]

freqs = jnp.array([freq for freq in freqs if freqs.index(freq) %5 == 0])
signal = jnp.array([sig for sig in signal if signal.index(sig) %5 == 0])
psd = H1_class.load_psd(freqs)

print(freqs[1]-freqs[0])

H1 = GroundBased2G('H1',
latitude = (46 + 27. / 60 + 18.528 / 3600) * DEG_TO_RAD,
longitude = -(119 + 24. / 60 + 27.5657 / 3600) * DEG_TO_RAD,
xarm_azimuth = 125.9994 * DEG_TO_RAD,
yarm_azimuth = 215.9994 * DEG_TO_RAD,
xarm_tilt = -6.195e-4,
yarm_tilt = 1.25e-5,
elevation = 142.554,
mode='pc')

H1.set_data(freqs, signal, psd)

likelihood = TransientLikelihoodFD([H1], RippleIMRPhenomD(), gps, 4, 2)
prior = Uniform(
    xmin = [10, 0.125, -1., -1., 0., -0.05, 0., -1, 0., 0.,-1.],
    xmax = [80., 1., 1., 1., 2000., 0.05, 2*jnp.pi, 1., jnp.pi, 2*jnp.pi, 1.],
    naming = ["M_c", "q", "s1_z", "s2_z", "d_L", "t_c", "phase_c", "cos_iota", "psi", "ra", "sin_dec"],
    transforms = {"q": ("eta", lambda q: q/(1+q)**2),
                 "cos_iota": ("iota",lambda cos_iota: jnp.arccos(jnp.arcsin(jnp.sin(cos_iota/2*jnp.pi))*2/jnp.pi)),
                 "sin_dec": ("dec",lambda sin_dec: jnp.arcsin(jnp.arcsin(jnp.sin(sin_dec/2*jnp.pi))*2/jnp.pi))} # sin and arcsin are periodize cos_iota and sin_dec
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
        n_chains=200,
        n_epochs=300,
        learning_rate = 0.001,
        momentum = 0.9,
        batch_size = 50000,
        use_global=True,
        keep_quantile=0.,
        train_thinning = 40,
        local_sampler_arg = local_sampler_arg,
        )

#print(jim.maximize_likelihood((prior.xmin, prior.xmax)))

if True:
        key, subkey = jax.random.split(jax.random.PRNGKey(1234))
        jim.sample(subkey)
        chains = jim.get_samples(training=True)
        chains = np.array(chains)
        figure = corner.corner(
        chains.reshape(-1, 11), labels = prior.naming, max_n_ticks = 4, plot_datapoints=False, quantiles=(0.16, 0.5, 0.84), show_titles = True)
        figure.set_size_inches(17, 17)
        figure.suptitle("Visualize PE run", fontsize = 33)
        plt.show(block=False)
        plt.savefig("/mnt/home/averhaeghe/jim/example/Second_PE_run.png")


