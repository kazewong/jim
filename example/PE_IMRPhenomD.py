from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1
from jimgw.likelihood import TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform
from ripple import ms_to_Mc_eta
import jax.numpy as jnp
import jax
from astropy.time import Time
from ripple.waveforms import IMRPhenomD
import matplotlib.pyplot as plt
from corner import corner
import numpy as np

from tap import Tap
import yaml
from tqdm import tqdm
import json

def read_complex_matrix_from_file(file_path):
    complex_matrix = []

    with open(file_path, 'r') as file:
        for line in file:
            row_elements = line.strip().split()  # Split the line by whitespace
            complex_row = [complex(element.replace('(', '').replace(')', '')) for element in row_elements]
            complex_matrix.append(complex_row)

    return jnp.array([item[0].real for item in complex_matrix]), jnp.array([item[1] for item in complex_matrix]), jnp.array([item[2] for item in complex_matrix])

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data




# opt = vars(args)
# yaml_var = yaml.load(open(opt['config'], 'r'), Loader=yaml.FullLoader)
# opt.update(yaml_var)

# Fetch noise parameters 

print("Constructing detectors")
print("Making noises")

#Fetch injection parameters and inject signal

print("Injection signals")

path_data = "/mnt/home/averhaeghe/ceph/NR_waveforms/NR_0001.txt"
path_param = "/mnt/home/averhaeghe/ceph/NR_waveforms/param_0001.txt"
freqs, hp, hc = read_complex_matrix_from_file(path_data)

NR = {}
NR['p'] = 
NR['c'] =   

intr_param =read_json_file(path_param)
Mc, eta, chi1, chi2 = intr_param["chirp_mass"], intr_param["eta"], intr_param["chi1"], intr_param["chi2"]

dist_mpc = 440
tc = 0.
phic = 0.2
inclination = 0.8
polarization_angle = 0.2
ra = 0.
dec = 0.
duration = 4
seed = 1234

f_ref = 30.0
trigger_time = 1126259462.4
post_trigger_duration = 2
epoch = duration - post_trigger_duration
gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad


waveform = RippleIMRPhenomD(f_ref=f_ref)
prior = Uniform(
    xmin = [10, 0.125, -1., -1., 0., -0.05, 0., -1, 0., 0.,-1.],
    xmax = [80., 1., 1., 1., 2000., 0.05, 2*jnp.pi, 1., jnp.pi, 2*jnp.pi, 1.],
    naming = ["M_c", "q", "s1_z", "s2_z", "d_L", "t_c", "phase_c", "cos_iota", "psi", "ra", "sin_dec"],
    transforms = {"q": ("eta", lambda q: q/(1+q)**2),
                 "cos_iota": ("iota",lambda cos_iota: jnp.arccos(jnp.arcsin(jnp.sin(cos_iota/2*jnp.pi))*2/jnp.pi)),
                 "sin_dec": ("dec",lambda sin_dec: jnp.arcsin(jnp.arcsin(jnp.sin(sin_dec/2*jnp.pi))*2/jnp.pi))} # sin and arcsin are periodize cos_iota and sin_dec
)
true_param = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle, ra, dec])


true_param = prior.add_name(true_param, with_transform=True)
detector_param = {"ra": ra, "dec": dec, "gmst": gmst, "psi": polarization_angle, "epoch": epoch, "t_c": tc}

h_sky = NR
key, subkey = jax.random.split(jax.random.PRNGKey(seed+1234))
H1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
L1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
V1.inject_signal(subkey, freqs, h_sky, detector_param)



likelihood = TransientLikelihoodFD([H1, L1], waveform, trigger_time, duration, post_trigger_duration)
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
        seed = seed,
        )


key, subkey = jax.random.split(jax.random.PRNGKey(1234))
jim.sample(subkey)
chains = jim.get_samples(training=True)
chains = np.array(chains)
figure = corner(chains.reshape(-1, 11), labels = prior.naming, max_n_ticks = 4, plot_datapoints=False, quantiles=(0.16, 0.5, 0.84), show_titles = True)
figure.set_size_inches(17, 17)
figure.suptitle("Visualize PE run", fontsize = 33)
plt.show(block=False)
plt.savefig("/mnt/home/averhaeghe/ceph/PE/new_coeffs/NR_0001.png")

