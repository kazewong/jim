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
jax.config.update('jax_enable_x64', True)

from corner import corner
import numpy as np
import matplotlib.pyplot as plt
from tap import Tap
import yaml
from tqdm import tqdm

class InjectionRecoveryParser(Tap):
    config: str 
    
    # Noise parameters
    seed: int = None
    f_sampling: int  = None
    duration: int = None
    fmin: float = None
    ifos: list[str]  = None

    # Injection parameters
    m1: float = None
    m2: float = None
    chi1: float = None
    chi2: float = None
    dist_mpc: float = None
    tc: float = None
    phic: float = None
    inclination: float = None
    polarization_angle: float = None
    ra: float = None
    dec: float = None

    # Sampler parameters
    n_dim: int = None
    n_chains: int = None
    n_loop_training: int = None
    n_loop_production: int = None
    n_local_steps: int = None
    n_global_steps: int = None
    learning_rate: float = None
    max_samples: int = None
    momentum: float = None
    num_epochs: int = None
    batch_size: int = None
    stepsize: float = None

    # Output parameters
    output_path: str = None
    downsample_factor: int = None


args = InjectionRecoveryParser().parse_args()



# Fetch noise parameters 

print("Constructing detectors")
print("Making noises")

#Fetch injection parameters and inject signal

print("Injection signals")

def read_complex_matrix_from_file(file_path):
    complex_matrix = []

    with open(file_path, 'r') as file:
        for line in file:
            row_elements = line.strip().split()  # Split the line by whitespace
            complex_row = [complex(element.replace('(', '').replace(')', '')) for element in row_elements]
            complex_matrix.append(complex_row)

    return jnp.array([item[0].real for item in complex_matrix]), jnp.array([item[1] for item in complex_matrix]), jnp.array([item[2] for item in complex_matrix])

path_data = "/mnt/home/averhaeghe/ceph/NR_waveforms/NR_0001.txt"
path_param = "/mnt/home/averhaeghe/ceph/NR_waveforms/param_0001.txt"
freqs, hp, hc = read_complex_matrix_from_file(path_data)

NR = {}
NR['p'] = hp 
NR['c'] =  hc 




#freqs = jnp.linspace(args.fmin, args.f_sampling/2, args.duration*args.f_sampling//2)

Mc, eta = ms_to_Mc_eta(jnp.array([args.m1, args.m2]))
f_ref = 30.0
trigger_time = 1126259462.4
post_trigger_duration = 2
epoch = args.duration - post_trigger_duration
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
true_param = jnp.array([Mc, eta, args.chi1, args.chi2, args.dist_mpc, args.tc, args.phic, args.inclination, args.polarization_angle, args.ra, args.dec])

#hp, hc = IMRPhenomD.gen_IMRPhenomD_polar(freqs, true_param, f_ref)

NR = {}
NR['p'] = hp 
NR['c'] =  hc 

true_param = prior.add_name(true_param, with_transform=True)
detector_param = {"ra": args.ra, "dec": args.dec, "gmst": gmst, "psi": args.polarization_angle, "epoch": epoch, "t_c": args.tc}

h_sky = NR
key, subkey = jax.random.split(jax.random.PRNGKey(args.seed+1234))
H1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
L1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
V1.inject_signal(subkey, freqs, h_sky, detector_param)



likelihood = TransientLikelihoodFD([H1, L1], waveform, trigger_time, args.duration, post_trigger_duration)
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
        seed = args.seed,
        )

sample = jim.maximize_likelihood([prior.xmin, prior.xmax], n_loops=2000)
key, subkey = jax.random.split(key)
jim.sample(subkey)
samples = jim.get_samples()