import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.single_event.waveform import RippleTaylorF2
from jimgw.prior import Uniform, Composite 
import jax.numpy as jnp
import jax
import time
jax.config.update("jax_enable_x64", True)
import numpy as np
import optax 
from gwosc.datasets import event_gps
print(f"GPU found? {jax.devices()}")


data_path = "/home/thibeau.wouters/gw-datasets/GW170817/" # on CIT

start_runtime = time.time()

############
### BODY ###
############

### Data definitions

total_time_start = time.time()
gps = 1187008882.43
trigger_time = gps
fmin = 20
fmax = 2048
minimum_frequency = fmin
maximum_frequency = fmax
duration = 128
# epoch = duration - post_trigger_duration
post_trigger_duration = 32
start_pad = duration - post_trigger_duration
end_pad = post_trigger_duration
f_ref = fmin 
tukey_alpha = 2 / (duration / 2)

ifos = ["H1", "L1"]#, "V1"]

H1.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=4*duration, tukey_alpha=tukey_alpha, gwpy_kwargs={"version": 2, "cache": False})
L1.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=4*duration, tukey_alpha=tukey_alpha, gwpy_kwargs={"version": 2, "cache": False})
# V1.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=16, tukey_alpha=0.05)

### Define priors

# Internal parameters
Mc_prior = Uniform(1.18, 1.21, naming=["M_c"])
q_prior = Uniform(
    0.125,
    1.0,
    naming=["q"],
    transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
)
s1z_prior = Uniform(-0.05, 0.05, naming=["s1_z"])
s2z_prior = Uniform(-0.05, 0.05, naming=["s2_z"])
lambda_1_prior = Uniform(0.0, 5000.0, naming=["lambda_1"])
lambda_2_prior = Uniform(0.0, 5000.0, naming=["lambda_2"])
dL_prior       = Uniform(1.0, 75.0, naming=["d_L"])
# dL_prior       = PowerLaw(1.0, 75.0, 2.0, naming=["d_L"])
t_c_prior      = Uniform(-0.1, 0.1, naming=["t_c"])
phase_c_prior  = Uniform(0.0, 2 * jnp.pi, naming=["phase_c"])
cos_iota_prior = Uniform(
    -1.0,
    1.0,
    naming=["cos_iota"],
    transforms={
        "cos_iota": (
            "iota",
            lambda params: jnp.arccos(
                jnp.arcsin(jnp.sin(params["cos_iota"] / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        )
    },
)
psi_prior     = Uniform(0.0, jnp.pi, naming=["psi"])
ra_prior      = Uniform(0.0, 2 * jnp.pi, naming=["ra"])
sin_dec_prior = Uniform(
    -1.0,
    1.0,
    naming=["sin_dec"],
    transforms={
        "sin_dec": (
            "dec",
            lambda params: jnp.arcsin(
                jnp.arcsin(jnp.sin(params["sin_dec"] / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        )
    },
)

prior_list = [
        Mc_prior,
        q_prior,
        s1z_prior,
        s2z_prior,
        lambda_1_prior,
        lambda_2_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        cos_iota_prior,
        psi_prior,
        ra_prior,
        sin_dec_prior,
    ]

prior = Composite(prior_list)

# The following only works if every prior has xmin and xmax property, which is OK for Uniform and Powerlaw
bounds = jnp.array([[p.xmin, p.xmax] for p in prior.priors])

### Create likelihood object

ref_params = {
    'M_c': 1.19793583,
    'eta': 0.24794374,
    's1_z': 0.00220637,
    's2_z': 0.0499,
    'lambda_1': 605.12916663,
    'lambda_2': 405.12916663,
    'd_L': 45.41592353,
    't_c': 0.00220588,
    'phase_c': 5.76822606,
    'iota': 2.46158044,
    'psi': 2.09118099,
    'ra': 5.03335133,
    'dec': 0.01679998
}

n_bins = 100

likelihood = HeterodynedTransientLikelihoodFD([H1, L1], prior=prior, bounds=bounds, waveform=RippleTaylorF2(f_ref=f_ref), trigger_time=gps, duration=duration, n_bins=n_bins, ref_params=ref_params)
print("Running with n_bins  = ", n_bins)

# Local sampler args

eps = 1e-3
n_dim = 13
mass_matrix = jnp.eye(n_dim)
mass_matrix = mass_matrix.at[0,0].set(1e-5)
mass_matrix = mass_matrix.at[1,1].set(1e-4)
mass_matrix = mass_matrix.at[2,2].set(1e-3)
mass_matrix = mass_matrix.at[3,3].set(1e-3)
mass_matrix = mass_matrix.at[7,7].set(1e-5)
mass_matrix = mass_matrix.at[11,11].set(1e-2)
mass_matrix = mass_matrix.at[12,12].set(1e-2)
local_sampler_arg = {"step_size": mass_matrix * eps}

# Build the learning rate scheduler

n_loop_training = 200
n_epochs = 50
total_epochs = n_epochs * n_loop_training
start = int(total_epochs / 10)
start_lr = 1e-3
end_lr = 1e-5
power = 4.0
schedule_fn = optax.polynomial_schedule(
    start_lr, end_lr, power, total_epochs-start, transition_begin=start)

scheduler_str = f"polynomial_schedule({start_lr}, {end_lr}, {power}, {total_epochs-start}, {start})"

# Create jim object

outdir_name = "./outdir/"

jim = Jim(
    likelihood,
    prior,
    n_loop_training=n_loop_training,
    n_loop_production=20,
    n_local_steps=10,
    n_global_steps=500,
    n_chains=1000,
    n_epochs=n_epochs,
    learning_rate=schedule_fn,
    max_samples=50000,
    momentum=0.9,
    batch_size=50000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=10,
    output_thinning=30,    
    local_sampler_arg=local_sampler_arg,
    stopping_criterion_global_acc = 0.20,
    outdir_name=outdir_name
)

### Sample and show results

jim.sample(jax.random.PRNGKey(41))
jim.print_summary()