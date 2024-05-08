import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"

import time

import jax
import jax.numpy as jnp
import optax

from jimgw.jim import Jim
from jimgw.prior import Composite, Unconstrained_Uniform
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import TransientLikelihoodFD
from jimgw.single_event.waveform import RippleTaylorF2
from flowMC.strategy.optimization import optimization_Adam

jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
duration = 4
post_trigger_duration = 2
start_pad = duration - post_trigger_duration
end_pad = post_trigger_duration
fmin = 20.0
fmax = 1024.0

ifos = ["H1", "L1"]

H1.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

Mc_prior = Unconstrained_Uniform(10.0, 80.0, naming=["M_c"])
q_prior = Unconstrained_Uniform(
    0.125,
    1.0,
    naming=["q"],
    transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
)
s1z_prior = Unconstrained_Uniform(-1.0, 1.0, naming=["s1_z"])
s2z_prior = Unconstrained_Uniform(-1.0, 1.0, naming=["s2_z"])
lambda1_prior = Unconstrained_Uniform(0.0, 5000.0, naming=["lambda_1"])
lambda2_prior = Unconstrained_Uniform(0.0, 5000.0, naming=["lambda_2"])
dL_prior = Unconstrained_Uniform(0.0, 2000.0, naming=["d_L"])
t_c_prior = Unconstrained_Uniform(-0.05, 0.05, naming=["t_c"])
phase_c_prior = Unconstrained_Uniform(0.0, 2 * jnp.pi, naming=["phase_c"])
cos_iota_prior = Unconstrained_Uniform(
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
psi_prior = Unconstrained_Uniform(0.0, jnp.pi, naming=["psi"])
ra_prior = Unconstrained_Uniform(0.0, 2 * jnp.pi, naming=["ra"])
sin_dec_prior = Unconstrained_Uniform(
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

prior = Composite(
    [
        Mc_prior,
        q_prior,
        s1z_prior,
        s2z_prior,
        lambda1_prior,
        lambda2_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        cos_iota_prior,
        psi_prior,
        ra_prior,
        sin_dec_prior,
    ]
)
likelihood = TransientLikelihoodFD(
    [H1, L1],
    waveform=RippleTaylorF2(),
    trigger_time=gps,
    duration=4,
    post_trigger_duration=2,
)

n_dim = 13
mass_matrix = jnp.eye(n_dim)
mass_matrix = mass_matrix.at[0,0].set(1e-5)
mass_matrix = mass_matrix.at[1,1].set(1e-4)
mass_matrix = mass_matrix.at[2,2].set(1e-3)
mass_matrix = mass_matrix.at[3,3].set(1e-3)
mass_matrix = mass_matrix.at[7,7].set(1e-5)
mass_matrix = mass_matrix.at[11,11].set(1e-2)
mass_matrix = mass_matrix.at[12,12].set(1e-2)
local_sampler_arg = {"step_size": mass_matrix * 1e-3}

# Build the learning rate scheduler

n_loop_training = 100
n_epochs = 100
total_epochs = n_epochs * n_loop_training
start = int(total_epochs / 10)
start_lr = 1e-3
end_lr = 1e-5
power = 4.0
schedule_fn = optax.polynomial_schedule(
    start_lr, end_lr, power, total_epochs-start, transition_begin=start)

jim = Jim(
    likelihood,
    prior,
    n_loop_training=n_loop_training,
    n_loop_production=20,
    n_local_steps=10,
    n_global_steps=1000,
    n_chains=1000,
    n_epochs=n_epochs,
    learning_rate=schedule_fn,
    n_max_examples=30000,
    n_flow_samples=100000,
    momentum=0.9,
    batch_size=30000,
    use_global=True,
    train_thinning=20,
    output_thinning=50,
    local_sampler_arg=local_sampler_arg,
)

jim.sample(jax.random.PRNGKey(24))
jim.print_summary()
