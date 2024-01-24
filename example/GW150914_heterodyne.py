import time
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import (
    HeterodynedTransientLikelihoodFD,
    TransientLikelihoodFD,
)
from jimgw.single_event.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform, Composite
import jax.numpy as jnp
import jax

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

Mc_prior = Uniform(10.0, 80.0, naming=["M_c"])
q_prior = Uniform(
    0.125,
    1.0,
    naming=["q"],
    transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
)
s1z_prior = Uniform(-1.0, 1.0, naming=["s1_z"])
s2z_prior = Uniform(-1.0, 1.0, naming=["s2_z"])
dL_prior = Uniform(0.0, 2000.0, naming=["d_L"])
t_c_prior = Uniform(-0.05, 0.05, naming=["t_c"])
phase_c_prior = Uniform(0.0, 2 * jnp.pi, naming=["phase_c"])
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
psi_prior = Uniform(0.0, jnp.pi, naming=["psi"])
ra_prior = Uniform(0.0, 2 * jnp.pi, naming=["ra"])
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

prior = Composite(
    [
        Mc_prior,
        q_prior,
        s1z_prior,
        s2z_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        cos_iota_prior,
        psi_prior,
        ra_prior,
        sin_dec_prior,
    ]
)

bounds = jnp.array(
    [
        [10.0, 80.0],
        [0.125, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [0.0, 2000.0],
        [-0.05, 0.05],
        [0.0, 2 * jnp.pi],
        [-1.0, 1.0],
        [0.0, jnp.pi],
        [0.0, 2 * jnp.pi],
        [-1.0, 1.0],
    ]
)

likelihood = HeterodynedTransientLikelihoodFD(
    [H1, L1],
    prior=prior,
    bounds=bounds,
    waveform=RippleIMRPhenomD(),
    trigger_time=gps,
    duration=duration,
    post_trigger_duration=post_trigger_duration,
    n_loops=300,
)

mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[5, 5].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 3e-3}

jim = Jim(
    likelihood,
    prior,
    n_loop_training=100,
    n_loop_production=10,
    n_local_steps=150,
    n_global_steps=150,
    n_chains=500,
    n_epochs=50,
    learning_rate=0.001,
    max_samples=45000,
    momentum=0.9,
    batch_size=50000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=1,
    output_thinning=10,
    local_sampler_arg=local_sampler_arg,
)

jim.sample(jax.random.PRNGKey(42))
