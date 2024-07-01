import time

import jax
import jax.numpy as jnp

from jimgw.jim import Jim
from jimgw.prior import Composite, Sphere, Unconstrained_Uniform
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomPv2
from flowMC.strategy.optimization import optimization_Adam


jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
start = gps - 2
end = gps + 2
fmin = 20.0
fmax = 1024.0

ifos = ["H1", "L1"]

H1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

waveform = RippleIMRPhenomPv2(f_ref=20)

###########################################
########## Set up priors ##################
###########################################

Mc_prior = Unconstrained_Uniform(10.0, 80.0, naming=["M_c"])
q_prior = Unconstrained_Uniform(
    0.125,
    1.0,
    naming=["q"],
    transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
)
s1_prior = Sphere(naming="s1")
s2_prior = Sphere(naming="s2")
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
        s1_prior,
        s2_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        cos_iota_prior,
        psi_prior,
        ra_prior,
        sin_dec_prior,
    ],
)

epsilon = 1e-3
bounds = jnp.array(
    [
        [10.0, 80.0],
        [0.125, 1.0],
        [0, jnp.pi],
        [0, 2 * jnp.pi],
        [0.0, 1.0],
        [0, jnp.pi],
        [0, 2 * jnp.pi],
        [0.0, 1.0],
        [0.0, 2000],
        [-0.05, 0.05],
        [0.0, 2 * jnp.pi],
        [-1.0, 1.0],
        [0.0, jnp.pi],
        [0.0, 2 * jnp.pi],
        [-1.0, 1.0],
    ]
) + jnp.array([[epsilon, -epsilon]])

likelihood = TransientLikelihoodFD(
    [H1, L1], waveform=waveform, trigger_time=gps, duration=4, post_trigger_duration=2
)
# likelihood = HeterodynedTransientLikelihoodFD([H1, L1], prior=prior, bounds=bounds, waveform=waveform, trigger_time=gps, duration=4, post_trigger_duration=2)


mass_matrix = jnp.eye(prior.n_dim)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[9, 9].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 1e-3}

Adam_optimizer = optimization_Adam(n_steps=3000, learning_rate=0.01, noise_level=1, bounds=bounds)

import optax
n_epochs = 20
n_loop_training = 100
total_epochs = n_epochs * n_loop_training
start = total_epochs//10
learning_rate = optax.polynomial_schedule(
    1e-3, 1e-4, 4.0, total_epochs - start, transition_begin=start
)

jim = Jim(
    likelihood,
    prior,
    n_loop_training=n_loop_training,
    n_loop_production=20,
    n_local_steps=10,
    n_global_steps=1000,
    n_chains=500,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    n_max_examples=30000,
    n_flow_sample=100000,
    momentum=0.9,
    batch_size=30000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=1,
    output_thinning=10,
    local_sampler_arg=local_sampler_arg,
    # strategies=[Adam_optimizer,"default"],
)

import numpy as np
# chains = np.load('./GW150914_init.npz')['chain']

jim.sample(jax.random.PRNGKey(42))#,initial_guess=chains)
