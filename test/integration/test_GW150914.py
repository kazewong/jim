import time

import jax
import jax.numpy as jnp

from jimgw.jim import Jim
from jimgw.prior import CombinePrior, UniformPrior, CosinePrior, SinePrior
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomD
from jimgw.transforms import BoundToUnbound, MassRatioToSymmetricMassRatioTransform
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

Mc_prior = UniformPrior(10.0, 80.0, parameter_names=["M_c"])
q_prior = UniformPrior(
    0.125,
    1,
    parameter_names=["q"],
)
s1z_prior = UniformPrior(-1.0, 1.0, parameter_names=["s1_z"])
s2z_prior = UniformPrior(-1.0, 1.0, parameter_names=["s2_z"])
# Current likelihood sampling will fail and give nan because of large number
dL_prior = UniformPrior(0.0, 2000.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.05, 0.05, parameter_names=["t_c"])
phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
iota_prior = SinePrior(parameter_names=["iota"])
psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
dec_prior = CosinePrior(parameter_names=["dec"])

prior = CombinePrior(
    [
        Mc_prior,
        q_prior,
        s1z_prior,
        s2z_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        iota_prior,
        psi_prior,
        ra_prior,
        dec_prior,
    ]
)

sample_transforms = [
    BoundToUnbound(name_mapping = [["M_c"], ["M_c_unbounded"]], original_lower_bound=10.0, original_upper_bound=80.0),
    BoundToUnbound(name_mapping = [["q"], ["q_unbounded"]], original_lower_bound=0.125, original_upper_bound=1),
    BoundToUnbound(name_mapping = [["s1_z"], ["s1_z_unbounded"]] , original_lower_bound=-1.0, original_upper_bound=1.0),
    BoundToUnbound(name_mapping = [["s2_z"], ["s2_z_unbounded"]] , original_lower_bound=-1.0, original_upper_bound=1.0),
    BoundToUnbound(name_mapping = [["d_L"], ["d_L_unbounded"]] , original_lower_bound=0.0, original_upper_bound=2000.0),
    BoundToUnbound(name_mapping = [["t_c"], ["t_c_unbounded"]] , original_lower_bound=-0.05, original_upper_bound=0.05),
    BoundToUnbound(name_mapping = [["phase_c"], ["phase_c_unbounded"]] , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = [["iota"], ["iota_unbounded"]], original_lower_bound=-jnp.pi/2, original_upper_bound=jnp.pi/2),
    BoundToUnbound(name_mapping = [["psi"], ["psi_unbounded"]], original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["ra"], ["ra_unbounded"]], original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = [["dec"], ["dec_unbounded"]],original_lower_bound=0.0, original_upper_bound=jnp.pi)
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform(name_mapping=[["q"], ["eta"]]),
]

likelihood = TransientLikelihoodFD(
    [H1, L1],
    waveform=RippleIMRPhenomD(),
    trigger_time=gps,
    duration=4,
    post_trigger_duration=2,
)


mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[5, 5].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 3e-3}

Adam_optimizer = optimization_Adam(n_steps=5, learning_rate=0.01, noise_level=1)

n_epochs = 2
n_loop_training = 1
learning_rate = 1e-4


jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    n_loop_training=n_loop_training,
    n_loop_production=1,
    n_local_steps=5,
    n_global_steps=5,
    n_chains=4,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    n_max_examples=30,
    n_flow_samples=100,
    momentum=0.9,
    batch_size=100,
    use_global=True,
    train_thinning=1,
    output_thinning=1,
    local_sampler_arg=local_sampler_arg,
    strategies=[Adam_optimizer, "default"],
)

jim.sample(jax.random.PRNGKey(42))
