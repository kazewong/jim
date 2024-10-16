import time

import jax
import jax.numpy as jnp

from jimgw.jim import Jim
from jimgw.jim import Jim
from jimgw.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    UniformSpherePrior,
)
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomPv2
from jimgw.transforms import BoundToUnbound
from jimgw.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
    UniformInComponentMassSecondaryMassTransform,
    ComponentMassesToChirpMassMassRatioTransform,
)
from jimgw.single_event.utils import Mc_q_to_m1_m2
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

ifos = [H1, L1]

for ifo in ifos:
    ifo.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

prior = []

# Mass prior
M_c_min, M_c_max = 10.0, 80.0
q_min, q_max = 0.125, 1.0
m1_prior = UniformPrior(15.0, 280.0, parameter_names=['m_1'])
m2_prior = UniformPrior(0.0, 1.0, parameter_names=['m_2_quantile'])

prior = prior + [m1_prior, m2_prior]

# Spin prior
s1_prior = UniformSpherePrior(parameter_names=["s1"])
s2_prior = UniformSpherePrior(parameter_names=["s2"])
iota_prior = SinePrior(parameter_names=["iota"])

prior = prior + [
    s1_prior,
    s2_prior,
    iota_prior,
]

# Extrinsic prior
dL_prior = PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.05, 0.05, parameter_names=["t_c"])
phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
dec_prior = CosinePrior(parameter_names=["dec"])

prior = prior + [
    dL_prior,
    t_c_prior,
    phase_c_prior,
    psi_prior,
    ra_prior,
    dec_prior,
]

prior = CombinePrior(prior)

# Defining Transforms

sample_transforms = [
    UniformInComponentMassSecondaryMassTransform(q_min=q_min, q_max=q_max, M_c_min=M_c_min, M_c_max=M_c_max, m_1_min=m1_prior.xmin, m_1_max=m1_prior.xmax),
    ComponentMassesToChirpMassMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform(gps_time=gps, ifos=ifos, dL_min=dL_prior.xmin, dL_max=dL_prior.xmax),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=ifos[0]),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(tc_min=t_c_prior.xmin, tc_max=t_c_prior.xmax, gps_time=gps, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
    BoundToUnbound(name_mapping = (["M_c"], ["M_c_unbounded"]), original_lower_bound=M_c_min, original_upper_bound=M_c_max),
    BoundToUnbound(name_mapping = (["q"], ["q_unbounded"]), original_lower_bound=q_min, original_upper_bound=q_max),
    BoundToUnbound(name_mapping = (["s1_phi"], ["s1_phi_unbounded"]) , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = (["s2_phi"], ["s2_phi_unbounded"]) , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = (["iota"], ["iota_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = (["s1_theta"], ["s1_theta_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = (["s2_theta"], ["s2_theta_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = (["s1_mag"], ["s1_mag_unbounded"]) , original_lower_bound=0.0, original_upper_bound=0.99),
    BoundToUnbound(name_mapping = (["s2_mag"], ["s2_mag_unbounded"]) , original_lower_bound=0.0, original_upper_bound=0.99),
    BoundToUnbound(name_mapping = (["phase_det"], ["phase_det_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = (["psi"], ["psi_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = (["zenith"], ["zenith_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = (["azimuth"], ["azimuth_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
]

likelihood_transforms = [
    UniformInComponentMassSecondaryMassTransform(q_min=q_min, q_max=q_max, M_c_min=M_c_min, M_c_max=M_c_max, m_1_min=m1_prior.xmin, m_1_max=m1_prior.xmax),
    ComponentMassesToChirpMassMassRatioTransform,
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform("s1"),
    SphereSpinToCartesianSpinTransform("s2"),
]

likelihood = TransientLikelihoodFD(
    ifos,
    waveform=RippleIMRPhenomPv2(),
    trigger_time=gps,
    duration=4,
    post_trigger_duration=2,
)


mass_matrix = jnp.eye(15)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[9, 9].set(1e-3)
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
jim.get_samples()
jim.print_summary()
