import time

import jax
import jax.numpy as jnp

from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    UniformSpherePrior,
)
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2
from jimgw.core.transforms import BoundToUnbound
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
)
from flowMC.strategy.optimization import optimization_Adam

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

# first, fetch a 128s segment centered on GW170817
# for the analysis
gps = 1187008882.43
duration = 128.0
# Request a segment with 2.0 s post-merger
start = gps + 2.0 - duration
end = start + duration

# fetch 8192s of data to estimate the PSD (to be
# careful we should avoid the on-source segment,
# but we don't do this in this example)
psd_start = gps - 4096
psd_end = gps + 4096

fmin = minimum_frequency = 20
fmax = maximum_frequency = 2048
f_ref = fmin

# initialize detectors
ifos = [get_H1(), get_L1(), get_V1()]

for ifo in ifos:
    # set analysis data
    strain_data = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(strain_data)

    # set PSD (Welch estimate)
    psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
    # set an NFFT corresponding to the analysis segment duration
    psd_fftlength = strain_data.duration * strain_data.sampling_frequency
    ifo.set_psd(psd_data.to_psd(nperseg=psd_fftlength))

###########################################
########## Set up waveform ################
###########################################

# initialize waveform
waveform = RippleIMRPhenomPv2(f_ref=20)

###########################################
########## Set up priors ##################
###########################################

prior = []

# Mass prior
M_c_min, M_c_max = 1.18, 1.21
q_min, q_max = 0.125, 1.0
Mc_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])

prior = prior + [Mc_prior, q_prior]

# Spin prior
s1_prior = UniformSpherePrior(parameter_names=["s1"], max_mag=0.05)
s2_prior = UniformSpherePrior(parameter_names=["s2"], max_mag=0.05)
iota_prior = SinePrior(parameter_names=["iota"])

prior = prior + [
    s1_prior,
    s2_prior,
    iota_prior,
]

# Extrinsic prior
dL_prior = PowerLawPrior(1.0, 75.0, 2.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.1, 0.1, parameter_names=["t_c"])
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
    DistanceToSNRWeightedDistanceTransform(
        gps_time=gps, ifos=ifos, dL_min=dL_prior.xmin, dL_max=dL_prior.xmax
    ),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=ifos[0]),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(
        tc_min=t_c_prior.xmin, tc_max=t_c_prior.xmax, gps_time=gps, ifo=ifos[0]
    ),
    SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
    BoundToUnbound(
        name_mapping=(["M_c"], ["M_c_unbounded"]),
        original_lower_bound=M_c_min,
        original_upper_bound=M_c_max,
    ),
    BoundToUnbound(
        name_mapping=(["q"], ["q_unbounded"]),
        original_lower_bound=q_min,
        original_upper_bound=q_max,
    ),
    BoundToUnbound(
        name_mapping=(["s1_phi"], ["s1_phi_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s2_phi"], ["s2_phi_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["iota"], ["iota_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s1_theta"], ["s1_theta_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s2_theta"], ["s2_theta_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s1_mag"], ["s1_mag_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=0.05,
    ),
    BoundToUnbound(
        name_mapping=(["s2_mag"], ["s2_mag_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=0.05,
    ),
    BoundToUnbound(
        name_mapping=(["phase_det"], ["phase_det_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["psi"], ["psi_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["zenith"], ["zenith_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["azimuth"], ["azimuth_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform("s1"),
    SphereSpinToCartesianSpinTransform("s2"),
]


likelihood = HeterodynedTransientLikelihoodFD(
    ifos,
    waveform=waveform,
    n_bins=1000,
    trigger_time=gps,
    prior=prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    popsize=10,
    n_steps=50,
)

mass_matrix = jnp.eye(prior.n_dim)
# mass_matrix = mass_matrix.at[1, 1].set(1e-3)
# mass_matrix = mass_matrix.at[9, 9].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 1e-3}

#### The rest of this script is not guaranteed to work ####

Adam_optimizer = optimization_Adam(n_steps=3000, learning_rate=0.01, noise_level=1)

import optax

n_epochs = 20
n_loop_training = 100
total_epochs = n_epochs * n_loop_training
start = total_epochs // 10
learning_rate = optax.polynomial_schedule(
    1e-3, 1e-4, 4.0, total_epochs - start, transition_begin=start
)

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    n_loop_training=n_loop_training,
    n_loop_production=20,
    n_local_steps=10,
    n_global_steps=1000,
    n_chains=500,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    n_max_examples=30000,
    n_NFproposal_batch_size=100000,
    momentum=0.9,
    batch_size=30000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=1,
    output_thinning=10,
    local_sampler_arg=local_sampler_arg,
    # strategies=[Adam_optimizer,"default"],
)


jim.sample(jax.random.PRNGKey(42))
