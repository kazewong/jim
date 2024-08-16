import psutil
p = psutil.Process()
p.cpu_affinity([0])

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from astropy.time import Time

import jax
import jax.numpy as jnp

from jimgw.jim import Jim
from jimgw.prior import CombinePrior, UniformPrior, CosinePrior, SinePrior, PowerLawPrior
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import ZeroLikelihood
from jimgw.transforms import BoundToUnbound
from jimgw.single_event.transforms import MassRatioToSymmetricMassRatioTransform, SkyFrameToDetectorFrameSkyPositionTransform, DistanceToSNRWeightedDistanceTransform, GeocentricArrivalTimeToDetectorArrivalTimeTransform, GeocentricArrivalPhaseToDetectorArrivalPhaseTransform
from flowMC.strategy.optimization import optimization_Adam

jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4

ifos = [H1, L1, V1]

M_c_prior = UniformPrior(10.0, 80.0, parameter_names=["M_c"])
dL_prior = PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.05, 0.05, parameter_names=["t_c"])
phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
iota_prior = SinePrior(parameter_names=["iota"])
psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
dec_prior = CosinePrior(parameter_names=["dec"])

prior = CombinePrior(
    [
        M_c_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        iota_prior,
        psi_prior,
        ra_prior,
        dec_prior,
    ]
)

# calculate the d_hat range
@jnp.vectorize
def calc_R_dets(ra, dec, psi, iota):
    gmst = (
        Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad
    )
    p_iota_term = (1.0 + jnp.cos(iota) ** 2) / 2.0
    c_iota_term = jnp.cos(iota)
    R_dets2 = 0.0
    for ifo in ifos:
        antenna_pattern = ifo.antenna_pattern(ra, dec, psi, gmst)
        p_mode_term = p_iota_term * antenna_pattern["p"]
        c_mode_term = c_iota_term * antenna_pattern["c"]
        R_dets2 += p_mode_term**2 + c_mode_term**2

    return jnp.sqrt(R_dets2)

key1, key2, key3, key4 = jax.random.split(jax.random.PRNGKey(1234), 4)
# generate 10000 samples for each
ra_samples = ra_prior.sample(key1, 10000)["ra"]
dec_samples = dec_prior.sample(key2, 10000)["dec"]
psi_samples = psi_prior.sample(key3, 10000)["psi"]
iota_samples = iota_prior.sample(key4, 10000)["iota"]
R_dets_samples = calc_R_dets(ra_samples, dec_samples, psi_samples, iota_samples)

d_hat_min = dL_prior.xmin / jnp.power(M_c_prior.xmax, 5. / 6.)
d_hat_max = dL_prior.xmax / jnp.power(M_c_prior.xmin, 5. / 6.) / jnp.amin(R_dets_samples)

sample_transforms = [
    # all the user reparametrization transform
    DistanceToSNRWeightedDistanceTransform(name_mapping=[["d_L"], ["d_hat"]], conditional_names=["M_c","ra", "dec", "psi", "iota"], gps_time=gps, ifos=ifos),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(name_mapping = [["phase_c"], ["phase_det"]], conditional_names=["ra", "dec", "psi", "iota"], gps_time=gps, ifo=ifos[0]),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(name_mapping = [["t_c"], ["t_det"]], conditional_names=["ra", "dec"], gps_time=gps, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(name_mapping = [["ra", "dec"], ["zenith", "azimuth"]], gps_time=gps, ifos=ifos),
    # all the bound to unbound transform
    BoundToUnbound(name_mapping = [["M_c"], ["M_c_unbounded"]], original_lower_bound=10.0, original_upper_bound=80.0),
    BoundToUnbound(name_mapping = [["iota"], ["iota_unbounded"]], original_lower_bound=0., original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["psi"], ["psi_unbounded"]], original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["zenith"], ["zenith_unbounded"]], original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["azimuth"], ["azimuth_unbounded"]], original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = [["phase_det"], ["phase_det_unbounded"]], original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = [["t_det"], ["t_det_unbounded"]], original_lower_bound=-0.1, original_upper_bound=0.1),
    BoundToUnbound(name_mapping = [["d_hat"], ["d_hat_unbounded"]], original_lower_bound=d_hat_min, original_upper_bound=d_hat_max),
]

likelihood_transforms = []

likelihood = ZeroLikelihood()

mass_matrix = jnp.eye(len(prior.base_prior))
#mass_matrix = mass_matrix.at[1, 1].set(1e-3)
#mass_matrix = mass_matrix.at[5, 5].set(1e-3)
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
    strategies=["default"],
)

print("Start sampling")
key = jax.random.PRNGKey(42)
jim.sample(key)
jim.print_summary()
samples = jim.get_samples()
