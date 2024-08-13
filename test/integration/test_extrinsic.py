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
q_prior = UniformPrior(0.125, 1.0, parameter_names=["q"])
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
        q_prior,
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
    # all the user reparametrization transform
    DistanceToSNRWeightedDistanceTransform(name_mapping=[["d_L"], ["d_hat"]], gps_time=gps, ifos=ifos),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(name_mapping = [["phase_c", "phase_det"]], gps_time=gps, ifo=ifos[0]),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(name_mapping = [["t_c", "t_det"]], gps_time=gps, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(name_mapping = [["ra", "dec"], ["zenith", "azimuth"]], gps_time=gps, ifos=ifos),
    # all the bound to unbound transform
    BoundToUnbound(name_mapping = [["M_c"], ["M_c_unbounded"]], original_lower_bound=10.0, original_upper_bound=80.0),
    BoundToUnbound(name_mapping = [["q"], ["q_unbounded"]], original_lower_bound=0.125, original_upper_bound=1.0),
    BoundToUnbound(name_mapping = [["iota"], ["iota_unbounded"]], original_lower_bound=0., original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["psi"], ["psi_unbounded"]], original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["zenith"], ["zenith_unbounded"]], original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["azimuth"], ["azimuth_unbounded"]], original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = [["d_hat"], ["d_hat_unbounded"]], original_lower_bound=2.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = [["phase_det"], ["phase_det_unbounded"]], original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = [["t_det"], ["t_det_unbounded"]], original_lower_bound=-0.1, original_upper_bound=0.1),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform(name_mapping=[["q"], ["eta"]]),
]

likelihood = ZeroLikelihood()

mass_matrix = jnp.eye(9)
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
    strategies=[Adam_optimizer, "default"],
)

jim.sample(jax.random.PRNGKey(42))
jim.get_samples()
jim.print_summary()
