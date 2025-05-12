import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from jimgw.jim import Jim
from jimgw.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
)
from jimgw.single_event.data import Data
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomD
from jimgw.transforms import BoundToUnbound
from jimgw.single_event.transforms import (
    ComponentMassesToChirpMassSymmetricMassRatioTransform,
    SkyFrameToDetectorFrameSkyPositionTransform,
    ComponentMassesToChirpMassMassRatioTransform,
)
from jimgw.single_event.utils import Mc_q_to_m1_m2

###########################################
########## First we grab data #############
###########################################

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
duration = 4
post_trigger_duration = 2
start = gps - 2
end = gps + 2
fmin = 20.0
fmax = 1024.0

ifos = [H1, L1]

for ifo in ifos:
    data = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(data)

    psd_data = Data.from_gwosc(ifo.name, gps-16, gps+16)
    psd_fftlength = data.duration * data.sampling_frequency
    ifo.set_psd(psd_data.to_psd(nperseg=psd_fftlength))

M_c_min, M_c_max = 10.0, 80.0
q_min, q_max = 0.125, 1.0
m_1_prior = UniformPrior(
    Mc_q_to_m1_m2(M_c_min, q_max)[0],
    Mc_q_to_m1_m2(M_c_max, q_min)[0],
    parameter_names=["m_1"],
)
m_2_prior = UniformPrior(
    Mc_q_to_m1_m2(M_c_min, q_min)[1],
    Mc_q_to_m1_m2(M_c_max, q_max)[1],
    parameter_names=["m_2"],
)
s1z_prior = UniformPrior(-1.0, 1.0, parameter_names=["s1_z"])
s2z_prior = UniformPrior(-1.0, 1.0, parameter_names=["s2_z"])
dL_prior = PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.05, 0.05, parameter_names=["t_c"])
phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
iota_prior = SinePrior(parameter_names=["iota"])
psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
dec_prior = CosinePrior(parameter_names=["dec"])

prior = CombinePrior(
    [
        m_1_prior,
        m_2_prior,
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
    ComponentMassesToChirpMassMassRatioTransform,
    SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
    BoundToUnbound(name_mapping=[["M_c"], ["M_c_unbounded"]], original_lower_bound=M_c_min, original_upper_bound=M_c_max,),
    BoundToUnbound(name_mapping=[["q"], ["q_unbounded"]], original_lower_bound=q_min, original_upper_bound=q_max,),
    BoundToUnbound(name_mapping=[["s1_z"], ["s1_z_unbounded"]], original_lower_bound=-1.0, original_upper_bound=1.0,),
    BoundToUnbound(name_mapping=[["s2_z"], ["s2_z_unbounded"]], original_lower_bound=-1.0, original_upper_bound=1.0,),
    BoundToUnbound(name_mapping=[["d_L"], ["d_L_unbounded"]], original_lower_bound=0.0, original_upper_bound=2000.0,),
    BoundToUnbound(name_mapping=[["t_c"], ["t_c_unbounded"]], original_lower_bound=-0.05, original_upper_bound=0.05,),
    BoundToUnbound(name_mapping=[["phase_c"], ["phase_c_unbounded"]], original_lower_bound=0.0, original_upper_bound=2 * jnp.pi,),
    BoundToUnbound(name_mapping=[["iota"], ["iota_unbounded"]], original_lower_bound=0.0, original_upper_bound=jnp.pi,),
    BoundToUnbound(name_mapping=[["psi"], ["psi_unbounded"]], original_lower_bound=0.0, original_upper_bound=jnp.pi,),
    BoundToUnbound(name_mapping=[["zenith"], ["zenith_unbounded"]], original_lower_bound=0.0, original_upper_bound=jnp.pi,),
    BoundToUnbound(name_mapping=[["azimuth"], ["azimuth_unbounded"]], original_lower_bound=0.0, original_upper_bound=2 * jnp.pi,),
]

likelihood_transforms = [
    ComponentMassesToChirpMassSymmetricMassRatioTransform,
]

likelihood = TransientLikelihoodFD(
    ifos,
    waveform=RippleIMRPhenomD(),
    f_min=fmin,
    f_max=fmax,
    trigger_time=gps,
)

mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[5, 5].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 3e-3}


jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    n_chains=4,
    n_local_steps=5,
    n_global_steps=5,
    n_training_loops=2,
    n_production_loops=2,
    n_epochs=2,
    learning_rate=1e-4,
    n_max_examples=30,
    batch_size=100,
)

jim.sample(jax.random.PRNGKey(42))
# jim.get_samples()
# jim.print_summary()
