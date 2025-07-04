import time

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

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
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
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

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

# first, fetch a 4s segment centered on GW150914
# for the analysis
gps = 1266645879.396484
start = gps - 2
end = gps + 2

# fetch 4096s of data to estimate the PSD (to be
# careful we should avoid the on-source segment,
# but we don't do this in this example)
psd_start = gps - 2048
psd_end = gps + 2048

# define frequency integration bounds for the likelihood
# we set fmax to 87.5% of the Nyquist frequency to avoid
# data corrupted by the GWOSC antialiasing filter
# (Note that Data.from_gwosc will pull data sampled at
# 4096 Hz by default)
fmin = 20.0
fmax = 896.0

# initialize detectors
ifos = [get_H1(), get_L1()]

for ifo in ifos:
    # set analysis data
    data = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(data)

    # set PSD (Welch estimate)
    psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
    # set an NFFT corresponding to the analysis segment duration
    psd_fftlength = data.duration * data.sampling_frequency
    if jnp.isnan(psd_data.td).any():
        raise ValueError(
            f"PSD FFT length is NaN for {ifo.name}. "
            "This can happen if the data segment is too short."
        )
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
M_c_min, M_c_max = 5.0, 80.0
q_min, q_max = 0.25, 1.0
Mc_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])

prior = prior + [Mc_prior, q_prior]

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
dL_prior = PowerLawPrior(100.0, 6000.0, 2.0, parameter_names=["d_L"])
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
    DistanceToSNRWeightedDistanceTransform(gps_time=gps, ifos=ifos, dL_min=dL_prior.xmin, dL_max=dL_prior.xmax),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=ifos[0]),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(tc_min=t_c_prior.xmin, tc_max=t_c_prior.xmax, gps_time=gps, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
    BoundToUnbound(name_mapping=(["M_c"], ["M_c_unbounded"]), original_lower_bound=M_c_min, original_upper_bound=M_c_max,),
    BoundToUnbound(name_mapping=(["q"], ["q_unbounded"]), original_lower_bound=q_min, original_upper_bound=q_max,),
    BoundToUnbound(name_mapping=(["s1_phi"], ["s1_phi_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi,),
    BoundToUnbound(name_mapping=(["s2_phi"], ["s2_phi_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi,),
    BoundToUnbound(name_mapping=(["iota"], ["iota_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi,),
    BoundToUnbound(name_mapping=(["s1_theta"], ["s1_theta_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi,),
    BoundToUnbound(name_mapping=(["s2_theta"], ["s2_theta_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi,),
    BoundToUnbound(name_mapping=(["s1_mag"], ["s1_mag_unbounded"]), original_lower_bound=0.0, original_upper_bound=0.99,),
    BoundToUnbound(name_mapping=(["s2_mag"], ["s2_mag_unbounded"]), original_lower_bound=0.0, original_upper_bound=0.99,),
    BoundToUnbound(name_mapping=(["phase_det"], ["phase_det_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi,),
    BoundToUnbound(name_mapping=(["psi"], ["psi_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi,),
    BoundToUnbound(name_mapping=(["zenith"], ["zenith_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi,),
    BoundToUnbound(name_mapping=(["azimuth"], ["azimuth_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi,),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform("s1"),
    SphereSpinToCartesianSpinTransform("s2"),
]


likelihood = TransientLikelihoodFD(
    ifos,
    waveform=waveform,
    trigger_time=gps,
    f_min=fmin,
    f_max=fmax,
    # marginalization="time",
)

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    n_chains=500,
    n_local_steps=100,
    n_global_steps=1000,
    n_training_loops=20,
    n_production_loops=10,
    n_epochs=20,
    mala_step_size=2e-3,
    rq_spline_hidden_units=[128, 128],
    rq_spline_n_bins=10,
    rq_spline_n_layers=8,
    learning_rate=1e-3,
    batch_size=10000,
    n_max_examples=30000,
    n_NFproposal_batch_size=100,
    local_thinning=1,
    global_thinning=10,
    history_window=200,
    n_temperatures=0,
    max_temperature=20.0,
    n_tempered_steps=10,
    verbose=True,
)

jim.sample()

print("Done!")

logprob = jim.sampler.resources["log_prob_production"].data
print(jnp.mean(logprob))

chains = jim.get_samples()

import numpy as np
import corner

fig = corner.corner(np.stack([chains[key] for key in jim.prior.parameter_names]).T[::10], labels=jim.prior.parameter_names)
fig.savefig('test')
