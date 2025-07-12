import time
from pathlib import Path

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
    RayleighPrior,
)
from jimgw.core.single_event.detector import H1, L1
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2
from jimgw.core.transforms import PeriodicTransform
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
)
from jimgw.core.single_event.gps_times import (
    greenwich_mean_sidereal_time as compute_gmst,
)

jax.config.update("jax_enable_x64", True)

#################################################
########## Parse the input settings #############
#################################################

label = "GW150914_like_injection_PhenomPv2"
outdir = Path("./" + label)

print(f"Get label as {label}")
print(f"Setting output directory to: {outdir.as_posix()}")

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()
rng_key = jax.random.PRNGKey(int(total_time_start))
rng_key, *sub_key = jax.random.split(rng_key, 2)

gps_time = total_time_start - 1000
gmst = compute_gmst(gps_time)
random_samples = jax.random.uniform(sub_key[0], 3, maxval=jnp.pi)

injection_parameters = {
    "M_c": 30.0,
    "eta": 0.21,
    "s1_x": 0.1,
    "s1_y": -0.1,
    "s1_z": 0.3,
    "s2_x": 0.2,
    "s2_y": -0.1,
    "s2_z": -0.2,
    "ra": random_samples[0] * 2.0,
    "dec": random_samples[1] - jnp.pi / 2,
    "psi": random_samples[2] - jnp.pi / 2,
    "d_L": 600.0,
    "iota": 0.5,
    "phase_c": jnp.pi - 0.3,
    "t_c": 0.1,
}
injection_parameters["gmst"] = compute_gmst(gps_time)

_inj_params = injection_parameters.copy()
q_eta_transform = MassRatioToSymmetricMassRatioTransform
s1_transform = SphereSpinToCartesianSpinTransform("s1")
s2_transform = SphereSpinToCartesianSpinTransform("s2")
_inj_params = q_eta_transform.backward(_inj_params)
_inj_params = s1_transform.backward(_inj_params)
_inj_params = s2_transform.backward(_inj_params)
injection_parameters.update(_inj_params)

print("The injection parameters are")
for key, value in injection_parameters.items():
    print(f"-- {key + ':':10} {float(value):> 13.6f}")
injection_parameters = {
    key: jnp.array(value) for key, value in injection_parameters.items()
}

f_min = 30.0
f_max = 1024.0
duration = 4.0
sampling_frequency = f_max * 2

# initialize waveform
PhenomPv2 = RippleIMRPhenomPv2(f_ref=20)

ifos = [H1, L1]
for ifo in ifos:
    ifo.load_and_set_psd()
    ifo.frequency_bounds = (f_min, f_max)
    ifo.inject_signal(
        duration,
        sampling_frequency,
        0.0,
        PhenomPv2,
        injection_parameters,
        is_zero_noise=False,
    )

###########################################
########## Set up priors ##################
###########################################

M_c_min, M_c_max = 21.418182160215295, 41.97447913941358
q_min, q_max = 0.125, 1.0
dL_min, dL_max = 10.0, 2e3  # 1e4
prior = [
    UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"]),
    UniformPrior(q_min, q_max, parameter_names=["q"]),
    UniformSpherePrior(parameter_names=["s1"]),
    UniformSpherePrior(parameter_names=["s2"]),
    SinePrior(parameter_names=["iota"]),
    PowerLawPrior(dL_min, dL_max, 2.0, parameter_names=["d_L"]),
    UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
    UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"]),
    UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
    UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
    CosinePrior(parameter_names=["dec"]),
]

prior += [
    RayleighPrior(1.5, parameter_names=["periodic_1"]),
    RayleighPrior(1.5, parameter_names=["periodic_2"]),
    RayleighPrior(1.5, parameter_names=["periodic_3"]),
    RayleighPrior(1.5, parameter_names=["periodic_4"]),
    RayleighPrior(1.5, parameter_names=["periodic_5"]),
]

prior = CombinePrior(prior)

# Defining Transforms
sample_transforms = [
    DistanceToSNRWeightedDistanceTransform(
        gps_time=gps_time, ifos=ifos, dL_min=dL_min, dL_max=dL_max
    ),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
        gps_time=gps_time, ifo=ifos[0]
    ),
    SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps_time, ifos=ifos),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(
        tc_min=-0.1, tc_max=0.1, gps_time=gps_time, ifo=ifos[0]
    ),
    PeriodicTransform(
        name_mapping=(["periodic_1", "s1_phi"], ["s1_phi_x", "s1_phi_y"]),
        xmin=0.0,
        xmax=2 * jnp.pi,
    ),
    PeriodicTransform(
        name_mapping=(["periodic_2", "s2_phi"], ["s2_phi_x", "s2_phi_y"]),
        xmin=0.0,
        xmax=2 * jnp.pi,
    ),
    PeriodicTransform(
        name_mapping=(["periodic_3", "ra"], ["ra_x", "ra_y"]), xmin=0.0, xmax=2 * jnp.pi
    ),
    PeriodicTransform(
        name_mapping=(["periodic_4", "phase_det"], ["phase_det_x", "phase_det_y"]),
        xmin=0.0,
        xmax=2 * jnp.pi,
    ),
    PeriodicTransform(
        name_mapping=(["periodic_5", "psi"], ["psi_base_x", "psi_base_y"]),
        xmin=0.0,
        xmax=jnp.pi,
    ),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform("s1"),
    SphereSpinToCartesianSpinTransform("s2"),
]


likelihood = TransientLikelihoodFD(
    [H1, L1],
    waveform=PhenomPv2,
    trigger_time=gps_time,
    f_min=f_min,
    f_max=f_max,
)

mass_matrix = jnp.eye(prior.n_dim)

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    rng_key=jax.random.PRNGKey(12345),
    n_chains=512,
    n_local_steps=20,
    n_global_steps=15,
    n_training_loops=200,
    n_production_loops=150,
    n_epochs=15,
    mala_step_size=mass_matrix * 1e-3,
    rq_spline_hidden_units=[128, 128],
    rq_spline_n_bins=10,
    rq_spline_n_layers=8,
    learning_rate=1e-3,
    batch_size=10000,
    n_max_examples=10000,
    n_NFproposal_batch_size=5,
    local_thinning=1,
    global_thinning=1,
    history_window=200,
    n_temperatures=15,
    max_temperature=20,
    n_tempered_steps=-1,
    verbose=True,
)

jim.sample()

resources = jim.sampler.resources
logprob_train = resources["log_prob_training"].data
logprob_prod = resources["log_prob_production"].data
print("Mean log posterior (Training): ", jnp.mean(logprob_train))
print("Mean log posterior (Production): ", jnp.mean(logprob_prod))
acceptance_train = resources["log_accs_training"].data
acceptance_prod = resources["log_accs_production"].data
print("Mean acceptance (Training): ", jnp.mean(acceptance_train))
print("Mean acceptance (Production): ", jnp.mean(acceptance_prod))

tempered_log_pdf = resources["tempered_logpdf"]

end_time = time.time()
print("Total time taken: ", end_time - total_time_start)

print("Sampling Done!")

print("Preparing samples")
samples = jim.get_samples()
samples = {key: samples[key] for key in samples.keys()}
jnp.savez(outdir / "samples.npz", **samples)

print("Preparing results")
log_poste = jim.sampler.resources["log_prob_production"].data.reshape(-1)
log_prior = jax.vmap(prior.log_prob)(samples)
log_likelihood = log_poste - log_prior
jnp.savez(
    outdir / "result.npz",
    log_prior=log_prior,
    log_prob=log_poste,
    tempered_log_pdf=tempered_log_pdf,
)

print("Preparing NF samples")
nf_samples, _ = jim.sampler.resources["global_sampler"].sample_flow(
    jax.random.PRNGKey(123), 5000
)
nf_samples = jax.vmap(jim.add_name)(nf_samples)
for transform in reversed(sample_transforms):
    nf_samples = jax.vmap(transform.backward)(nf_samples)
jnp.savez(outdir / "nf_samples.npz", **nf_samples)

print("DONE!")
