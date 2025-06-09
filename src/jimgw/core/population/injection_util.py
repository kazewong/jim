"""
TODO: This file should be refactored into better structured modules.
Right now this is here just because I am developing the injection pipeline
"""

import numpy as np
import jax
import jax.numpy as jnp
from jimgw.core.prior import (
  UniformPrior,
  UniformSpherePrior,
  SinePrior,
  CosinePrior,
  PowerLawPrior,
  CombinePrior,
)
from jimgw.core.single_event.transforms import (
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
)

def generate_fidiual_population(path_prefix:str = "./", seed: int = 2046, n_events: int = 7):
    prior = []
    M_c_min, M_c_max = 10.0, 80.0
    q_min, q_max = 0.125, 1.0
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
    dL_prior = PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"])
    t_c_prior = UniformPrior(-0.05, 0.05, parameter_names=["t_c"])
    phase_c_prior = UniformPrior(0.0, 2 * np.pi, parameter_names=["phase_c"])
    psi_prior = UniformPrior(0.0, np.pi, parameter_names=["psi"])
    ra_prior = UniformPrior(0.0, 2 * np.pi, parameter_names=["ra"])
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

    samples = prior.sample(
        jax.random.PRNGKey(seed),
        n_events,
    )

    likelihood_transforms = [
        MassRatioToSymmetricMassRatioTransform,
        SphereSpinToCartesianSpinTransform("s1"),
        SphereSpinToCartesianSpinTransform("s2"),
    ]

    # Save injection campaign configuration

    # Save population data
    with open(path_prefix + "fiducial_population.csv", "w") as f:
        params = {key: samples[key][0] for key in samples.keys()}
        for transform in likelihood_transforms:
            params = transform.forward(params)
        f.write(",".join([key for key in params.keys()]) + "\n")
        f.write(",".join([str(params[key]) for key in params.keys()]) + "\n")
        for i in range(1, n_events):
            params = {key: samples[key][i] for key in samples.keys()}
            # Apply transforms
            for transform in likelihood_transforms:
                params = transform.forward(params)
            f.write(",".join([str(params[key]) for key in params.keys()]) + "\n")