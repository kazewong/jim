import dagster as dg
from jimgw.core.population.injection_util import generate_fidiual_population
from jimgw.core.single_event.detector import get_detector_preset
from jimgw.run.library.IMRPhenomPv2_standard_cbc import (
    IMRPhenomPv2StandardCBCRunDefinition,
)
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2
from jimgw.core.single_event.gps_times import (
    greenwich_mean_sidereal_time as compute_gmst,
)
import numpy as np
import os
import yaml

"""
TODO: Make a IO manager to handle the common prefix.
https://docs.dagster.io/guides/build/io-managers/defining-a-custom-io-manager
"""

# Sample a fiducial population


@dg.asset(
    group_name="prerun",
    key_prefix="InjectionRecovery",
)
def sample_population():
    """
    This is a placeholder function for the fiducial population.
    It is used to demonstrate how to create a Dagster asset.
    """
    generate_fidiual_population(
        path_prefix="./data/",
    )


# TODO: Add diagnostics regarding the sampled population.

# Create asset group for run and configuration


@dg.asset(
    group_name="prerun",
    description="Configuration file for the run.",
    deps=[["InjectionRecovery", "sample_population"]],
    key_prefix="InjectionRecovery",
)
def config_file():
    """
    TODO: This should be made more flexible later
    """
    parameters = np.genfromtxt(
        "./data/fiducial_population.csv", delimiter=",", names=True
    )
    run = IMRPhenomPv2StandardCBCRunDefinition(
        M_c_range=(10.0, 80.0),
        q_range=(0.125, 1.0),
        max_s1=0.99,
        max_s2=0.99,
        iota_range=(0.0, np.pi),
        dL_range=(1.0, 2000.0),
        t_c_range=(-0.05, 0.05),
        phase_c_range=(0.0, 2 * np.pi),
        psi_range=(0.0, np.pi),
        ra_range=(0.0, 2 * np.pi),
        dec_range=(-np.pi / 2, np.pi / 2),
        gps=1126259462.4,
        f_min=20.0,
        f_max=1024.0,
        segment_length=4.0,
        post_trigger_length=2.0,
        ifos=["H1", "L1"],
        f_ref=20.0,
    )
    for idx, param in enumerate(parameters):
        os.makedirs(f"./data/runs/{idx}/", exist_ok=True)
        run.working_dir = f"./data/runs/{idx}/"
        run.seed = idx
        run.local_data_prefix = f"./data/runs/{idx}/strains/"
        run.serialize(f"./data/runs/{idx}/config.yaml")


@dg.multi_asset(
    specs=[
        dg.AssetSpec(
            key=["InjectionRecovery", "strain"],
            deps=[["InjectionRecovery", "config_file"]],
        ),
        dg.AssetSpec(
            key=["InjectionRecovery", "psd"],
            deps=[["InjectionRecovery", "config_file"]],
        ),
    ],
    group_name="prerun",
)
def raw_data():
    """
    This is a placeholder function for the raw data.
    It is used to demonstrate how to create a Dagster asset.
    """
    parameters = np.genfromtxt(
        "./data/fiducial_population.csv", delimiter=",", names=True
    )
    keys = list(parameters.dtype.names)  # type: ignore
    for idx, param in enumerate(parameters):
        os.makedirs(f"./data/runs/{idx}/strains/", exist_ok=True)
        config_path = f"./data/runs/{idx}/config.yaml"
        injection_parameters = {key: float(param[idx]) for idx, key in enumerate(keys)}
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            ifos = config["ifos"]
            f_min = config["f_min"]
            f_max = config["f_max"]
            duration = config["segment_length"]
            sampling_frequency = f_max * 2
            f_ref = config["f_ref"]
            gmst = compute_gmst(config["gps"])
            injection_parameters["gmst"] = gmst
            injection_parameters["trigger_time"] = config["gps"]
            for ifo in ifos:
                detector = get_detector_preset()[ifo]
                detector.load_and_set_psd()
                detector.frequency_bounds = (f_min, f_max)
                detector.inject_signal(
                    duration,
                    sampling_frequency,
                    0.0,
                    RippleIMRPhenomPv2(f_ref=f_ref),
                    injection_parameters,
                    is_zero_noise=False,
                )
                detector.data.to_file(f"./data/runs/{idx}/strains/{ifo}_data")
                detector.psd.to_file(f"./data/runs/{idx}/strains/{ifo}_psd")


@dg.multi_asset(
    specs=[
        dg.AssetSpec(
            key=["InjectionRecovery", "training_chains"],
            deps=[["InjectionRecovery", "raw_data"]],
        ),
        dg.AssetSpec(
            key=["InjectionRecovery", "training_log_prob"],
            deps=[["InjectionRecovery", "raw_data"]],
        ),
        dg.AssetSpec(
            key=["InjectionRecovery", "training_local_acceptance"],
            deps=[["InjectionRecovery", "raw_data"]],
        ),
        dg.AssetSpec(
            key=["InjectionRecovery", "training_global_acceptance"],
            deps=[["InjectionRecovery", "raw_data"]],
        ),
        dg.AssetSpec(
            key=["InjectionRecovery", "training_loss"],
            deps=[["InjectionRecovery", "raw_data"]],
        ),
        dg.AssetSpec(
            key=["InjectionRecovery", "production_chains"],
            deps=[["InjectionRecovery", "raw_data"]],
        ),
        dg.AssetSpec(
            key=["InjectionRecovery", "production_log_prob"],
            deps=[["InjectionRecovery", "raw_data"]],
        ),
        dg.AssetSpec(
            key=["InjectionRecovery", "production_local_acceptance"],
            deps=[["InjectionRecovery", "raw_data"]],
        ),
        dg.AssetSpec(
            key=["InjectionRecovery", "production_global_acceptance"],
            deps=[["InjectionRecovery", "raw_data"]],
        ),
        dg.AssetSpec(
            key=["InjectionRecovery", "auxiliary_nf_samples"],
            deps=[["InjectionRecovery", "raw_data"]],
        ),
        dg.AssetSpec(
            key=["InjectionRecovery", "auxiliary_prior_samples"],
            deps=[["InjectionRecovery", "raw_data"]],
        ),
    ],
    group_name="run",
)
def run():
    """
    This is a placeholder function for the IMRPhenomPv2StandardCBCRun class.
    It is used to demonstrate how to create a Dagster asset.
    """
    pass


# Create asset group for diagnostics


@dg.asset(
    group_name="diagnostics",
    deps=[["InjectionRecovery", "training_loss"]],
    key_prefix="InjectionRecovery",
)
def loss_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=[["InjectionRecovery", "training_chains"]],
    key_prefix="InjectionRecovery",
)
def training_chains_corner_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=[["InjectionRecovery", "training_chains"]],
    key_prefix="InjectionRecovery",
)
def training_chains_trace_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=[["InjectionRecovery", "training_chains"]],
    key_prefix="InjectionRecovery",
)
def training_chains_rhat_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=[["InjectionRecovery", "training_log_prob"]],
    key_prefix="InjectionRecovery",
)
def training_log_prob_distribution():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=[["InjectionRecovery", "training_log_prob"]],
    key_prefix="InjectionRecovery",
)
def training_log_prob_evolution():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=[["InjectionRecovery", "training_local_acceptance"]],
    key_prefix="InjectionRecovery",
)
def training_local_acceptance_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=[["InjectionRecovery", "training_global_acceptance"]],
    key_prefix="InjectionRecovery",
)
def training_global_acceptance_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=[["InjectionRecovery", "production_chains"]],
    key_prefix="InjectionRecovery",
)
def production_chains_corner_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=[["InjectionRecovery", "production_chains"]],
    key_prefix="InjectionRecovery",
)
def production_chains_trace_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=[["InjectionRecovery", "production_chains"]],
    key_prefix="InjectionRecovery",
)
def production_chains_rhat_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=[["InjectionRecovery", "production_log_prob"]],
    key_prefix="InjectionRecovery",
)
def production_log_prob_distribution():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=[["InjectionRecovery", "production_log_prob"]],
    key_prefix="InjectionRecovery",
)
def production_log_prob_evolution():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=[["InjectionRecovery", "production_local_acceptance"]],
    key_prefix="InjectionRecovery",
)
def production_local_acceptance_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=[["InjectionRecovery", "production_global_acceptance"]],
    key_prefix="InjectionRecovery",
)
def production_global_acceptance_plot():
    pass
