import dagster as dg
from jimgw.core.population.injection_util import generate_fidiual_population
from jimgw.run.library.IMRPhenomPv2_standard_cbc import IMRPhenomPv2StandardCBCRunDefinition
import numpy as np
import os

"""
TODO: Make a IO manager to handle the common prefix.
https://docs.dagster.io/guides/build/io-managers/defining-a-custom-io-manager
"""

# Sample a fiducial population

@dg.asset(
    group_name="prerun",    
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
    deps=[sample_population],
)
def config_file():
    """
    TODO: This should be made more flexible later
    """
    parameters = np.genfromtxt('./data/fiducial_population.csv', delimiter=',', names=True)
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
        dg.AssetSpec("strain", deps=[config_file]),
        dg.AssetSpec("psd", deps=[config_file]),
    ],
    group_name="prerun",
)
def raw_data():
    """
    This is a placeholder function for the raw data.
    It is used to demonstrate how to create a Dagster asset.
    """
    parameters = np.genfromtxt('./data/fiducial_population.csv', delimiter=',', names=True)
    for idx, param in enumerate(parameters):
        os.makedirs(f"./data/runs/{idx}/strains/", exist_ok=True)


@dg.multi_asset(
    specs=[
        dg.AssetSpec("training_chains", deps=[raw_data]),
        dg.AssetSpec("training_log_prob", deps=[raw_data]),
        dg.AssetSpec("training_local_acceptance", deps=[raw_data]),
        dg.AssetSpec("training_global_acceptance", deps=[raw_data]),
        dg.AssetSpec("training_loss", deps=[raw_data]),
        dg.AssetSpec("production_chains", deps=[raw_data]),
        dg.AssetSpec("production_log_prob", deps=[raw_data]),
        dg.AssetSpec("production_local_acceptance", deps=[raw_data]),
        dg.AssetSpec("production_global_acceptance", deps=[raw_data]),
        dg.AssetSpec("auxiliary_nf_samples", deps=[raw_data]),
        dg.AssetSpec("auxiliary_prior_samples", deps=[raw_data]),
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


@dg.asset(group_name="diagnostics", deps=["training_loss"])
def loss_plot():
    pass


@dg.asset(group_name="diagnostics", deps=["training_chains"])
def training_chains_corner_plot():
    pass


@dg.asset(group_name="diagnostics", deps=["training_chains"])
def training_chains_trace_plot():
    pass


@dg.asset(group_name="diagnostics", deps=["training_chains"])
def training_chains_rhat_plot():
    pass


@dg.asset(group_name="diagnostics", deps=["training_log_prob"])
def training_log_prob_distribution():
    pass


@dg.asset(group_name="diagnostics", deps=["training_log_prob"])
def training_log_prob_evolution():
    pass


@dg.asset(group_name="diagnostics", deps=["training_local_acceptance"])
def training_local_acceptance_plot():
    pass


@dg.asset(group_name="diagnostics", deps=["training_global_acceptance"])
def training_global_acceptance_plot():
    pass


@dg.asset(group_name="diagnostics", deps=["production_chains"])
def production_chains_corner_plot():
    pass


@dg.asset(group_name="diagnostics", deps=["production_chains"])
def production_chains_trace_plot():
    pass


@dg.asset(group_name="diagnostics", deps=["production_chains"])
def production_chains_rhat_plot():
    pass


@dg.asset(group_name="diagnostics", deps=["production_log_prob"])
def production_log_prob_distribution():
    pass


@dg.asset(group_name="diagnostics", deps=["production_log_prob"])
def production_log_prob_evolution():
    pass


@dg.asset(group_name="diagnostics", deps=["production_local_acceptance"])
def production_local_acceptance_plot():
    pass


@dg.asset(group_name="diagnostics", deps=["production_global_acceptance"])
def production_global_acceptance_plot():
    pass
