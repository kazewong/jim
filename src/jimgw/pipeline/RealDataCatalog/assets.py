import dagster as dg
import gwosc
import os
from jimgw.core.single_event.data import Data
import numpy as np

# Create asset group for run and configuration0

@dg.asset(
    key_prefix="RealDataCatalog",
    group_name="prerun",
    description="Fetch all confident events and their gps time",
)
def event_list():
    catalogs = ['GWTC-1-confident', 'GWTC-2.1-confident', 'GWTC-3-confident']
    result = []
    for catalog in catalogs:
        event_list = gwosc.api.fetch_catalog_json(catalog)['events']
        for event in event_list.values():
            name = event['commonName']
            gps_time = event['GPS']
            result.append((name, gps_time))
    with open("data/event_list.txt", "w") as f:
        for name, gps_time in result:
            f.write(f"{name} {gps_time}\n")


@dg.multi_asset(
    specs=[
        dg.AssetSpec("RealDataCatalog_strain", deps=[event_list]),
        dg.AssetSpec("RealDataCatalog_psd", deps=[event_list]),
    ],
    group_name="prerun",
)
def raw_data():
    ifos = ["H1", "L1", "V1"]
    with open("data/event_list.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            name, gps_time = line.strip().split()
            start = float(gps_time) - 2
            end = float(gps_time) + 2
            event_dir = os.path.join("data", "raw", name)
            os.makedirs(event_dir, exist_ok=True)
            for ifo in ifos:
                try:
                    data = Data.from_gwosc(ifo, start, end)
                    data.to_file(os.path.join(event_dir, f"{ifo}data"))
                    psd_data = Data.from_gwosc(ifo, start - 512, end + 512)  # This needs to be changed at some point
                    # psd_fftlength = data.duration * data.sampling_frequency  # Not used
                    if np.isnan(psd_data.td).any():
                        raise ValueError(
                            f"PSD FFT length is NaN for {ifo}. "
                            "This can happen when the selected time range contains contaminated data or missing data."
                        )
                    else:
                        psd_data.to_file(os.path.join(event_dir, f"{ifo}psd"))
                except Exception as e:
                    print(f"Error fetching data for {ifo} during {name}: {e}")
                    continue


@dg.asset(
    key_prefix="RealDataCatalog",
    group_name="prerun",
    description="Configuration file for the run.",
)
def config_file():
    pass


        


@dg.multi_asset(
    specs=[
        dg.AssetSpec("RealDataCatalog_training_chains", deps=[raw_data]),
        dg.AssetSpec("RealDataCatalog_training_log_prob", deps=[raw_data]),
        dg.AssetSpec("RealDataCatalog_training_local_acceptance", deps=[raw_data]),
        dg.AssetSpec("RealDataCatalog_training_global_acceptance", deps=[raw_data]),
        dg.AssetSpec("RealDataCatalog_training_loss", deps=[raw_data]),
        dg.AssetSpec("RealDataCatalog_production_chains", deps=[raw_data]),
        dg.AssetSpec("RealDataCatalog_production_log_prob", deps=[raw_data]),
        dg.AssetSpec("RealDataCatalog_production_local_acceptance", deps=[raw_data]),
        dg.AssetSpec("RealDataCatalog_production_global_acceptance", deps=[raw_data]),
        dg.AssetSpec("RealDataCatalog_auxiliary_nf_samples", deps=[raw_data]),
        dg.AssetSpec("RealDataCatalog_auxiliary_prior_samples", deps=[raw_data]),
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
    deps=["RealDataCatalog_training_loss"],
    key_prefix="RealDataCatalog",
)
def loss_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["RealDataCatalog_training_chains"],
    key_prefix="RealDataCatalog",
)
def training_chains_corner_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["RealDataCatalog_training_chains"],
    key_prefix="RealDataCatalog",
)
def training_chains_trace_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["RealDataCatalog_training_chains"],
    key_prefix="RealDataCatalog",
)
def training_chains_rhat_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["RealDataCatalog_training_log_prob"],
    key_prefix="RealDataCatalog",
)
def training_log_prob_distribution():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["RealDataCatalog_training_log_prob"],
    key_prefix="RealDataCatalog",
)
def training_log_prob_evolution():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["RealDataCatalog_training_local_acceptance"],
    key_prefix="RealDataCatalog",
)
def training_local_acceptance_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["RealDataCatalog_training_global_acceptance"],
    key_prefix="RealDataCatalog",
)
def training_global_acceptance_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["RealDataCatalog_production_chains"],
    key_prefix="RealDataCatalog",
)
def production_chains_corner_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["RealDataCatalog_production_chains"],
    key_prefix="RealDataCatalog",
)
def production_chains_trace_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["RealDataCatalog_production_chains"],
    key_prefix="RealDataCatalog",
)
def production_chains_rhat_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["RealDataCatalog_production_log_prob"],
    key_prefix="RealDataCatalog",
)
def production_log_prob_distribution():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["RealDataCatalog_production_log_prob"],
    key_prefix="RealDataCatalog",
)
def production_log_prob_evolution():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["RealDataCatalog_production_local_acceptance"],
    key_prefix="RealDataCatalog",
)
def production_local_acceptance_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["RealDataCatalog_production_global_acceptance"],
    key_prefix="RealDataCatalog",
)
def production_global_acceptance_plot():
    pass
