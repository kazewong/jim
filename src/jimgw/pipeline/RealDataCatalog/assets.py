from sys import prefix
import dagster as dg
import gwosc
import os
import numpy as np
from dagster import DynamicPartitionsDefinition, AssetExecutionContext
from jimgw.core.single_event.data import Data
from jimgw.run.library.IMRPhenomPv2_standard_cbc import (
    IMRPhenomPv2StandardCBCRunDefinition,
)

# Create asset group for run and configuration0

event_partitions_def = DynamicPartitionsDefinition(name="event_name")

@dg.asset(
    key_prefix="RealDataCatalog",
    group_name="prerun",
    description="Fetch all confident events and their gps time",
)
def event_list(context: AssetExecutionContext):
    catalogs = ['GWTC-1-confident', 'GWTC-2.1-confident', 'GWTC-3-confident']
    result = []
    event_names = []
    for catalog in catalogs:
        event_list = gwosc.api.fetch_catalog_json(catalog)['events']
        for event in event_list.values():
            name = event['commonName']
            gps_time = event['GPS']
            result.append((name, gps_time))
            event_names.append(name)
    os.makedirs("data", exist_ok=True)
    with open("data/event_list.txt", "w") as f:
        for name, gps_time in result:
            f.write(f"{name} {gps_time}\n")
    # Register dynamic partitions for event_name
    context.instance.add_dynamic_partitions("event_name", event_names)


# We should be able to partition this asset and run it in parallel for each event.
@dg.multi_asset(
    specs=[
        dg.AssetSpec("Realdata_strain", deps=[event_list]),
        dg.AssetSpec("Realdata_psd", deps=[event_list]),
    ],
    group_name="prerun",
    partitions_def=event_partitions_def,
)
def raw_data(context: AssetExecutionContext):
    ifos = ["H1", "L1", "V1"]
    event_name = context.partition_key
    with open("data/event_list.txt", "r") as f:
        lines = f.readlines()
        event_dict = dict(line.strip().split() for line in lines)
    gps_time = event_dict[event_name]
    start = float(gps_time) - 2
    end = float(gps_time) + 2
    event_dir = os.path.join("data", event_name, "raw")
    os.makedirs(event_dir, exist_ok=True)
    for ifo in ifos:
        try:
            data = Data.from_gwosc(ifo, start, end)
            data.to_file(os.path.join(event_dir, f"{ifo}_data"))
            psd_data = Data.from_gwosc(ifo, start - 512, end + 512)  # This needs to be changed at some point
            if np.isnan(psd_data.td).any():
                raise ValueError(
                    f"PSD FFT length is NaN for {ifo}. "
                    "This can happen when the selected time range contains contaminated data or missing data."
                )
            else:
                psd_data.to_file(os.path.join(event_dir, f"{ifo}_psd"))
        except Exception as e:
            print(f"Error fetching data for {ifo} during {event_name}: {e}")
            continue


@dg.asset(
    key_prefix="RealDataCatalog",
    group_name="prerun",
    description="Configuration file for the run.",
    deps=[raw_data],
    partitions_def=event_partitions_def,

)
def config_file(context: AssetExecutionContext):
    event_name = context.partition_key
    with open("data/event_list.txt", "r") as f:
        lines = f.readlines()
        event_dict = dict(line.strip().split() for line in lines)
    gps_time = float(event_dict[event_name])
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
        gps=gps_time,
        f_min=20.0,
        f_max=2000.0,
        segment_length=4.0,
        post_trigger_length=2.0,
        ifos=["H1", "L1"],
        f_ref=20.0,
    )
    run_dir = f"./data/{event_name}/"
    run.working_dir = run_dir
    run.seed = hash(int(gps_time)) % (2**32 - 1)
    run.local_data_prefix = os.path.join(run_dir, "raw/")
    run.serialize(os.path.join(run_dir, "config.yaml"))

@dg.multi_asset(
    specs=[
        dg.AssetSpec("RealDataCatalog_training_chains", deps=[raw_data, config_file]),
        dg.AssetSpec("RealDataCatalog_training_log_prob", deps=[raw_data, config_file]),
        dg.AssetSpec("RealDataCatalog_training_local_acceptance", deps=[raw_data, config_file]),
        dg.AssetSpec("RealDataCatalog_training_global_acceptance", deps=[raw_data, config_file]),
        dg.AssetSpec("RealDataCatalog_training_loss", deps=[raw_data, config_file]),
        dg.AssetSpec("RealDataCatalog_production_chains", deps=[raw_data, config_file]),
        dg.AssetSpec("RealDataCatalog_production_log_prob", deps=[raw_data, config_file]),
        dg.AssetSpec("RealDataCatalog_production_local_acceptance", deps=[raw_data, config_file]),
        dg.AssetSpec("RealDataCatalog_production_global_acceptance", deps=[raw_data, config_file]),
        dg.AssetSpec("RealDataCatalog_auxiliary_nf_samples", deps=[raw_data, config_file]),
        dg.AssetSpec("RealDataCatalog_auxiliary_prior_samples", deps=[raw_data, config_file]),
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
