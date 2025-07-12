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
    catalogs = ["GWTC-1-confident", "GWTC-2.1-confident", "GWTC-3-confident"]
    result = []
    event_names = []
    for catalog in catalogs:
        event_list = gwosc.api.fetch_catalog_json(catalog)["events"]
        for event in event_list.values():
            name = event["commonName"]
            gps_time = event["GPS"]
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
        dg.AssetSpec(["RealDataCatalog", "strain"], deps=[event_list]),
        dg.AssetSpec(["RealDataCatalog", "psd"], deps=[event_list]),
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
            # TODO: Perhaps we should make sure the PSD estimation window are the same accross all IFOs?
            psd_data = Data.from_gwosc(
                ifo, start - 4098, end - 2
            )  # This needs to be changed at some point
            if np.isnan(psd_data.td).any():
                psd_data = Data.from_gwosc(ifo, start + 2, end + 4098)
            if np.isnan(psd_data.td).any():
                psd_data = Data.from_gwosc(ifo, start - 2048, end + 2048)
            if np.isnan(psd_data.td).any():
                raise ValueError(
                    f"PSD FFT length is NaN for {ifo}. "
                    "This can happen when the selected time range contains contaminated data or missing data."
                )
            else:
                psd_fftlength = data.duration * data.sampling_frequency
                psd_data = psd_data.to_psd(nperseg=psd_fftlength)
                psd_data.to_file(os.path.join(event_dir, f"{ifo}_psd"))
        except Exception as e:
            print(f"Error fetching data for {ifo} during {event_name}: {e}")
            continue


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "strain"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def raw_data_plot(context: AssetExecutionContext):
    """
    Plot the raw strain data for each IFO for the event.
    """
    import matplotlib.pyplot as plt

    event_name = context.partition_key
    event_dir = os.path.join("data", event_name, "raw")
    plots_dir = os.path.join("data", event_name, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    ifos = ["H1", "L1", "V1"]
    plot_paths = []
    for ifo in ifos:
        data_file = os.path.join(event_dir, f"{ifo}_data.npz")
        if os.path.exists(data_file):
            data = np.load(data_file)
            t = data["epoch"] + np.arange(data["td"].shape[0]) * data["dt"]
            td = data["td"]
            if t is not None and td is not None:
                plt.figure()
                plt.plot(t, td)
                plt.xlabel("Time (s)")
                plt.ylabel("Strain")
                plt.title(f"{ifo} Strain for {event_name}")
                plot_path = os.path.join(plots_dir, f"{ifo}_strain.png")
                plt.savefig(plot_path)
                plt.close()
                plot_paths.append(plot_path)
    return plot_paths


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "psd"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def psd_plot(context: AssetExecutionContext):
    """
    Plot the PSD for each IFO for the event.
    """
    import matplotlib.pyplot as plt

    event_name = context.partition_key
    event_dir = os.path.join("data", event_name, "raw")
    plots_dir = os.path.join("data", event_name, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    ifos = ["H1", "L1", "V1"]
    plot_paths = []
    for ifo in ifos:
        psd_file = os.path.join(event_dir, f"{ifo}_psd.npz")
        if os.path.exists(psd_file):
            data = np.load(psd_file)
            f = data["frequencies"]
            psd = data["values"]
            plt.figure()
            plt.loglog(f, psd)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD")
            plt.title(f"{ifo} PSD for {event_name}")
            plot_path = os.path.join(plots_dir, f"{ifo}_psd.png")
            plt.savefig(plot_path)
            plt.close()
            plot_paths.append(plot_path)
    return plot_paths


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
    # Check which IFOs have both data and PSD files present
    available_ifos: list[str] = []
    raw_dir = os.path.join("data", event_name, "raw")
    for ifo in ["H1", "L1", "V1"]:
        data_file = os.path.join(raw_dir, f"{ifo}_data.npz")
        psd_file = os.path.join(raw_dir, f"{ifo}_psd.npz")
        if os.path.exists(data_file) and os.path.exists(psd_file):
            available_ifos.append(ifo)
    if available_ifos == []:
        raise RuntimeError(
            f"No IFOs with both data and PSD found for event {event_name}"
        )
    run = IMRPhenomPv2StandardCBCRunDefinition(
        n_chains=500,
        n_local_steps=100,
        n_global_steps=1000,
        n_training_loops=20,
        n_production_loops=10,
        n_epochs=20,
        mala_step_size=0.002,
        rq_spline_hidden_units=[128, 128],
        rq_spline_n_bins=10,
        rq_spline_n_layers=8,
        learning_rate=0.001,
        batch_size=10000,
        n_max_examples=30000,
        local_thinning=1,
        global_thinning=10,
        n_NFproposal_batch_size=100,
        history_window=200,
        n_temperatures=0,
        max_temperature=20.0,
        n_tempered_steps=10,
        M_c_range=(5.0, 80.0),
        q_range=(0.25, 1.0),
        max_s1=0.99,
        max_s2=0.99,
        iota_range=(0.0, 3.141592653589793),
        dL_range=(100.0, 6000.0),
        # t_c_range=(-0.05, 0.05),
        phase_c_range=(0.0, 6.283185307179586),
        psi_range=(0.0, 3.141592653589793),
        ra_range=(0.0, 6.283185307179586),
        dec_range=(-1.5707963267948966, 1.5707963267948966),
        gps=gps_time,
        f_min=20.0,
        f_max=896.0,
        segment_length=4.0,
        post_trigger_length=2.0,
        ifos=available_ifos,
        f_ref=20.0,
    )
    run_dir = f"./data/{event_name}/"
    run.working_dir = run_dir
    run.seed = hash(int(gps_time)) % (2**32 - 1)
    run.local_data_prefix = os.path.join(run_dir, "raw/")
    run.serialize(os.path.join(run_dir, "config.yaml"))


@dg.multi_asset(
    specs=[
        dg.AssetSpec(
            key=["RealDataCatalog", "training_loss"], deps=[raw_data, config_file]
        ),
        dg.AssetSpec(
            key=["RealDataCatalog", "production_chains"], deps=[raw_data, config_file]
        ),
        dg.AssetSpec(
            key=["RealDataCatalog", "production_log_prob"], deps=[raw_data, config_file]
        ),
        dg.AssetSpec(
            key=["RealDataCatalog", "production_local_acceptance"],
            deps=[raw_data, config_file],
        ),
        dg.AssetSpec(
            key=["RealDataCatalog", "production_global_acceptance"],
            deps=[raw_data, config_file],
        ),
        dg.AssetSpec(
            key=["RealDataCatalog", "auxiliary_nf_samples"],
            deps=[raw_data, config_file],
        ),
        dg.AssetSpec(
            key=["RealDataCatalog", "auxiliary_prior_samples"],
            deps=[raw_data, config_file],
        ),
    ],
    group_name="run",
    partitions_def=event_partitions_def,
)
def run(context: AssetExecutionContext):
    """
    Loads results from the output of execute_single_run.py and yields each asset.
    """
    pass


# Create asset group for diagnostics
@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "training_loss"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def loss_plot(context: AssetExecutionContext):
    """
    Generate and save a loss plot from the training_loss asset.
    """
    import matplotlib.pyplot as plt

    event_name = context.partition_key
    run_dir = os.path.join("data", event_name)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.npz")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    results = np.load(results_path, allow_pickle=True)
    loss = results["loss_data"]
    if loss is None:
        raise ValueError("No 'loss' key found in loss_data.")
    plt.figure()
    plt.plot(loss)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss for {event_name}")
    plot_path = os.path.join(plots_dir, "training_loss.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "production_chains"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def production_chains_corner_plot(context: AssetExecutionContext):
    """
    Generate and save a corner plot from the production_chains asset.
    """
    import matplotlib.pyplot as plt
    import corner

    event_name = context.partition_key
    run_dir = os.path.join("data", event_name)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.npz")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    results = np.load(results_path, allow_pickle=True)
    chains = results["chains"].item()
    # keys = np.sort(list(chains.keys()))
    keys = [
        "M_c",
        "q",
        "s1_mag",
        "s1_theta",
        "s1_phi",
        "s2_mag",
        "s2_theta",
        "s2_phi",
        "iota",
        "d_L",
        "phase_c",
        "psi",
        "ra",
        "dec",
    ]
    samples = np.array([chains[key] for key in keys]).T
    fig = corner.corner(samples[::10], labels=keys)
    plot_path = os.path.join(plots_dir, "production_chains_corner.png")
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "auxiliary_nf_samples"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def nf_samples_corner_plot(context: AssetExecutionContext):
    """
    Generate and save a corner plot from the auxiliary_nf_samples asset.
    """
    import matplotlib.pyplot as plt
    import corner

    event_name = context.partition_key
    run_dir = os.path.join("data", event_name)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.npz")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    results = np.load(results_path, allow_pickle=True)
    nf_samples = results["nf_samples"].item()
    # keys = np.sort(list(nf_samples.keys()))
    keys = [
        "M_c",
        "q",
        "s1_mag",
        "s1_theta",
        "s1_phi",
        "s2_mag",
        "s2_theta",
        "s2_phi",
        "iota",
        "d_L",
        "phase_c",
        "psi",
        "ra",
        "dec",
    ]
    nf_samples = np.array([nf_samples[key] for key in keys]).T
    fig = corner.corner(nf_samples, labels=keys)  # Thinning for better visualization
    plot_path = os.path.join(plots_dir, "nf_samples_corner.png")
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "auxiliary_prior_samples"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def prior_samples_corner_plot(context: AssetExecutionContext):
    """
    Generate and save a corner plot from the auxiliary_prior_samples asset.
    """
    import matplotlib.pyplot as plt
    import corner

    event_name = context.partition_key
    run_dir = os.path.join("data", event_name)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.npz")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    results = np.load(results_path, allow_pickle=True)
    prior_samples = results["prior_samples"].item()
    # keys = np.sort(list(prior_samples.keys()))
    keys = [
        "M_c",
        "q",
        "s1_mag",
        "s1_theta",
        "s1_phi",
        "s2_mag",
        "s2_theta",
        "s2_phi",
        "iota",
        "d_L",
        "phase_c",
        "psi",
        "ra",
        "dec",
    ]
    prior_samples = np.array([prior_samples[key] for key in keys]).T
    fig = corner.corner(prior_samples, labels=keys)  # Thinning for better visualization
    plot_path = os.path.join(plots_dir, "prior_samples_corner.png")
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "production_chains"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def production_chains_trace_plot(context: AssetExecutionContext):
    """
    Generate and save a trace plot for the production chains.
    """
    import matplotlib.pyplot as plt

    event_name = context.partition_key
    run_dir = os.path.join("data", event_name)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.npz")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    results = np.load(results_path, allow_pickle=True)
    chains = results["chains"].item()
    keys = [
        "M_c",
        "q",
        "s1_mag",
        "s1_theta",
        "s1_phi",
        "s2_mag",
        "s2_theta",
        "s2_phi",
        "iota",
        "d_L",
        "phase_c",
        "psi",
        "ra",
        "dec",
    ]
    n_params = len(keys)
    samples = [chains[key] for key in keys]

    plt.figure(figsize=(15, 2.5 * n_params))
    for i, key in enumerate(keys):
        plt.subplot(n_params, 1, i + 1)
        plt.plot(samples[i])
        plt.ylabel(key)
        if i == 0:
            plt.title(f"Production Chains Trace Plot for {event_name}")
    plt.xlabel("Sample")
    plot_path = os.path.join(plots_dir, "production_chains_trace.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "production_log_prob"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def production_log_prob_distribution(context: AssetExecutionContext):
    """
    Generate and save a histogram of the production log probability.
    """
    import matplotlib.pyplot as plt

    event_name = context.partition_key
    run_dir = os.path.join("data", event_name)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.npz")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    results = np.load(results_path, allow_pickle=True)
    log_prob = results["log_probs"]
    if log_prob is None:
        raise ValueError("No 'log_prob' key found in loss_data.")
    plt.figure()
    plt.hist(log_prob.flatten(), bins=50, alpha=0.7)
    plt.xlabel("Log Probability")
    plt.ylabel("Frequency")
    plt.title(f"Production Log Probability Distribution for {event_name}")
    plot_path = os.path.join(plots_dir, "production_log_prob_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "production_log_prob"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def production_log_prob_evolution(context: AssetExecutionContext):
    """
    Generate and save a plot of the evolution of the production log probability.
    """
    import matplotlib.pyplot as plt

    event_name = context.partition_key
    run_dir = os.path.join("data", event_name)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.npz")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    results = np.load(results_path, allow_pickle=True)
    log_prob = results["log_probs"]
    if log_prob is None:
        raise ValueError("No 'log_prob' key found in loss_data.")
    plt.figure()
    plt.plot(log_prob)
    plt.xlabel("Sample")
    plt.ylabel("Log Probability")
    plt.title(f"Production Log Probability Evolution for {event_name}")
    plot_path = os.path.join(plots_dir, "production_log_prob_evolution.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "production_local_acceptance"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def production_local_acceptance_plot(context: AssetExecutionContext):
    """
    Generate and save a plot of the local acceptance rate.
    """
    import matplotlib.pyplot as plt

    event_name = context.partition_key
    run_dir = os.path.join("data", event_name)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.npz")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    results = np.load(results_path, allow_pickle=True)
    acceptance = results["acceptance"].item().get("local", None)
    if acceptance is None:
        raise ValueError("No 'local' key found in acceptance.")
    plt.figure()
    plt.plot(acceptance)
    plt.xlabel("Sample")
    plt.ylabel("Local Acceptance Rate")
    plt.title(f"Production Local Acceptance Rate for {event_name}")
    plot_path = os.path.join(plots_dir, "production_local_acceptance.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "production_global_acceptance"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def production_global_acceptance_plot(context: AssetExecutionContext):
    """
    Generate and save a plot of the global acceptance rate.
    """
    import matplotlib.pyplot as plt

    event_name = context.partition_key
    run_dir = os.path.join("data", event_name)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.npz")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    results = np.load(results_path, allow_pickle=True)
    acceptance = results["acceptance"].item().get("global", None)
    if acceptance is None:
        raise ValueError("No 'global' key found in acceptance.")
    plt.figure()
    plt.plot(acceptance)
    plt.xlabel("Sample")
    plt.ylabel("Global Acceptance Rate")
    plt.title(f"Production Global Acceptance Rate for {event_name}")
    plot_path = os.path.join(plots_dir, "production_global_acceptance.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path
