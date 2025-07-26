import dagster as dg
import gwosc
import os
import io
import numpy as np
from dagster import DynamicPartitionsDefinition, AssetExecutionContext
from RealDataCatalog.minio_resource import MinioResource
import matplotlib.pyplot as plt
import tempfile

# Create asset group for run and configuration0

event_partitions_def = DynamicPartitionsDefinition(name="event_name")


@dg.asset(
    key_prefix="RealDataCatalog",
    group_name="prerun",
    description="Fetch all confident events and their gps time",
)
def event_list(context: AssetExecutionContext, minio: MinioResource):

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
    buffer = io.StringIO()
    for name, gps_time in result:
        buffer.write(f"{name} {gps_time}\n")
    buffer.seek(0)
    data = buffer.getvalue().encode("utf-8")
    minio.put_object(
        object_name="event_list.txt",
        data=io.BytesIO(data),
        size=len(data),
        content_type="text/plain"
    )
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
def raw_data(context: AssetExecutionContext, minio: MinioResource):
    from jimgw.core.single_event.data import Data

    ifos = ["H1", "L1", "V1"]
    event_name = context.partition_key
    # Use minio to fetch event_list.txt instead of reading from local disk
    event_list_obj = minio.get_object("event_list.txt")
    lines = event_list_obj.read().decode("utf-8").splitlines()
    event_dict = dict(line.strip().split() for line in lines)
    gps_time = event_dict[event_name]
    start = float(gps_time) - 2
    end = float(gps_time) + 2
    # Use a temp directory, but keep event_name and "raw" part
    event_dir = os.path.join('tmp', event_name, "raw")
    os.makedirs(event_dir, exist_ok=True)
    for ifo in ifos:
        try:
            data = Data.from_gwosc(ifo, start, end)
            data_file_path = os.path.join(event_dir, f"{ifo}_data")
            data.to_file(data_file_path)
            # Upload raw data file to minio
            with open(data_file_path +'.npz', "rb") as f:
                file_size = os.path.getsize(data_file_path +'.npz')
                minio.put_object(
                    object_name=f"{event_name}/raw/{ifo}_data.npz",
                    data=f,
                    size=file_size,
                    content_type="application/x-npz",
                )
            os.remove(data_file_path + '.npz')

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
                psd_file_path = os.path.join(event_dir, f"{ifo}_psd")
                psd_data.to_file(psd_file_path)
                # Upload psd data file to minio
                with open(psd_file_path+'.npz', "rb") as f:
                    file_size = os.path.getsize(psd_file_path+'.npz')
                    minio.put_object(
                        object_name=f"{event_name}/raw/{ifo}_psd.npz",
                        data=f,
                        size=file_size,
                        content_type="application/x-npz",
                    )
            # Cleanup local files
            os.remove(psd_file_path + '.npz')
        except Exception as e:
            print(f"Error fetching data for {ifo} during {event_name}: {e}")
            continue


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "strain"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def raw_data_plot(context: AssetExecutionContext, minio: MinioResource):
    """
    Plot the raw strain data for each IFO for the event using Minio for data access and plot storage.
    """

    event_name = context.partition_key
    ifos = ["H1", "L1", "V1"]
    plot_paths = []
    for ifo in ifos:
        object_name = f"{event_name}/raw/{ifo}_data.npz"
        try:
            obj = minio.get_object(object_name)
            data = np.load(io.BytesIO(obj.read()))
            t = data["epoch"] + np.arange(data["td"].shape[0]) * data["dt"]
            td = data["td"]
            if t is not None and td is not None:
                plt.figure()
                plt.plot(t, td)
                plt.xlabel("Time (s)")
                plt.ylabel("Strain")
                plt.title(f"{ifo} Strain for {event_name}")
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    plt.savefig(tmpfile.name)
                    plt.close()
                    tmpfile.flush()
                    tmpfile.seek(0)
                    tmpfile_size = os.path.getsize(tmpfile.name)
                    minio_plot_path = f"{event_name}/plots/{ifo}_strain.png"
                    with open(tmpfile.name, "rb") as plotfile:
                        minio.put_object(
                            object_name=minio_plot_path,
                            data=plotfile,
                            size=tmpfile_size,
                            content_type="image/png",
                        )
                    plot_paths.append(minio_plot_path)
                os.remove(tmpfile.name)
        except Exception as e:
            print(f"Error processing {ifo} for {event_name}: {e}")
            continue
    return plot_paths


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "psd"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def psd_plot(context: AssetExecutionContext, minio: MinioResource):
    """
    Plot the PSD for each IFO for the event using Minio for data access and plot storage.
    """


    event_name = context.partition_key
    ifos = ["H1", "L1", "V1"]
    plot_paths = []
    for ifo in ifos:
        object_name = f"{event_name}/raw/{ifo}_psd.npz"
        try:
            obj = minio.get_object(object_name)
            data = np.load(io.BytesIO(obj.read()))
            f = data["frequencies"]
            psd = data["values"]
            plt.figure()
            plt.loglog(f, psd)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD")
            plt.title(f"{ifo} PSD for {event_name}")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plt.savefig(tmpfile.name)
                plt.close()
                tmpfile.flush()
                tmpfile.seek(0)
                tmpfile_size = os.path.getsize(tmpfile.name)
                minio_plot_path = f"{event_name}/plots/{ifo}_psd.png"
                with open(tmpfile.name, "rb") as plotfile:
                    minio.put_object(
                        object_name=minio_plot_path,
                        data=plotfile,
                        size=tmpfile_size,
                        content_type="image/png",
                    )
                plot_paths.append(minio_plot_path)
            os.remove(tmpfile.name)
        except Exception as e:
            print(f"Error processing {ifo} for {event_name}: {e}")
            continue
    return plot_paths


@dg.asset(
    key_prefix="RealDataCatalog",
    group_name="prerun",
    description="Configuration file for the run.",
    deps=[raw_data],
    partitions_def=event_partitions_def,
)
def config_file(context: AssetExecutionContext, minio: MinioResource):
    from jimgw.run.library.IMRPhenomPv2_standard_cbc import (
        IMRPhenomPv2StandardCBCRunDefinition,
    )
    event_name = context.partition_key
    # Read event_list.txt from Minio
    event_list_obj = minio.get_object("event_list.txt")
    lines = event_list_obj.read().decode("utf-8").splitlines()
    event_dict = dict(line.strip().split() for line in lines)
    gps_time = float(event_dict[event_name])

    # Check which IFOs have both data and PSD files present in Minio
    available_ifos: list[str] = []
    for ifo in ["H1", "L1", "V1"]:
        data_obj_name = f"{event_name}/raw/{ifo}_data.npz"
        psd_obj_name = f"{event_name}/raw/{ifo}_psd.npz"
        try:
            # Try to fetch both objects from Minio
            minio.get_object(data_obj_name)
            minio.get_object(psd_obj_name)
            available_ifos.append(ifo)
        except Exception:
            continue

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
        t_c_range=(-0.05, 0.05),
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
    run_dir = f"{event_name}/"
    run.working_dir = run_dir
    run.seed = hash(int(gps_time)) % (2**32 - 1)
    run.local_data_prefix = f"{run_dir}raw/"
    # Serialize config.yaml to a temp file, then upload to Minio
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmpfile:
        run.serialize(tmpfile.name)
        tmpfile.flush()
        tmpfile.seek(0)
        tmpfile_size = os.path.getsize(tmpfile.name)
        minio_config_path = f"{event_name}/config.yaml"
        with open(tmpfile.name, "rb") as configfile:
            minio.put_object(
                object_name=minio_config_path,
                data=configfile,
                size=tmpfile_size,
                content_type="application/x-yaml",
            )
    os.remove(tmpfile.name)


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
def loss_plot(context: AssetExecutionContext, minio: MinioResource):
    """
    Generate and save a loss plot from the training_loss asset using Minio.
    """

    event_name = context.partition_key
    results_obj = minio.get_object(f"{event_name}/results.npz")
    results = np.load(io.BytesIO(results_obj.read()), allow_pickle=True)
    loss = results["loss_data"]
    if loss is None:
        raise ValueError("No 'loss' key found in loss_data.")
    plt.figure()
    plt.plot(loss)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss for {event_name}")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name)
        plt.close()
        tmpfile.flush()
        tmpfile.seek(0)
        tmpfile_size = os.path.getsize(tmpfile.name)
        minio_plot_path = f"{event_name}/plots/training_loss.png"
        with open(tmpfile.name, "rb") as plotfile:
            minio.put_object(
                object_name=minio_plot_path,
                data=plotfile,
                size=tmpfile_size,
                content_type="image/png",
            )
    os.remove(tmpfile.name)
    return minio_plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "production_chains"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def production_chains_corner_plot(context: AssetExecutionContext, minio: MinioResource):
    """
    Generate and save a corner plot from the production_chains asset using Minio.
    """
    import corner

    event_name = context.partition_key
    results_obj = minio.get_object(f"{event_name}/results.npz")
    results = np.load(io.BytesIO(results_obj.read()), allow_pickle=True)
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
    samples = np.array([chains[key] for key in keys]).T
    fig = corner.corner(samples[::10], labels=keys)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.savefig(tmpfile.name)
        plt.close(fig)
        tmpfile.flush()
        tmpfile.seek(0)
        tmpfile_size = os.path.getsize(tmpfile.name)
        minio_plot_path = f"{event_name}/plots/production_chains_corner.png"
        with open(tmpfile.name, "rb") as plotfile:
            minio.put_object(
                object_name=minio_plot_path,
                data=plotfile,
                size=tmpfile_size,
                content_type="image/png",
            )
    os.remove(tmpfile.name)
    return minio_plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "auxiliary_nf_samples"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def nf_samples_corner_plot(context: AssetExecutionContext, minio: MinioResource):
    """
    Generate and save a corner plot from the auxiliary_nf_samples asset using Minio.
    """
    import corner

    event_name = context.partition_key
    results_obj = minio.get_object(f"{event_name}/results.npz")
    results = np.load(io.BytesIO(results_obj.read()), allow_pickle=True)
    nf_samples = results["nf_samples"].item()
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
    fig = corner.corner(nf_samples, labels=keys)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.savefig(tmpfile.name)
        plt.close(fig)
        tmpfile.flush()
        tmpfile.seek(0)
        tmpfile_size = os.path.getsize(tmpfile.name)
        minio_plot_path = f"{event_name}/plots/nf_samples_corner.png"
        with open(tmpfile.name, "rb") as plotfile:
            minio.put_object(
                object_name=minio_plot_path,
                data=plotfile,
                size=tmpfile_size,
                content_type="image/png",
            )
    os.remove(tmpfile.name)
    return minio_plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "auxiliary_prior_samples"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def prior_samples_corner_plot(context: AssetExecutionContext, minio: MinioResource):
    """
    Generate and save a corner plot from the auxiliary_prior_samples asset using Minio.
    """
    import corner

    event_name = context.partition_key
    results_obj = minio.get_object(f"{event_name}/results.npz")
    results = np.load(io.BytesIO(results_obj.read()), allow_pickle=True)
    prior_samples = results["prior_samples"].item()
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
    fig = corner.corner(prior_samples, labels=keys)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.savefig(tmpfile.name)
        plt.close(fig)
        tmpfile.flush()
        tmpfile.seek(0)
        tmpfile_size = os.path.getsize(tmpfile.name)
        minio_plot_path = f"{event_name}/plots/prior_samples_corner.png"
        with open(tmpfile.name, "rb") as plotfile:
            minio.put_object(
                object_name=minio_plot_path,
                data=plotfile,
                size=tmpfile_size,
                content_type="image/png",
            )
    os.remove(tmpfile.name)
    return minio_plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "production_chains"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def production_chains_trace_plot(context: AssetExecutionContext, minio: MinioResource):
    """
    Generate and save a trace plot for the production chains using Minio.
    """

    event_name = context.partition_key
    results_obj = minio.get_object(f"{event_name}/results.npz")
    results = np.load(io.BytesIO(results_obj.read()), allow_pickle=True)
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
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name)
        plt.close()
        tmpfile.flush()
        tmpfile.seek(0)
        tmpfile_size = os.path.getsize(tmpfile.name)
        minio_plot_path = f"{event_name}/plots/production_chains_trace.png"
        with open(tmpfile.name, "rb") as plotfile:
            minio.put_object(
                object_name=minio_plot_path,
                data=plotfile,
                size=tmpfile_size,
                content_type="image/png",
            )
    os.remove(tmpfile.name)
    return minio_plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "production_log_prob"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def production_log_prob_distribution(context: AssetExecutionContext, minio: MinioResource):
    """
    Generate and save a histogram of the production log probability using Minio.
    """

    event_name = context.partition_key
    results_obj = minio.get_object(f"{event_name}/results.npz")
    results = np.load(io.BytesIO(results_obj.read()), allow_pickle=True)
    log_prob = results["log_probs"]
    if log_prob is None:
        raise ValueError("No 'log_prob' key found in loss_data.")
    plt.figure()
    plt.hist(log_prob.flatten(), bins=50, alpha=0.7)
    plt.xlabel("Log Probability")
    plt.ylabel("Frequency")
    plt.title(f"Production Log Probability Distribution for {event_name}")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name)
        plt.close()
        tmpfile.flush()
        tmpfile.seek(0)
        tmpfile_size = os.path.getsize(tmpfile.name)
        minio_plot_path = f"{event_name}/plots/production_log_prob_distribution.png"
        with open(tmpfile.name, "rb") as plotfile:
            minio.put_object(
                object_name=minio_plot_path,
                data=plotfile,
                size=tmpfile_size,
                content_type="image/png",
            )
    os.remove(tmpfile.name)
    return minio_plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "production_log_prob"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def production_log_prob_evolution(context: AssetExecutionContext, minio: MinioResource):
    """
    Generate and save a plot of the evolution of the production log probability using Minio.
    """

    event_name = context.partition_key
    results_obj = minio.get_object(f"{event_name}/results.npz")
    results = np.load(io.BytesIO(results_obj.read()), allow_pickle=True)
    log_prob = results["log_probs"]
    if log_prob is None:
        raise ValueError("No 'log_prob' key found in loss_data.")
    plt.figure()
    plt.plot(log_prob)
    plt.xlabel("Sample")
    plt.ylabel("Log Probability")
    plt.title(f"Production Log Probability Evolution for {event_name}")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name)
        plt.close()
        tmpfile.flush()
        tmpfile.seek(0)
        tmpfile_size = os.path.getsize(tmpfile.name)
        minio_plot_path = f"{event_name}/plots/production_log_prob_evolution.png"
        with open(tmpfile.name, "rb") as plotfile:
            minio.put_object(
                object_name=minio_plot_path,
                data=plotfile,
                size=tmpfile_size,
                content_type="image/png",
            )
    os.remove(tmpfile.name)
    return minio_plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "production_local_acceptance"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def production_local_acceptance_plot(context: AssetExecutionContext, minio: MinioResource):
    """
    Generate and save a plot of the local acceptance rate using Minio.
    """

    event_name = context.partition_key
    results_obj = minio.get_object(f"{event_name}/results.npz")
    results = np.load(io.BytesIO(results_obj.read()), allow_pickle=True)
    acceptance = results["acceptance"].item().get("local", None)
    if acceptance is None:
        raise ValueError("No 'local' key found in acceptance.")
    plt.figure()
    plt.plot(acceptance)
    plt.xlabel("Sample")
    plt.ylabel("Local Acceptance Rate")
    plt.title(f"Production Local Acceptance Rate for {event_name}")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name)
        plt.close()
        tmpfile.flush()
        tmpfile.seek(0)
        tmpfile_size = os.path.getsize(tmpfile.name)
        minio_plot_path = f"{event_name}/plots/production_local_acceptance.png"
        with open(tmpfile.name, "rb") as plotfile:
            minio.put_object(
                object_name=minio_plot_path,
                data=plotfile,
                size=tmpfile_size,
                content_type="image/png",
            )
    os.remove(tmpfile.name)
    return minio_plot_path


@dg.asset(
    group_name="diagnostics",
    deps=[["RealDataCatalog", "production_global_acceptance"]],
    key_prefix="RealDataCatalog",
    partitions_def=event_partitions_def,
)
def production_global_acceptance_plot(context: AssetExecutionContext, minio: MinioResource):
    """
    Generate and save a plot of the global acceptance rate using Minio.
    """

    event_name = context.partition_key
    results_obj = minio.get_object(f"{event_name}/results.npz")
    results = np.load(io.BytesIO(results_obj.read()), allow_pickle=True)
    acceptance = results["acceptance"].item().get("global", None)
    if acceptance is None:
        raise ValueError("No 'global' key found in acceptance.")
    plt.figure()
    plt.plot(acceptance)
    plt.xlabel("Sample")
    plt.ylabel("Global Acceptance Rate")
    plt.title(f"Production Global Acceptance Rate for {event_name}")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name)
        plt.close()
        tmpfile.flush()
        tmpfile.seek(0)
        tmpfile_size = os.path.getsize(tmpfile.name)
        minio_plot_path = f"{event_name}/plots/production_global_acceptance.png"
        with open(tmpfile.name, "rb") as plotfile:
            minio.put_object(
                object_name=minio_plot_path,
                data=plotfile,
                size=tmpfile_size,
                content_type="image/png",
            )
    os.remove(tmpfile.name)
    return minio_plot_path
