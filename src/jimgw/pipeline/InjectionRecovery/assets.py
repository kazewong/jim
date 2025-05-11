import dagster as dg



# Sample a fiducial population


@dg.asset
def sample_population():
    """
    This is a placeholder function for the fiducial population.
    It is used to demonstrate how to create a Dagster asset.
    """
    pass


@dg.asset
def generate_configs():
    """
    This is a placeholder function for the generation of configuration files.
    It is used to demonstrate how to create a Dagster asset.
    """
    pass


# Create asset group for run and configuration0


@dg.asset(
    group_name="prerun",
    description="Configuration file for the run.",
)
def config_file():
    pass


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
    pass


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
