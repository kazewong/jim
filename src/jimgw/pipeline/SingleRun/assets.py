import dagster as dg

from jimgw.run.single_event_run_definition import SingleEventRunDefinition


# Create asset group for run and configuration0


@dg.asset(
    key_prefix="SingleRun",
    group_name="prerun",
    description="Configuration file for the run.",
)
def config_file():
    pass


@dg.multi_asset(
    specs=[
        dg.AssetSpec("SingleRun_strain", deps=[config_file]),
        dg.AssetSpec("SingleRun_psd", deps=[config_file]),
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
        dg.AssetSpec("SingleRun_training_chains", deps=[raw_data]),
        dg.AssetSpec("SingleRun_training_log_prob", deps=[raw_data]),
        dg.AssetSpec("SingleRun_training_local_acceptance", deps=[raw_data]),
        dg.AssetSpec("SingleRun_training_global_acceptance", deps=[raw_data]),
        dg.AssetSpec("SingleRun_training_loss", deps=[raw_data]),
        dg.AssetSpec("SingleRun_production_chains", deps=[raw_data]),
        dg.AssetSpec("SingleRun_production_log_prob", deps=[raw_data]),
        dg.AssetSpec("SingleRun_production_local_acceptance", deps=[raw_data]),
        dg.AssetSpec("SingleRun_production_global_acceptance", deps=[raw_data]),
        dg.AssetSpec("SingleRun_auxiliary_nf_samples", deps=[raw_data]),
        dg.AssetSpec("SingleRun_auxiliary_prior_samples", deps=[raw_data]),
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
    group_name="diagnostics", deps=["SingleRun_training_loss"], key_prefix="SingleRun"
)
def loss_plot():
    pass


@dg.asset(
    group_name="diagnostics", deps=["SingleRun_training_chains"], key_prefix="SingleRun"
)
def training_chains_corner_plot():
    pass


@dg.asset(
    group_name="diagnostics", deps=["SingleRun_training_chains"], key_prefix="SingleRun"
)
def training_chains_trace_plot():
    pass


@dg.asset(
    group_name="diagnostics", deps=["SingleRun_training_chains"], key_prefix="SingleRun"
)
def training_chains_rhat_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["SingleRun_training_log_prob"],
    key_prefix="SingleRun",
)
def training_log_prob_distribution():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["SingleRun_training_log_prob"],
    key_prefix="SingleRun",
)
def training_log_prob_evolution():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["SingleRun_training_local_acceptance"],
    key_prefix="SingleRun",
)
def training_local_acceptance_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["SingleRun_training_global_acceptance"],
    key_prefix="SingleRun",
)
def training_global_acceptance_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["SingleRun_production_chains"],
    key_prefix="SingleRun",
)
def production_chains_corner_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["SingleRun_production_chains"],
    key_prefix="SingleRun",
)
def production_chains_trace_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["SingleRun_production_chains"],
    key_prefix="SingleRun",
)
def production_chains_rhat_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["SingleRun_production_log_prob"],
    key_prefix="SingleRun",
)
def production_log_prob_distribution():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["SingleRun_production_log_prob"],
    key_prefix="SingleRun",
)
def production_log_prob_evolution():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["SingleRun_production_local_acceptance"],
    key_prefix="SingleRun",
)
def production_local_acceptance_plot():
    pass


@dg.asset(
    group_name="diagnostics",
    deps=["SingleRun_production_global_acceptance"],
    key_prefix="SingleRun",
)
def production_global_acceptance_plot():
    pass
