import dagster as dg

from jimgw.run_manager.single_event_run import SingleEventRun


# Create asset group for run and configuration0

@dg.asset
def data_strain():
    pass

@dg.asset
def psd():
    pass

@dg.asset
def config_file():
    pass

@dg.asset
def imrphenom_pv2_standard_cbc_run():
    """
    This is a placeholder function for the IMRPhenomPv2StandardCBCRun class.
    It is used to demonstrate how to create a Dagster asset.
    """
    pass

# Create asset group for diagnostics

@dg.multi_asset(specs=[dg.AssetSpec("")])

@dg.asset
def loss_plot():
    pass

@dg.asset
def sample_corner():
    pass

@dg.asset
def nf_sample_corner():
    pass

@dg.asset
def prior