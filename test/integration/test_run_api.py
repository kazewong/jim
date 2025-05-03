from jimgw.run.library.IMRPhenomPv2_standard_cbc import TestIMRPhenomPv2StandardCBCRun
from jimgw.run.single_event_run_manager import SingleEventRunManager

run_manager = SingleEventRunManager(
    TestIMRPhenomPv2StandardCBCRun(),
    n_training_loops=1,
    n_epochs=1,
    n_production_loops=1,

)
print(run_manager.run)
run_manager.sample()
