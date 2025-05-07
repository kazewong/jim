from jimgw.run.library.IMRPhenomPv2_standard_cbc import (
    TestIMRPhenomPv2StandardCBCRunDefinition,
)
from jimgw.run.single_event_run_manager import SingleEventRunManager

run_manager = SingleEventRunManager(
    TestIMRPhenomPv2StandardCBCRunDefinition(),
    n_chains=4,
    n_training_loops=1,
    n_epochs=10,
    n_production_loops=5,
)
print(run_manager.run)
run_manager.sample()

# Run all diagnostics
run_manager.plot_chains()
run_manager.plot_loss()
run_manager.plot_nf_sample()
run_manager.plot_prior()
run_manager.plot_acceptances()
summary = run_manager.generate_summary()
print("Sampling summary:", summary)
