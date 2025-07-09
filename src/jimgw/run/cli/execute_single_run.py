import argparse
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision

from jimgw.run.run_definition import RunDefinition
from jimgw.run.single_event_run_manager import SingleEventRunManager
from jimgw.run.library.class_definitions import AvailableDefinitions
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute a single run of a JimGW run definition."
    )
    parser.add_argument(
        "run_definition", type=str, help="Path to the run definition file."
    )
    args = parser.parse_args()

    assert args.run_definition.endswith(
        ".yaml"
    ), "Run definition file must be a YAML file."

    definitions_name = yaml.safe_load(open(args.run_definition))["definition_name"]

    assert issubclass(
        definition := AvailableDefinitions[definitions_name].value, RunDefinition
    ), f"Invalid run definition: {definitions_name}"

    # Load the run definition
    run_definition = definition.from_file(args.run_definition)

    # Create a SingleEventRunManager instance
    run_manager = SingleEventRunManager(run_definition)

    # # Execute the sampling
    run_manager.sample()

    chains = run_manager.get_chain_samples(training=False)
    log_probs = run_manager.get_log_prob(training=False)
    loss_data = run_manager.get_loss_data()
    nf_samples = run_manager.get_nf_samples()
    prior_samples = run_manager.get_prior_samples()
    acceptance = run_manager.get_acceptance_rates(training=False)

    np.savez(
        f"{run_manager.working_dir}/results",
        chains=chains,
        log_probs=log_probs,
        loss_data=loss_data,
        nf_samples=nf_samples,
        prior_samples=prior_samples,
        acceptance=acceptance,
    ) # type: ignore

