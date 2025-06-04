import argparse
from jimgw.run.single_event_run_manager import SingleEventRunManager
from jimgw.run.run_definition import RunDefinition

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute a single run of a JimGW run definition.")
    parser.add_argument("run_definition", type=str, help="Path to the run definition file.")
    args = parser.parse_args()

    # Load the run definition
    run_definition = RunDefinition.from_file(args.run_definition)

    # Create a SingleEventRunManager instance
    run_manager = SingleEventRunManager(run_definition)

    # Execute the sampling
    run_manager.sample()

    # Optionally, you can retrieve and print samples
    samples = run_manager.get_samples()
    print(samples)
