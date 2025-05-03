from jimgw.run.single_event_run import TestSingleEventRun
from jimgw.run.single_event_run_manager import SingleEventRunManager

run_manager = SingleEventRunManager(TestSingleEventRun())
print(run_manager.run)