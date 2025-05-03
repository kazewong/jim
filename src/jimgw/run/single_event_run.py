from jimgw.run.run import Run
from jimgw.core.single_event.detector import GroundBased2G, H1, L1, detector_preset
from jimgw.core.single_event.likelihood import ZeroLikelihood
from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
)
from typing import Sequence
import yaml


class SingleEventRun(Run):
    """
    A `SingleEventRun` is a template of priors, likelihood transforms, and sample transforms for single event analysis.

    It is aimed to be an abstraction which wrap the flexible but complicated APIs of core jim into an object that the users only interact with the underlying `jim` through the parameters defined in the Run. It is responsible for constructing the likelihood object, the prior, sample_transform, and likelihood_transform needed in jim.

    The most important property of a Run instance is it needs to be able to deterministically declared. All arguments to a run has to be explicitly provided, and the content of a Run should be exactly the same given the same arguments.
    """

    # Likelihood parameters
    gps: int  # GPS time of the event trigger
    segment_length: int  # Length of the segment
    post_trigger_length: int  # Length of segment after the trigger
    f_min: int  # Minimum frequency
    f_max: int  # Maximum frequency
    ifos: Sequence[GroundBased2G]  # Set of detectors
    f_ref: int  # Reference frequency


class TestSingleEventRun(SingleEventRun):
    """
    A test run with zero likelihood
    """

    def __init__(self):
        self.gps = 1234567890
        self.segment_length = 10
        self.post_trigger_length = 5
        self.f_min = 20
        self.f_max = 2000
        self.ifos = [H1, L1]
        self.f_ref = 100

        self.likelihood = ZeroLikelihood()
        self.prior = CombinePrior(
            [
                UniformPrior(1, 100,["mass1"]),
                UniformPrior(1, 100,["mass2"]),
            ]
        )
        self.sample_transforms = []
        self.likelihood_transforms = []

    def serialize(self, path: str = "./"):
        """
        Serialize a `Run` object into a human readble config file.
        """
        run_dict = {
            "gps": self.gps,
            "segment_length": self.segment_length,
            "post_trigger_length": self.post_trigger_length,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "ifos": [ifo.name for ifo in self.ifos],
            "f_ref": self.f_ref,
        }
        with open(path, "w") as f:
            yaml.dump(run_dict, f)
    
    @classmethod
    def deserialize(cls, path: str) -> SingleEventRun:
        """
        Deserialize a config file into a `Run` object
        """
        with open(path, "r") as f:
            run_dict = yaml.safe_load(f)
        run = cls()
        run.gps = run_dict["gps"]
        run.segment_length = run_dict["segment_length"]
        run.post_trigger_length = run_dict["post_trigger_length"]
        run.f_min = run_dict["f_min"]
        run.f_max = run_dict["f_max"]
        run.ifos = [detector_preset[ifo] for ifo in run_dict["ifos"]]
        run.f_ref = run_dict["f_ref"]
        return run