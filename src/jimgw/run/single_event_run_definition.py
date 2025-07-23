from jimgw.run.run_definition import RunDefinition
from jimgw.core.single_event.detector import GroundBased2G
from typing import Optional, Sequence
from jimgw.core.single_event.detector import get_detector_preset


class SingleEventRunDefinition(RunDefinition):
    """
    A `SingleEventRun` is a template of priors, likelihood transforms, and sample transforms for single event analysis.

    It is aimed to be an abstraction which wrap the flexible but complicated APIs of core jim into an object that the users only interact with the underlying `jim` through the parameters defined in the Run. It is responsible for constructing the likelihood object, the prior, sample_transform, and likelihood_transform needed in jim.

    The most important property of a Run instance is it needs to be able to deterministically declared. All arguments to a run has to be explicitly provided, and the content of a Run should be exactly the same given the same arguments.
    """

    # Likelihood parameters
    gps: float  # GPS time of the event trigger
    segment_length: float  # Length of the segment
    post_trigger_length: float  # Length of segment after the trigger
    f_min: float  # Minimum frequency
    f_max: float  # Maximum frequency
    ifos: Sequence[GroundBased2G]  # Set of detectors
    f_ref: float  # Reference frequency
    local_data_prefix: Optional[str] = None

    def __init__(
        self,
        gps: float,
        segment_length: float,
        post_trigger_length: float,
        f_min: float,
        f_max: float,
        ifos: set[str],
        f_ref: float,
        injection_data_prefix: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gps = gps
        self.segment_length = segment_length
        self.post_trigger_length = post_trigger_length
        self.f_min = f_min
        self.f_max = f_max
        self.ifos = [get_detector_preset()[ifo] for ifo in ifos]
        self.f_ref = f_ref
        self.local_data_prefix = injection_data_prefix

    def load_single_event_params(self, inputs: dict):
        """Load the single event parameters into the Run object."""
        self.gps = inputs.get("gps", self.gps)
        self.segment_length = inputs.get("segment_length", self.segment_length)
        self.post_trigger_length = inputs.get(
            "post_trigger_length", self.post_trigger_length
        )
        self.f_min = inputs.get("f_min", self.f_min)
        self.f_max = inputs.get("f_max", self.f_max)
        ifos_input = inputs.get("ifos", [ifo.name for ifo in self.ifos])
        self.ifos = [get_detector_preset()[ifo] for ifo in ifos_input]
        self.f_ref = inputs.get("f_ref", self.f_ref)
        self.local_data_prefix = inputs.get("local_data_prefix", self.local_data_prefix)

    def serialize(self, path: str = "./") -> dict:
        """Serialize a `SingleEventRun` object into a human readable config file."""
        run_dict = super().serialize(path)
        run_dict.update(
            {
                "gps": self.gps,
                "segment_length": self.segment_length,
                "post_trigger_length": self.post_trigger_length,
                "f_min": self.f_min,
                "f_max": self.f_max,
                "ifos": [ifo.name for ifo in self.ifos],
                "f_ref": self.f_ref,
                "local_data_prefix": self.local_data_prefix,
            }
        )
        return run_dict
