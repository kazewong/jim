from jimgw.run.run_definition import RunDefinition
from jimgw.core.single_event.detector import GroundBased2G
from typing import Sequence


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
