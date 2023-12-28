from jimgw.base import RunManager
from jimgw.prior import Prior
from jimgw.jim import Jim
from jimgw.single_event.likelihood import SingleEventLiklihood
from dataclasses import dataclass


@dataclass
class SingleEventRun:
    seed: int
    waveform: dict[str, str | float | int | bool]
    detectors: list[str]
    data: list[str]
    psds: list[str]
    priors: list[str]
    jim_parameters: dict[str, str | float | int | bool]
    gps_time: int
    duration: int
    post_trigger_duration: int
    fmin: float
    fmax: float


class SingleEventPERunManager(RunManager):
    likelihood: SingleEventLiklihood

    @property
    def waveform(self):
        return self.likelihood.waveform

    @property
    def detectors(self):
        return self.likelihood.detectors

    @property
    def data(self):
        return [detector.data for detector in self.likelihood.detectors]

    @property
    def psds(self):
        return [detector.psd for detector in self.likelihood.detectors]

    def __init__(self, *args, **kwargs):
        if "run_file" in kwargs:
            print("Run file provided. Loading from file.")
            self.load(kwargs["run_file"])
        elif "likelihood" in kwargs and "prior" in kwargs and "jim" not in kwargs:
            print("Loading from provided likelihood, prior and jim instances.")
            assert isinstance(
                kwargs["likelihood"], SingleEventLiklihood
            ), "Likelihood must be a SingleEventLikelihood instance."
            assert isinstance(kwargs["prior"], Prior), "Prior must be a Prior instance."
            assert isinstance(kwargs["jim"], Jim), "Jim must be a Jim instance."

            self.likelihood = kwargs["likelihood"]
            self.prior = kwargs["prior"]
            self.jim = kwargs["jim"]
        else:
            raise ValueError(
                "Please provide a run file or a likelihood, prior and jim instances."
            )

    def log_metadata(self):
        pass

    def summarize(self):
        pass

    def save(self):
        pass

    def load(self, path: str):
        pass
