from jimgw.base import RunManager
from dataclasses import dataclass
from jimgw.single_event import likelihood
from jimgw import prior
from jimgw.jim import Jim


@dataclass
class SingleEventRun:
    seed: int
    path: str

    detectors: list[str]
    priors: dict[
        str, dict[str, str | float | int | bool]
    ]  # TODO: Incorporate custom transform
    waveform: str
    waveform_parameters: dict[str, str | float | int | bool]
    jim_parameters: dict[str, str | float | int | bool]
    likelihood_parameters: dict[str, str | float | int | bool]
    trigger_time: float
    duration: int
    post_trigger_duration: int
    fmin: float
    fmax: float
    injection_parameters: dict[str, float]


class SingleEventPERunManager(RunManager):
    run: SingleEventRun
    jim: Jim

    @property
    def waveform(self):
        return self.run.waveform

    @property
    def detectors(self):
        return self.run.detectors

    @property
    def data(self):
        return [detector.data for detector in self.likelihood.detectors]

    @property
    def psds(self):
        return self.run.detectors

    def __init__(self, **kwargs):
        if "run" in kwargs:
            print("Run instance provided. Loading from instance.")
            self.run = kwargs["run"]
        elif "path" in kwargs:
            print("Run instance not provided. Loading from path.")
            self.run = self.load_from_path(kwargs["path"])
        else:
            print("Neither run instance nor path provided.")
            raise ValueError

        local_likelihood = self.initialize_likelihood()
        local_prior = self.initialize_prior()
        self.jim = Jim(local_likelihood, local_prior, **self.run.jim_parameters)

    def log_metadata(self):
        pass

    def summarize(self):
        pass

    def save(self, path: str):
        pass

    def load_from_path(self, path: str) -> SingleEventRun:
        raise NotImplementedError

    def initialize_likelihood(self) -> likelihood.TransientLikelihoodFD:
        raise NotImplementedError

    def initialize_prior(self) -> prior.Prior:
        raise NotImplementedError

    def fetch_data(self):
        """
        Given a run config that specify using real data, fetch the data from the server.


        """
        try:
            pass
        except Exception as e:
            raise e

    def generate_data(self):
        """
        Given a run config that specify using simulated data, generate the data.
        """
        try:
            pass
        except Exception as e:
            raise e

    def initialize_detector(self):
        """
        Initialize the detectors.
        """
        try:
            pass
        except Exception as e:
            raise e

    def initialize_waveform(self):
        """
        Initialize the waveform.
        """
        try:
            pass
        except Exception as e:
            raise e
