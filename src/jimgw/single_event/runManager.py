from jimgw.base import RunManager
from dataclasses import dataclass, field
from jimgw.single_event.likelihood import likelihood_presets, SingleEventLiklihood
from jimgw.single_event.detector import detector_preset, Detector
from jimgw.single_event.waveform import waveform_preset, Waveform
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
    jim_parameters: dict[str, str | float | int | bool]
    injection_parameters: dict[str, float]
    injection: bool = False
    likelihood_parameters: dict[str, str | float | int | bool] = field(
        default_factory=lambda: {"name": "TransientLikelihoodFD"}
    )
    waveform_parameters: dict[str, str | float | int | bool] = field(
        default_factory=lambda: {"name": ""}
    )
    data_parameters: dict[str, float | int] = field(
        default_factory=lambda: {
            "trigger_time": 0.0,
            "duration": 0,
            "post_trigger_duration": 0,
            "f_min": 0.0,
            "f_max": 0.0,
            "tukey_alpha": 0.2,
        }
    )


class SingleEventPERunManager(RunManager):
    run: SingleEventRun
    jim: Jim

    @property
    def waveform(self):
        return self.run.waveform_parameters["name"]

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

    def initialize_likelihood(self) -> SingleEventLiklihood:
        detectors = self.initialize_detector()
        waveform = self.initialize_waveform()
        name = self.run.likelihood_parameters["name"]
        assert isinstance(name, str), "Likelihood name must be a string."
        return likelihood_presets[name](
            detectors, waveform, **self.run.likelihood_parameters
        )

    def initialize_prior(self) -> prior.Prior:
        raise NotImplementedError

    def initialize_detector(self) -> list[Detector]:
        """
        Initialize the detectors.
        """
        print("Initializing detectors.")
        trigger_time = self.run.data_parameters["trigger_time"]
        duration = self.run.data_parameters["duration"]
        post_trigger_duration = self.run.data_parameters["post_trigger_duration"]
        f_min = self.run.data_parameters["f_min"]
        f_max = self.run.data_parameters["f_max"]
        tukey_alpha = self.run.data_parameters["tukey_alpha"]

        assert trigger_time >= 0, "Trigger time must be positive."
        assert duration > 0, "Duration must be positive."
        assert post_trigger_duration >= 0, "Post trigger duration must be positive."
        assert f_min >= 0, "f_min must be positive."
        assert f_max > f_min, "f_max must be greater than f_min."
        assert 0 <= tukey_alpha <= 1, "Tukey alpha must be between 0 and 1."

        epoch = duration - post_trigger_duration
        detectors = []
        for name in self.run.detectors:
            detector = detector_preset[name]
            if not self.run.injection:
                print("Loading real data.")
                detector.load_data(
                    trigger_time=trigger_time,
                    gps_start_pad=int(epoch),
                    gps_end_pad=int(post_trigger_duration),
                    psd_pad=int(duration * 4),
                    f_min=f_min,
                    f_max=f_max,
                    tukey_alpha=tukey_alpha,
                )
            else:
                print("Injection mode. Need to wait until waveform model is loaded.")
            detectors.append(detector_preset[name])
        return detectors

    def initialize_waveform(self) -> Waveform:
        """
        Initialize the waveform.
        """
        print("Initializing waveform.")
        name = self.run.waveform_parameters["name"]
        if name not in waveform_preset:
            raise ValueError(f"Waveform {name} not recognized.")
        waveform = waveform_preset[name](**self.run.waveform_parameters)
        return waveform
