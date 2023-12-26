from jaxtyping import Array, Float

from jimgw.base import RunManager
from jimgw.prior import Prior
from jimgw.jim import Jim
from jimgw.single_event.likelihood import SingleEventLiklihood


class SingleEventPERunManager(RunManager):
    time: Float[Array, " n_sample"]
    data: Float[Array, " n_sample"]
    psd: Float[Array, " n_sample"]

    @property
    def waveform(self):
        return self.likelihood

    def __init__(self, *args, **kwargs):
        if "run_file" in kwargs:
            print("Run file provided. Loading from file.")
            self.load(kwargs["run_file"])
        elif "likelihood" in kwargs and "prior" in kwargs and "jim" not in kwargs:
            print("Loading from provided likelihood, prior and jim instances.")
            self.likelihood = kwargs["likelihood"]
            self.prior = kwargs["prior"]
            self.jim = kwargs["jim"]
        else:
            raise ValueError(
                "Please provide a run file or a likelihood, prior and jim instances."
            )

        assert isinstance(
            self.likelihood, SingleEventLiklihood
        ), "Likelihood must be a SingleEventLikelihood instance."
        assert isinstance(self.prior, Prior), "Prior must be a Prior instance."
        assert isinstance(self.jim, Jim), "Jim must be a Jim instance."

    def log_metadata(self):
        pass

    def summarize(self):
        pass

    def save(self):
        pass

    def load(self, path: str):
        pass
