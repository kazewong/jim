from jaxtyping import Float, Array

from jimgw.run.run_manager import RunManager


class SingleEventRunManager(RunManager):

    ### Utility functions ###

    def get_detector_waveform(self, params: dict[str, float]) -> tuple[
        Float[Array, " n_sample"],
        dict[str, Float[Array, " n_sample"]],
        dict[str, Float[Array, " n_sample"]],
    ]:
        raise NotImplementedError

    def plot_injection_waveform(self, path: str):
        raise NotImplementedError

    def plot_data(self, path: str):
        raise NotImplementedError
