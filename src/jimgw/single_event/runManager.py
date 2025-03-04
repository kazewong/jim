from dataclasses import asdict, dataclass, field
from typing import Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import corner
import numpy as np
import yaml
from astropy.time import Time
from jaxlib.xla_extension import ArrayImpl
from jaxtyping import Array, Float, PyTree

from jimgw import prior
from jimgw.base import RunManager
from jimgw.jim import Jim
from jimgw.single_event.detector import Detector, detector_preset
from jimgw.single_event.likelihood import SingleEventLiklihood, likelihood_presets
from jimgw.single_event.waveform import Waveform, waveform_preset


def jaxarray_representer(dumper: yaml.Dumper, data: ArrayImpl):
    return dumper.represent_list(data.tolist())


yaml.add_representer(ArrayImpl, jaxarray_representer)  # type: ignore

prior_presets = {
    "Unconstrained_Uniform": prior.Unconstrained_Uniform,
    "Uniform": prior.Uniform,
    "Sphere": prior.Sphere,
    "AlignedSpin": prior.AlignedSpin,
    "PowerLaw": prior.PowerLaw,
    "Composite": prior.Composite,
    "MassRatio": lambda **kwargs: prior.Uniform(
        0.125,
        1.0,
        naming=["q"],
        transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
    ),
    "CosIota": lambda **kwargs: prior.Uniform(
        -1.0,
        1.0,
        naming=["cos_iota"],
        transforms={
            "cos_iota": (
                "iota",
                lambda params: jnp.arccos(params["cos_iota"]),
            )
        },
    ),
    "SinDec": lambda **kwargs: prior.Uniform(
        -1.0,
        1.0,
        naming=["sin_dec"],
        transforms={
            "sin_dec": (
                "dec",
                lambda params: jnp.arcsin(params["sin_dec"]),
            )
        },
    ),
    "EarthFrame": prior.EarthFrame,
}


@dataclass
class SingleEventRun:
    seed: int

    detectors: list[str]
    priors: dict[
        str, dict[str, Union[str, float, int, bool]]
    ]  # Transform cannot be included in this way, add it to preset if used often.
    jim_parameters: dict[str, Union[str, float, int, bool, dict]]
    path: str = "./experiment"
    injection_parameters: dict[str, float] = field(default_factory=lambda: {})
    injection: bool = False
    likelihood_parameters: dict[str, Union[str, float, int, bool, PyTree]] = field(
        default_factory=lambda: {"name": "TransientLikelihoodFD"}
    )
    waveform_parameters: dict[str, Union[str, float, int, bool]] = field(
        default_factory=lambda: {"name": ""}
    )
    data_parameters: dict[str, Union[float, int]] = field(
        default_factory=lambda: {
            "trigger_time": 0.0,
            "duration": 0,
            "post_trigger_duration": 0,
            "f_min": 0.0,
            "f_max": 0.0,
            "tukey_alpha": 0.2,
            "f_sampling": 4096.0,
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

        if self.run.injection and not self.run.injection_parameters:
            raise ValueError("Injection mode requires injection parameters.")

        local_prior = self.initialize_prior()
        local_likelihood = self.initialize_likelihood(local_prior)
        self.jim = Jim(local_likelihood, local_prior, **self.run.jim_parameters)

    def save(self, path: str):
        output_dict = asdict(self.run)
        with open(path + ".yaml", "w") as f:
            yaml.dump(output_dict, f, sort_keys=False)

    def load_from_path(self, path: str) -> SingleEventRun:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return SingleEventRun(**data)

    ### Initialization functions ###

    def initialize_likelihood(self, prior: prior.Prior) -> SingleEventLiklihood:
        """
        Since prior contains information about types, naming and ranges of parameters,
        some of the likelihood class require the prior to be initialized, such as the
        heterodyned likelihood.

        """
        detectors = self.initialize_detector()
        waveform = self.initialize_waveform()
        name = self.run.likelihood_parameters["name"]
        assert isinstance(name, str), "Likelihood name must be a string."
        assert name in likelihood_presets, f"Likelihood {name} not recognized."
        if self.run.injection:
            freqs = jnp.linspace(
                self.run.data_parameters["f_min"],
                self.run.data_parameters["f_sampling"] / 2,
                int(
                    self.run.data_parameters["f_sampling"]
                    * self.run.data_parameters["duration"]
                ),
            )
            freqs = freqs[
                (freqs >= self.run.data_parameters["f_min"])
                & (freqs <= self.run.data_parameters["f_max"])
            ]
            gmst = (
                Time(self.run.data_parameters["trigger_time"], format="gps")
                .sidereal_time("apparent", "greenwich")
                .rad
            )
            h_sky = waveform(freqs, self.run.injection_parameters)
            detector_parameters = {
                "ra": self.run.injection_parameters["ra"],
                "dec": self.run.injection_parameters["dec"],
                "psi": self.run.injection_parameters["psi"],
                "t_c": self.run.injection_parameters["t_c"],
                "gmst": gmst,
                "epoch": self.run.data_parameters["duration"]
                - self.run.data_parameters["post_trigger_duration"],
            }
            key, subkey = jax.random.split(jax.random.PRNGKey(self.run.seed + 1901))
            for detector in detectors:
                detector.inject_signal(subkey, freqs, h_sky, detector_parameters)  # type: ignore
                key, subkey = jax.random.split(key)
        return likelihood_presets[name](
            detectors,
            waveform,
            prior=prior,
            **self.run.likelihood_parameters,
            **self.run.data_parameters,
        )

    def initialize_prior(self) -> prior.Prior:
        priors = []
        for name, parameters in self.run.priors.items():
            if parameters["name"] not in prior_presets:
                raise ValueError(f"Prior {name} not recognized.")
            if parameters["name"] == "EarthFrame":
                priors.append(
                    prior.EarthFrame(
                        gps=self.run.data_parameters["trigger_time"],
                        ifos=self.run.detectors,
                    )
                )
            else:
                priors.append(
                    prior_presets[parameters["name"]](naming=[name], **parameters)
                )
        return prior.Composite(priors)

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

    ### Utility functions ###

    def get_detector_waveform(self, params: dict[str, float]) -> tuple[
        Float[Array, " n_sample"],
        dict[str, Float[Array, " n_sample"]],
        dict[str, Float[Array, " n_sample"]],
    ]:
        """
        Get the waveform in each detector.
        """
        if not self.run.injection:
            raise ValueError("No injection provided.")
        freqs = jnp.linspace(
            self.run.data_parameters["f_min"],
            self.run.data_parameters["f_sampling"] / 2,
            int(
                self.run.data_parameters["f_sampling"]
                * self.run.data_parameters["duration"]
            ),
        )
        freqs = freqs[
            (freqs >= self.run.data_parameters["f_min"])
            & (freqs <= self.run.data_parameters["f_max"])
        ]
        gmst = (
            Time(self.run.data_parameters["trigger_time"], format="gps")
            .sidereal_time("apparent", "greenwich")
            .rad
        )
        h_sky = self.jim.Likelihood.waveform(freqs, params)  # type: ignore
        align_time = jnp.exp(
            -1j * 2 * jnp.pi * freqs * (self.jim.Likelihood.epoch + params["t_c"])  # type: ignore
        )
        detector_parameters = {
            "ra": params["ra"],
            "dec": params["dec"],
            "psi": params["psi"],
            "t_c": params["t_c"],
            "gmst": gmst,
            "epoch": self.run.data_parameters["duration"]
            - self.run.data_parameters["post_trigger_duration"],
        }
        print(detector_parameters)
        detector_waveforms = {}
        for detector in self.jim.Likelihood.detectors:  # type: ignore
            detector_waveforms[detector.name] = (
                detector.fd_response(freqs, h_sky, detector_parameters) * align_time
            )
        return freqs, detector_waveforms, h_sky

    def plot_injection_waveform(self, path: str):
        """
        Plot the injection waveform.
        """
        freqs, waveforms, h_sky = self.get_detector_waveform(
            self.run.injection_parameters
        )
        plt.figure()
        for detector in self.jim.Likelihood.detectors:  # type: ignore
            plt.loglog(
                freqs,
                jnp.abs(waveforms[detector.name]),
                label=detector.name + " (injection)",
            )
            plt.loglog(
                freqs, jnp.sqrt(jnp.abs(detector.psd)), label=detector.name + " (PSD)"
            )
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.savefig(path)

    def plot_data(self, path: str):
        """
        Plot the data.
        """

        plt.figure()
        for detector in self.jim.Likelihood.detectors:  # type: ignore
            plt.loglog(
                detector.freqs,
                jnp.abs(detector.data),
                label=detector.name + " (data)",
            )
            plt.loglog(
                detector.freqs,
                jnp.sqrt(jnp.abs(detector.psd)),
                label=detector.name + " (PSD)",
            )
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.savefig(path)

    def sample(self):
        self.jim.sample(jax.random.PRNGKey(self.run.seed))

    def get_samples(self):
        return self.jim.get_samples()

    def plot_corner(self, path: str = "corner.jpeg", **kwargs):
        """
        plot corner plot of the samples.
        """
        plot_datapoint = kwargs.get("plot_datapoints", False)
        title_quantiles = kwargs.get("title_quantiles", [0.16, 0.5, 0.84])
        show_titles = kwargs.get("show_titles", True)
        title_fmt = kwargs.get("title_fmt", ".2E")
        use_math_text = kwargs.get("use_math_text", True)

        samples = self.jim.get_samples()
        param_names = list(samples.keys())
        samples = np.array(list(samples.values())).reshape(int(len(param_names)), -1).T
        corner.corner(
            samples,
            labels=param_names,
            plot_datapoints=plot_datapoint,
            title_quantiles=title_quantiles,
            show_titles=show_titles,
            title_fmt=title_fmt,
            use_math_text=use_math_text,
            **kwargs,
        )
        plt.savefig(path)
        plt.close()

    def plot_diagnostic(self, path: str = "diagnostic.jpeg", **kwargs):
        """
        plot diagnostic plot of the samples.
        """
        summary = self.jim.Sampler.get_sampler_state(training=True)
        chains, log_prob, local_accs, global_accs, loss_vals = summary.values()
        log_prob = np.array(log_prob)

        plt.figure(figsize=(10, 10))
        axs = [plt.subplot(2, 2, i + 1) for i in range(4)]
        plt.sca(axs[0])
        plt.title("log probability")
        plt.plot(log_prob.mean(0))
        plt.xlabel("iteration")
        plt.xlim(0, None)

        plt.sca(axs[1])
        plt.title("NF loss")
        plt.plot(loss_vals.reshape(-1))
        plt.xlabel("iteration")
        plt.xlim(0, None)

        plt.sca(axs[2])
        plt.title("Local Acceptance")
        plt.plot(local_accs.mean(0))
        plt.xlabel("iteration")
        plt.xlim(0, None)

        plt.sca(axs[3])
        plt.title("Global Acceptance")
        plt.plot(global_accs.mean(0))
        plt.xlabel("iteration")
        plt.xlim(0, None)
        plt.tight_layout()

        plt.savefig(path)
        plt.close()
