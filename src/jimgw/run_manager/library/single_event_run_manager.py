
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import corner
import numpy as np
import yaml
from astropy.time import Time

from jimgw.core.jim import Jim
from jimgw.run_manager.run_manager import RunManager
from jimgw.run_manager.run import Run

class SingleEventRunManager(RunManager):
    run: Run
    jim: Jim

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

    def load_from_path(self, path: str) -> Run:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return SingleEventRun(**data)


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
