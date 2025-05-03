import matplotlib.pyplot as plt
import corner
import numpy as np
import jax
from jaxtyping import Float, Array
from jimgw.core.jim import Jim
from jimgw.run.run import Run

from flowMC.resource.buffers import Buffer
from flowMC.resource.nf_model.base import NFModel
import logging


class RunManager:
    run: Run
    jim: Jim
    result_dir: str

    def __init__(self, run: Run | str, result_dir: str = "./", **flowMC_params):
        if isinstance(run, Run):
            self.run = run
        elif isinstance(run, str):
            self.run = Run.from_file(run)
        else:
            logging.ERROR("Run object or path not given.")

        assert isinstance(
            run, Run
        ), "Run object or path not given. Please provide a Run object or a path to a serialized Run object."
        self.jim = Jim(
            run.likelihood,
            run.prior,
            run.sample_transforms,
            run.likelihood_transforms,
            **flowMC_params,
        )

        self.result_dir = result_dir

    ### Utility functions ###

    def sample(self):
        logging.info("Starting sampling...")
        self.jim.sample(self.jim.sample_initial_condition())

    def get_samples(
        self, training: bool = False
    ) -> dict[str, Float[Array, "n_chains n_dims"]]:
        return self.jim.get_samples(training=training)

    def plot_chains(
        self,
        training: bool = False,
        plot_datapoints: bool = False,
        title_quantiles: list[float] = [0.16, 0.5, 0.84],
        show_titles: bool = True,
        title_fmt: str = ".2E",
        use_math_text: bool = True,
    ):
        """
        Plot corner plot of the samples.
        """
        samples = self.jim.get_samples(training=training)
        param_names = list(samples.keys())
        samples = np.array(list(samples.values())).reshape(int(len(param_names)), -1).T
        corner.corner(
            samples,
            labels=param_names,
            plot_datapoints=plot_datapoints,
            title_quantiles=title_quantiles,
            show_titles=show_titles,
            title_fmt=title_fmt,
            use_math_text=use_math_text,
        )
        path = f"{self.result_dir}/corner.jpeg"
        plt.savefig(path)
        plt.close()

    def plot_loss(self):
        """
        Plot the loss history during training.
        """
        assert isinstance(
            loss := self.jim.sampler.resources["loss_buffer"], Buffer
        ), "Loss buffer is not a Buffer"
        plt.figure(figsize=(10, 8))
        plt.title("NF loss")
        plt.plot(loss.data)
        plt.xlabel("iteration")
        plt.xlim(0, None)
        plt.ylabel("loss")
        path = f"{self.result_dir}/loss.jpeg"
        plt.savefig(path)
        plt.close()

    def plot_nf_sample(self):
        """
        Plot samples from the normalizing flow to visualize the learned distribution.
        """
        assert isinstance(
            nf_model := self.jim.sampler.resources["model"], NFModel
        ), "NF model is not a normalizing flow model"
        samples = nf_model.sample(
            jax.random.PRNGKey(0), 10000
        )
        param_names = list(self.jim.get_samples().keys())
        samples = np.array(samples).reshape(int(len(param_names)), -1).T
        corner.corner(
            samples,
            labels=param_names,
            plot_datapoints=True,
            show_titles=True,
            title_fmt=".2E",
            use_math_text=True,
        )
        path = f"{self.result_dir}/nf_samples.jpeg"
        plt.savefig(path)
        plt.close()

    def plot_prior(self):
        """
        Plot samples from the prior distribution.
        """
        samples = self.jim.prior.sample(jax.random.PRNGKey(0), 10000)
        param_names = list(samples.keys())
        samples = np.array(list(samples.values())).reshape(int(len(param_names)), -1).T
        corner.corner(
            samples,
            labels=param_names,
            plot_datapoints=True,
            show_titles=True,
            title_fmt=".2E",
            use_math_text=True,
        )
        path = f"{self.result_dir}/prior_samples.jpeg"
        plt.savefig(path)
        plt.close()

    def plot_acceptances(self):
        """
        Plot the local and global acceptance rates during sampling.
        """
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.title("Local acceptance rate")
        local_acc = np.array(self.jim.sampler.resources["local_accs_training"])
        plt.plot(local_acc)
        plt.xlabel("iteration")
        plt.xlim(0, None)
        plt.ylabel("acceptance rate")

        plt.subplot(2, 1, 2)
        plt.title("Global acceptance rate")
        global_acc = np.array(self.jim.sampler.resources["global_accs_training"])
        plt.plot(global_acc)
        plt.xlabel("iteration")
        plt.xlim(0, None)
        plt.ylabel("acceptance rate")

        path = f"{self.result_dir}/acceptance_rates.jpeg"
        plt.savefig(path)
        plt.close()

    def generate_summary(self):
        """
        Generate a summary of the sampling results including statistics
        and diagnostics about the chains.
        """
        samples = self.jim.get_samples()
        param_names = list(samples.keys())
        samples_array = (
            np.array(list(samples.values())).reshape(int(len(param_names)), -1).T
        )

        summary = {}
        for i, param in enumerate(param_names):
            param_samples = samples_array[:, i]
            quantiles = np.percentile(param_samples, [16, 50, 84])
            summary[param] = {
                "median": quantiles[1],
                "lower_err": quantiles[1] - quantiles[0],
                "upper_err": quantiles[2] - quantiles[1],
                "mean": np.mean(param_samples),
                "std": np.std(param_samples),
            }

        return summary

    def serialize_run(self):
        """
        Serialize the run object to a file.
        """
        self.run.serialize(self.result_dir + "/config.yaml")        