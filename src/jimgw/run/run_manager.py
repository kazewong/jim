import matplotlib.pyplot as plt
import corner
import numpy as np
import jax
from jaxtyping import Float, Array
from jimgw.core.jim import Jim
from jimgw.run.run_definition import RunDefinition

from flowMC.resource.buffers import Buffer
from flowMC.resource.nf_model.base import NFModel
import logging
import os


class RunManager:
    run: RunDefinition
    jim: Jim

    def __init__(self, run: RunDefinition | str):
        if isinstance(run, RunDefinition):
            self.run = run
        elif isinstance(run, str):
            self.run = RunDefinition.from_file(run)
        else:
            logging.ERROR("Run object or path not given.")

        assert isinstance(
            run, RunDefinition
        ), "Run object or path not given. Please provide a Run object or a path to a serialized Run object."

        # Initialize the jim objects needed for the run
        run.initialize_jim_objects()

        self.jim = Jim(
            run.likelihood,
            run.prior,
            run.sample_transforms,
            run.likelihood_transforms,
            **run.flowMC_params,  # type: ignore
        )

        if not os.path.exists(run.working_dir):
            os.makedirs(run.working_dir)
        self.working_dir = run.working_dir

    def sample(self):
        logging.info("Starting sampling...")
        self.jim.sample(self.jim.sample_initial_condition())

    ### Data-fetching functions ###

    def get_chain_samples(
        self, training: bool = False
    ) -> dict[str, Float[Array, "n_chains n_dims"]]:
        """
        Fetch the samples from the sampler.
        """
        return self.jim.get_samples(training=training)

    def get_log_prob(self, training: bool = False) -> Float[Array, " n_chains"]:
        """
        Fetch the log probabilities of the samples.
        """
        log_prob = (
            self.jim.sampler.resources["log_prob_training"]
            if training
            else self.jim.sampler.resources["log_prob_production"]
        )
        assert isinstance(log_prob, Buffer), "Log probability is not a Buffer"
        return log_prob.data

    def get_loss_data(self):
        """
        Fetch the loss history data from the sampler resources.
        """
        loss = self.jim.sampler.resources["loss_buffer"]
        assert isinstance(loss, Buffer), "Loss buffer is not a Buffer"
        return loss.data

    def get_nf_samples(self, n_samples: int = 10000):
        """
        Fetch samples from the normalizing flow model.
        """
        nf_model = self.jim.sampler.resources["model"]
        assert isinstance(nf_model, NFModel), "NF model is not a normalizing flow model"
        samples = nf_model.sample(jax.random.PRNGKey(0), n_samples)
        param_names = list(self.jim.get_samples().keys())
        samples = np.array(samples)
        return {name: samples[:, i] for i, name in enumerate(param_names)}

    def get_prior_samples(self, n_samples: int = 10000):
        """
        Fetch samples from the prior distribution.
        """
        samples = self.jim.prior.sample(jax.random.PRNGKey(0), n_samples)
        return samples

    def get_acceptance_rates(self, training: bool = False):
        """
        Fetch the local and global acceptance rates from the sampler resources.
        """
        local_acc = (
            self.jim.sampler.resources["local_accs_training"]
            if training
            else self.jim.sampler.resources["local_accs_production"]
        )
        global_acc = (
            self.jim.sampler.resources["global_accs_training"]
            if training
            else self.jim.sampler.resources["global_accs_production"]
        )
        assert isinstance(local_acc, Buffer), "Local acceptance rate is not a Buffer"
        assert isinstance(global_acc, Buffer), "Global acceptance rate is not a Buffer"
        return {
            "local": local_acc.data,
            "global": global_acc.data,
        }

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
        samples_dict = self.get_chain_samples(training=training)
        param_names = list(samples_dict.keys())
        samples = (
            np.array(list(samples_dict.values())).reshape(int(len(param_names)), -1).T
        )
        corner.corner(
            samples,
            labels=param_names,
            plot_datapoints=plot_datapoints,
            title_quantiles=title_quantiles,
            show_titles=show_titles,
            title_fmt=title_fmt,
            use_math_text=use_math_text,
        )
        path = f"{self.working_dir}/corner.jpeg"
        plt.savefig(path)
        plt.close()

    def plot_loss(self):
        """
        Plot the loss history during training.
        """
        loss_data = self.get_loss_data()
        plt.figure(figsize=(10, 8))
        plt.title("NF loss")
        plt.plot(loss_data)
        plt.xlabel("iteration")
        plt.xlim(0, None)
        plt.ylabel("loss")
        path = f"{self.working_dir}/loss.jpeg"
        plt.savefig(path)
        plt.close()

    def plot_nf_sample(self):
        """
        Plot samples from the normalizing flow to visualize the learned distribution.
        """
        nf_samples = self.get_nf_samples(10000)
        param_names = list(nf_samples.keys())
        samples = np.array([nf_samples[name] for name in param_names]).T
        corner.corner(
            samples,
            labels=param_names,
            plot_datapoints=True,
            show_titles=True,
            title_fmt=".2E",
            use_math_text=True,
        )
        path = f"{self.working_dir}/nf_samples.jpeg"
        plt.savefig(path)
        plt.close()

    def plot_prior(self):
        """
        Plot samples from the prior distribution.
        """
        prior_samples = self.get_prior_samples(10000)
        param_names = list(prior_samples.keys())
        samples = (
            np.array(list(prior_samples.values())).reshape(int(len(param_names)), -1).T
        )
        corner.corner(
            samples,
            labels=param_names,
            plot_datapoints=True,
            show_titles=True,
            title_fmt=".2E",
            use_math_text=True,
        )
        path = f"{self.working_dir}/prior_samples.jpeg"
        plt.savefig(path)
        plt.close()

    def plot_acceptances(self):
        """
        Plot the local and global acceptance rates during sampling.
        """
        acc_rates = self.get_acceptance_rates()
        local_acc = acc_rates["local"]
        global_acc = acc_rates["global"]

        plt.figure(figsize=(10, 8))
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle("Acceptance Rates", fontsize=16)

        axs[0].plot(np.mean(local_acc, axis=0))
        axs[0].set_title("Local acceptance rate", fontsize=12)
        axs[0].set_ylabel("Acceptance rate")

        axs[1].plot(np.mean(global_acc, axis=0))
        axs[1].set_title("Global acceptance rate", fontsize=12)
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Acceptance rate")

        plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust layout to fit the suptitle
        path = f"{self.working_dir}/acceptance_rates.jpeg"
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
            quantiles = np.percentile(param_samples, [5, 50, 95])
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
        self.run.serialize(self.working_dir + "/config.yaml")
