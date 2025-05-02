import matplotlib.pyplot as plt
import corner
import numpy as np
from jaxtyping import Float, Array
from jimgw.core.jim import Jim
from jimgw.run_manager.run import Run

from flowMC.resource.buffers import Buffer
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
        raise NotImplementedError
        # self.jim.sample(jax.random.PRNGKey(self.run.seed))

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
        plot corner plot of the samples.
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

    def plot_trace(self):
        raise NotImplementedError

    def plot_loss(self):
        assert isinstance(
            loss := self.jim.sampler.resources["loss_buffer"], Buffer
        ), "Loss buffer is not a jax array"
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
        raise NotImplementedError

    def serialize_run(self):
        raise NotImplementedError

    def plot_prior(self):
        raise NotImplementedError

    def plot_acceptances(self):
        raise NotImplementedError

    def generate_summar(self):
        raise NotImplementedError
