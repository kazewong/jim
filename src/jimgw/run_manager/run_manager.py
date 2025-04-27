from dataclasses import asdict
import jax
import matplotlib.pyplot as plt
import corner
import numpy as np
import yaml
from jaxlib.xla_extension import ArrayImpl

from jimgw.core.jim import Jim
from jimgw.run_manager.run import Run
import logging

def jaxarray_representer(dumper: yaml.Dumper, data: ArrayImpl):
    return dumper.represent_list(data.tolist())


yaml.add_representer(ArrayImpl, jaxarray_representer)  # type: ignore


class RunManager:
    run: Run
    jim: Jim

    def __init__(self, run: Run | str):
        if isinstance(run, Run):
            self.run = run
        elif isinstance(run, str):
            self.run = Run.deserialize(run)
        else:
            logging.ERROR("Run object or path not given.")
        
        self.jim = Jim(run.likelihood, run.prior, run.sample_transforms, run.likelihood_transforms, **flowMC_params)

    ### Utility functions ###

    def sample(self):
        self.jim.sample(jax.random.PRNGKey(self.run.seed))

    def get_samples(self):
        return self.jim.get_samples()

    def plot_chains(self, path: str = "corner.jpeg", **kwargs):
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

    def plot_trace(self):
        raise NotImplementedError
    
    def plot_loss(self):
        raise NotImplementedError

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