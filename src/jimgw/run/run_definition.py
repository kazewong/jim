from abc import ABC, abstractmethod
from typing import Self
from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import Prior
from jimgw.core.transforms import BijectiveTransform, NtoMTransform
from typing import Sequence
import jax


class RunDefinition(ABC):
    """

    A `Run` is a template of priors, likelihood transforms, and sample transforms.

    It is aimed to be an abstraction which wrap the flexible but complicated APIs of core jim into an object that the users only interact with the underlying `jim` through the parameters defined in the Run. It is responsible for constructing the likelihood object, the prior, sample_transform, and likelihood_transform needed in jim.

    The most important property of a Run instance is it needs to be able to deterministically declared. All arguments to a run has to be explicitly provided, and the content of a Run should be exactly the same given the same arguments.
    """

    working_dir: str
    seed: int
    flowMC_params: dict
    likelihood: LikelihoodBase
    prior: Prior
    sample_transforms: Sequence[BijectiveTransform]
    likelihood_transforms: Sequence[NtoMTransform]

    def __init__(
        self,
        working_dir: str = "./",
        seed: int = 0,
        likelihood: LikelihoodBase | None = None,
        prior: Prior | None = None,
        sample_transforms: Sequence[BijectiveTransform] | None = None,
        likelihood_transforms: Sequence[NtoMTransform] | None = None,
        rng_key: jax.Array = jax.random.PRNGKey(0),
        n_chains: int = 50,
        n_local_steps: int = 10,
        n_global_steps: int = 10,
        n_training_loops: int = 20,
        n_production_loops: int = 20,
        n_epochs: int = 20,
        mala_step_size: float = 0.01,
        rq_spline_hidden_units: Sequence[int] = [128, 128],
        rq_spline_n_bins: int = 10,
        rq_spline_n_layers: int = 2,
        learning_rate: float = 1e-3,
        batch_size: int = 10000,
        n_max_examples: int = 10000,
        local_thinning: int = 1,
        global_thinning: int = 1,
        n_NFproposal_batch_size: int = 1000,
        history_window: int = 100,
        n_temperatures: int = 5,
        max_temperature: float = 10.0,
        n_tempered_steps: int = 5
    ):
        self.working_dir = working_dir
        self.seed = seed
        self.flowMC_params = {
            "rng_key": rng_key,
            "n_chains": n_chains,
            "n_local_steps": n_local_steps,
            "n_global_steps": n_global_steps,
            "n_training_loops": n_training_loops,
            "n_production_loops": n_production_loops,
            "n_epochs": n_epochs,
            "mala_step_size": mala_step_size,
            "rq_spline_hidden_units": rq_spline_hidden_units,
            "rq_spline_n_bins": rq_spline_n_bins,
            "rq_spline_n_layers": rq_spline_n_layers,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_max_examples": n_max_examples,
            "local_thinning": local_thinning,
            "global_thinning": global_thinning,
            "n_NFproposal_batch_size": n_NFproposal_batch_size,
            "history_window": history_window,
            "n_temperatures": n_temperatures,
            "max_temperature": max_temperature,
            "n_tempered_steps": n_tempered_steps
        }


    @abstractmethod
    def initialize_jim_objects(self):
        """Initialize the jim objects needed for the run.
        In __init__, the default should be not to initialize any of the likelihood, prior, sample_transforms since they could take a significant amount of time to initialize, and the user may not need them immediately.
        
        Instead, this method is called when initializing the RunManager.
        """

    def serialize(self, path: str = "./") -> dict:
        """Serialize a `Run` object into a human readble config file."""
        results = {}
        results["working_dir"] = self.working_dir
        results["seed"] = self.seed
        results.update(self.flowMC_params)
        results.pop("rng_key", None)  # rng_key is not serializable
        
        return results

    def load_flowMC_params(self, inputs:dict):
        """Load the flowMC parameters into the Run object."""
        self.flowMC_params = {
            "rng_key": jax.random.PRNGKey(inputs.get("seed", self.seed)),
            "n_chains": inputs.get("n_chains", 50),
            "n_local_steps": inputs.get("n_local_steps", 10),
            "n_global_steps": inputs.get("n_global_steps", 10),
            "n_training_loops": inputs.get("n_training_loops", 20),
            "n_production_loops": inputs.get("n_production_loops", 20),
            "n_epochs": inputs.get("n_epochs", 20),
            "mala_step_size": inputs.get("mala_step_size", 0.01),
            "rq_spline_hidden_units": inputs.get("rq_spline_hidden_units", [128, 128]),
            "rq_spline_n_bins": inputs.get("rq_spline_n_bins", 10),
            "rq_spline_n_layers": inputs.get("rq_spline_n_layers", 2),
            "learning_rate": inputs.get("learning_rate", 1e-3),
            "batch_size": inputs.get("batch_size", 10000),
            "n_max_examples": inputs.get("n_max_examples", 10000),
            "local_thinning": inputs.get("local_thinning", 1),
            "global_thinning": inputs.get("global_thinning", 1),
            "n_NFproposal_batch_size": inputs.get("n_NFproposal_batch_size", 1000),
            "history_window": inputs.get("history_window", 100),
            "n_temperatures": inputs.get("n_temperatures", 5),
            "max_temperature": inputs.get("max_temperature", 10.0),
            "n_tempered_steps": inputs.get("n_tempered_steps", 5)
        }

    @classmethod
    @abstractmethod
    def deserialize(cls, path: str) -> Self:
        """Deserialize a config file into a `Run` object"""

    @classmethod
    def from_file(
        cls,
        path: str,
    ):
        """Load a `Run` object from a config file."""
        return cls.deserialize(path)
