from abc import ABC, abstractmethod
from typing import Self
from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import Prior
from jimgw.core.transforms import BijectiveTransform, NtoMTransform
from typing import Sequence


class RunDefinition(ABC):
    """

    A `Run` is a template of priors, likelihood transforms, and sample transforms.

    It is aimed to be an abstraction which wrap the flexible but complicated APIs of core jim into an object that the users only interact with the underlying `jim` through the parameters defined in the Run. It is responsible for constructing the likelihood object, the prior, sample_transform, and likelihood_transform needed in jim.

    The most important property of a Run instance is it needs to be able to deterministically declared. All arguments to a run has to be explicitly provided, and the content of a Run should be exactly the same given the same arguments.
    """

    working_dir: str
    seed: int
    flowMC_params: dict[str, float | int]
    likelihood: LikelihoodBase
    prior: Prior
    sample_transforms: Sequence[BijectiveTransform]
    likelihood_transforms: Sequence[NtoMTransform]

    @abstractmethod
    def serialize(self, path: str = "./"):
        """Serialize a `Run` object into a human readble config file."""

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
