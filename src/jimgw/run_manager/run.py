from abc import ABC, abstractmethod
from typing import Self
from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import Prior
from jimgw.core.transforms import BijectiveTransform, NtoMTransform

class Run(ABC):
    """

    A `Run` is a template of priors, likelihood transforms, and sample transforms.

    It is aimed to be an abstraction which wrap the flexible but complicated APIs of core jim into an object that the users only interact with the underlying `jim` through the parameters defined in the Run. It is responsible for constructing the likelihood object, the prior, sample_transform, and likelihood_transform needed in jim.

    The most important property of a Run instance is it needs to be able to deterministically declared. All arguments to a run has to be explicitly provided, and the content of a Run should be exactly the same given the same arguments.
    """

    likelihood: LikelihoodBase
    prior: Prior
    sample_transforms: list[BijectiveTransform]
    likelihood_transforms: list[NtoMTransform]

    @abstractmethod
    def serialize(self, path: str = "./"):
        """Serialize a `Run` object into a human readble config file.
        
        """
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, path: str) -> Self:
        """ Deserialize a config file into a `Run` object
        
        """
        raise NotImplementedError