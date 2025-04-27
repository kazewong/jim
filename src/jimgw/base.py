from abc import ABC, abstractmethod

import equinox as eqx
from jaxtyping import Array, Float


class Data(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def fetch(self):
        raise NotImplementedError


class Model(eqx.Module):
    params: dict

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    def __call__(self, x: Array) -> float:
        raise NotImplementedError


class LikelihoodBase(ABC):
    """
    Base class for likelihoods.
    Note that this likelihood class should work
    for a some what general class of problems.
    In light of that, this class would be some what abstract,
    but the idea behind it is this handles two main components of a likelihood:
    the data and the model.
    It should be able to take the data and model and evaluate the likelihood for
    a given set of parameters.

    """

    _model: object
    _data: object

    @property
    def model(self):
        """
        The model for the likelihood.
        """
        return self._model

    @property
    def data(self):
        """
        The data for the likelihood.
        """
        return self._data

    @abstractmethod
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        """
        Evaluate the likelihood for a given set of parameters.
        """
        raise NotImplementedError
