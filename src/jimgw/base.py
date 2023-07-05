import equinox as eqx
from abc import ABC, abstractmethod
from jaxtyping import Array

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

class Jim(object):
    """ Master class for interfacing with flowMC
    
    """