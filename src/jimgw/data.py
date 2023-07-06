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
