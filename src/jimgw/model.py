import equinox as eqx
from abc import abstractmethod
from jaxtyping import Array


class Model(eqx.Module):
    params: dict

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    def __call__(self, x: Array) -> float:
        raise NotImplementedError
