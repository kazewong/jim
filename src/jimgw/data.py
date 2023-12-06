from abc import ABC, abstractmethod


class Data(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def fetch(self):
        raise NotImplementedError
