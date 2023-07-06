from abc import ABC, abstractmethod
from jaxtyping import Array
from jimgw.likelihood import LikelihoodBase
from flowMC.sampler.Sampler import Sampler

class Jim(object):
    """ Master class for interfacing with flowMC
    
    """

    def __init__(self, Sampler: Sampler, Likelihood: LikelihoodBase, **kwargs):
        pass

    def maximize_likleihood(self):
        pass

    def sample(self):
        pass

    def plot(self):
        pass