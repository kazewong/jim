import jax
import jax.numpy as jnp
from flowMC.nfmodel.base import Distribution
from jaxtyping import Array, Float
from typing import Callable, Union
from dataclasses import field

class Prior(Distribution):
    """
    A thin wrapper build on top of flowMC distributions to do book keeping.

    Should not be used directly since it does not implement any of the real method.

    The rationale behind this is to have a class that can be used to keep track of
    the names of the parameters and the transforms that are applied to them.
    """

    naming: list[str]
    transforms: dict[tuple[str,Callable]] = field(default_factory=dict)

    @property
    def n_dim(self):
        return len(self.naming)
    
    def __init__(self, naming: list[str], transforms: dict[tuple[str,Callable]] = {}):
        """
        Parameters
        ----------
        naming : list[str]
            A list of names for the parameters of the prior.
        transforms : dict[tuple[str,Callable]]
            A dictionary of transforms to apply to the parameters. The keys are
            the names of the parameters and the values are a tuple of the name
            of the transform and the transform itself.
        """
        self.naming = naming
        self.transforms = {}

        def make_lambda(name):
                return lambda x: x[name]

        for name in naming:
            if name in transforms:
                self.transforms[name] = transforms[name]
            else:
                self.transforms[name] = (name, make_lambda(name)) # Without the function, the lambda will refer to the variable name instead of its value, which will make lambda reference the last value of the variable name

    def transform(self, x: Array) -> Array:
        """
        Apply the transforms to the parameters.

        Parameters
        ----------
        x : dict
            A dictionary of parameters. Names should match the ones in the prior.

        Returns
        -------
        x : dict
            A dictionary of parameters with the transforms applied.
        """
        output = self.add_name(x, transform_name = False, transform_value = False)
        for i, (key, value) in enumerate(self.transforms.items()):
            x = x.at[i].set(value[1](output))
        return x

    def add_name(self, x: Array, transform_name: bool = False, transform_value: bool = False) -> dict:
        """
        Turn an array into a dictionary
        """
        if transform_name:
            naming = [value[0] for value in self.transforms.values()]
        else:
            naming = self.naming
        if transform_value:
            x = self.transform(x)
            value = x
        else:
            value = x
        return dict(zip(naming,value))

class Uniform(Prior):

    xmin: Array
    xmax: Array

    def __init__(self, xmin: Union[float,Array], xmax: Union[float,Array], **kwargs):
        super().__init__(kwargs.get("naming"), kwargs.get("transforms"))
        self.xmax = jnp.array(xmax)
        self.xmin = jnp.array(xmin)
    
    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
        """
        Sample from a uniform distribution.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : Array
            An array of shape (n_samples, n_dim) containing the samples.
        
        """
        samples = jax.random.uniform(rng_key, (n_samples,self.n_dim), minval=self.xmin, maxval=self.xmax)
        return samples # TODO: remember to cast this to a named array

    def log_prob(self, x: Array) -> Float:
        output = jnp.sum(jnp.where((x>=self.xmax) | (x<=self.xmin), jnp.zeros_like(x)-jnp.inf, jnp.zeros_like(x)))
        return output + jnp.sum(jnp.log(1./(self.xmax-self.xmin))) 
