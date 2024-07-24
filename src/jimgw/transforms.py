from dataclasses import field
from typing import Callable, Union

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Float, Array, jaxtyped


class Transform:
    """
    Base class for transform.

    The idea of transform should be used on distribtuion,
    """

    transform_func: Callable[[Float[Array, " N"]], Float[Array, " M"]]
    jacobian_func: Callable[[Float[Array, " N"]], Float]
    name_mapping: tuple[list[str], list[str]]

    def __init__(
        self,
        transform_func: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        name_mapping: tuple[list[str], list[str]],
    ):
        self.transform_func = transform_func
        self.jacobian_func = jax.jacfwd(transform_func)
        self.name_mapping = name_mapping

    def __call__(self, x: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        """
        Transform the input x to transformed coordinate y and return the log Jacobian determinant.
        This only works if the transform is a N -> N transform.
        
        Parameters
        ----------
        x : dict[str, Float]
                The input dictionary.

        Returns
        -------
        y : dict[str, Float]
                The transformed dictionary.
        log_det : Float
                The log Jacobian determinant.
        """
        return self.transform_func(x), jnp.log(jnp.linalg.det(self.jacobian_func(x)))
    
    def push_forward(self, x: dict[str, Float]) -> dict[str, Float]:
        """
        Push forward the input x to transformed coordinate y.

        Parameters
        ----------
        x : dict[str, Float]
                The input dictionary.

        Returns
        -------
        y : dict[str, Float]
                The transformed dictionary.
        """
        return self.transform_func(x)