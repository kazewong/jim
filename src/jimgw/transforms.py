from abc import ABC, abstractmethod
from dataclasses import field
from typing import Callable, Union

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from chex import assert_rank
from jaxtyping import Array, Float, jaxtyped


class Transform(ABC):
    """
    Base class for transform.

    The idea of transform should be used on distribtuion,
    """

    name_mapping: tuple[list[str], list[str]]
    transform_func: Callable[[dict[str, Float]], dict[str, Float]]

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        self.name_mapping = name_mapping

    def __call__(self, x: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        return self.transform(x)

    @abstractmethod
    def transform(self, x: dict[str, Float]) -> tuple[dict[str, Float], Float]:
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
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: dict[str, Float]) -> dict[str, Float]:
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
        raise NotImplementedError

    def propagate_name(self, x: list[str]) -> list[str]:
        input_set = set(x)
        from_set = set(self.name_mapping[0])
        to_set = set(self.name_mapping[1])
        return list(input_set - from_set | to_set)


class UnivariateTransform(Transform):

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        super().__init__(name_mapping)

    def transform(self, x: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        input_params = x.pop(self.name_mapping[0][0])
        assert_rank(input_params, 0)
        output_params = self.transform_func(input_params)
        jacobian = jax.jacfwd(self.transform_func)(input_params)
        x[self.name_mapping[1][0]] = output_params
        return x, jnp.log(jacobian)

    def forward(self, x: dict[str, Float]) -> dict[str, Float]:
        input_params = x.pop(self.name_mapping[0][0])
        assert_rank(input_params, 0)
        output_params = self.transform_func(input_params)
        x[self.name_mapping[1][0]] = output_params
        return x


class ScaleToRange(UnivariateTransform):

    range: tuple[Float, Float]

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        range: tuple[Float, Float],
    ):
        super().__init__(name_mapping)
        self.range = range
        self.transform_func = (
            lambda x: (self.range[1] - self.range[0]) * x + self.range[0]
        )


class Logit(UnivariateTransform):
    """
    Logit transform following

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        super().__init__(name_mapping)
        self.transform_func = lambda x: 1 / (1 + jnp.exp(-x))


class Sine(UnivariateTransform):
    """
    Transform from unconstrained space to uniform space.

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        super().__init__(name_mapping)
        self.transform_func = lambda x: jnp.sin(x)
