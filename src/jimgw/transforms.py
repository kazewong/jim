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
    inverse_func: Callable[[dict[str, Float]], dict[str, Float]]

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
    def inverse_transform(self, x: dict[str, Float]) -> dict[str, Float]:
        """
        Inverse transform the input x to transformed coordinate y.

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
    
    @abstractmethod
    def backward(self, x: dict[str, Float]) -> dict[str, Float]:
        """
        Pull back the input x to transformed coordinate y.

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

    def inverse_transform(self, x: dict[str, Float]) -> dict[str, Float]:
        output_params = x.pop(self.name_mapping[1][0])
        assert_rank(output_params, 0)
        input_params = self.inverse_func(output_params)
        jacobian = jax.jacfwd(self.inverse_func)(output_params)
        x[self.name_mapping[0][0]] = input_params
        return x, jnp.log(jacobian)

    def forward(self, x: dict[str, Float]) -> dict[str, Float]:
        input_params = x.pop(self.name_mapping[0][0])
        assert_rank(input_params, 0)
        output_params = self.transform_func(input_params)
        x[self.name_mapping[1][0]] = output_params
        return x
    
    def backward(self, x: dict[str, Float]) -> dict[str, Float]:
        output_params = x.pop(self.name_mapping[1][0])
        assert_rank(output_params, 0)
        input_params = self.inverse_func(output_params)
        x[self.name_mapping[0][0]] = input_params
        return x


class Scale(UnivariateTransform):
    scale: Float

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        scale: Float,
    ):
        super().__init__(name_mapping)
        self.scale = scale
        self.transform_func = lambda x: x * self.scale
        self.inverse_func = lambda x: x / self.scale

class Offset(UnivariateTransform):
    offset: Float

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        offset: Float,
    ):
        super().__init__(name_mapping)
        self.offset = offset
        self.transform_func = lambda x: x + self.offset
        self.inverse_func = lambda x: x - self.offset

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
        self.inverse_func = lambda x: jnp.log(x / (1 - x))

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
