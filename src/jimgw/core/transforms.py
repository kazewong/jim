from abc import ABC
from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.special import logit
from beartype import beartype as typechecker
from jaxtyping import Float, Array, jaxtyped


class Transform(ABC):
    """
    Base class for transform.
    The purpose of this class is purely for keeping name
    """

    name_mapping: tuple[list[str], list[str]]

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        self.name_mapping = name_mapping

    def propagate_name(self, x: list[str]) -> list[str]:
        result = [name for name in x if name not in self.name_mapping[0]]
        result += [name for name in self.name_mapping[1] if name not in result]
        return result


class NtoMTransform(Transform):

    transform_func: Callable[[dict[str, Float]], dict[str, Float]]

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
        x_copy = x.copy()
        output_params = self.transform_func(x_copy)
        jax.tree.map(
            lambda key: x_copy.pop(key),
            self.name_mapping[0],
        )
        jax.tree.map(
            lambda key: x_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return x_copy


class NtoNTransform(NtoMTransform):

    @property
    def n_dim(self) -> int:
        return len(self.name_mapping[0])

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
        x_copy = x.copy()
        transform_params = dict((key, x_copy[key]) for key in self.name_mapping[0])
        output_params = self.transform_func(transform_params)
        jacobian = jax.jacfwd(self.transform_func)(transform_params)
        jacobian = jnp.array(jax.tree.leaves(jacobian))
        jacobian = jnp.log(
            jnp.absolute(jnp.linalg.det(jacobian.reshape(self.n_dim, self.n_dim)))
        )
        jax.tree.map(
            lambda key: x_copy.pop(key),
            self.name_mapping[0],
        )
        jax.tree.map(
            lambda key: x_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return x_copy, jacobian


class BijectiveTransform(NtoNTransform):

    inverse_transform_func: Callable[[dict[str, Float]], dict[str, Float]]

    def inverse(self, y: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        """
        Inverse transform the input y to original coordinate x.

        Parameters
        ----------
        y : dict[str, Float]
                The transformed dictionary.

        Returns
        -------
        x : dict[str, Float]
                The original dictionary.
        log_det : Float
                The log Jacobian determinant.
        """
        y_copy = y.copy()
        transform_params = dict((key, y_copy[key]) for key in self.name_mapping[1])
        output_params = self.inverse_transform_func(transform_params)
        jacobian = jax.jacfwd(self.inverse_transform_func)(transform_params)
        jacobian = jnp.array(jax.tree.leaves(jacobian))
        jacobian = jnp.log(
            jnp.absolute(jnp.linalg.det(jacobian.reshape(self.n_dim, self.n_dim)))
        )
        jax.tree.map(
            lambda key: y_copy.pop(key),
            self.name_mapping[1],
        )
        jax.tree.map(
            lambda key: y_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return y_copy, jacobian

    def backward(self, y: dict[str, Float]) -> dict[str, Float]:
        """
        Pull back the input y to original coordinate x and return the log Jacobian determinant.

        Parameters
        ----------
        y : dict[str, Float]
                The transformed dictionary.

        Returns
        -------
        x : dict[str, Float]
                The original dictionary.
        """
        y_copy = y.copy()
        output_params = self.inverse_transform_func(y_copy)
        jax.tree.map(
            lambda key: y_copy.pop(key),
            self.name_mapping[1],
        )
        jax.tree.map(
            lambda key: y_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return y_copy


class ConditionalBijectiveTransform(BijectiveTransform):

    conditional_names: list[str]

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        conditional_names: list[str],
    ):
        super().__init__(name_mapping)
        self.conditional_names = conditional_names

    def transform(self, x: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        x_copy = x.copy()
        transform_params = dict((key, x_copy[key]) for key in self.name_mapping[0])
        transform_params.update(
            dict((key, x_copy[key]) for key in self.conditional_names)
        )
        output_params = self.transform_func(transform_params)
        jacobian = jax.jacfwd(self.transform_func)(transform_params)
        jacobian_copy = {
            key1: {key2: jacobian[key1][key2] for key2 in self.name_mapping[0]}
            for key1 in self.name_mapping[1]
        }
        jacobian = jnp.array(jax.tree.leaves(jacobian_copy))
        jacobian = jnp.log(
            jnp.absolute(jnp.linalg.det(jacobian.reshape(self.n_dim, self.n_dim)))
        )
        jax.tree.map(
            lambda key: x_copy.pop(key),
            self.name_mapping[0],
        )
        jax.tree.map(
            lambda key: x_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return x_copy, jacobian

    def inverse(self, y: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        y_copy = y.copy()
        transform_params = dict((key, y_copy[key]) for key in self.name_mapping[1])
        transform_params.update(
            dict((key, y_copy[key]) for key in self.conditional_names)
        )
        output_params = self.inverse_transform_func(transform_params)
        jacobian = jax.jacfwd(self.inverse_transform_func)(transform_params)
        jacobian_copy = {
            key1: {key2: jacobian[key1][key2] for key2 in self.name_mapping[1]}
            for key1 in self.name_mapping[0]
        }
        jacobian = jnp.array(jax.tree.leaves(jacobian_copy))
        jacobian = jnp.log(
            jnp.absolute(jnp.linalg.det(jacobian.reshape(self.n_dim, self.n_dim)))
        )
        jax.tree.map(
            lambda key: y_copy.pop(key),
            self.name_mapping[1],
        )
        jax.tree.map(
            lambda key: y_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return y_copy, jacobian


@jaxtyped(typechecker=typechecker)
class ScaleTransform(BijectiveTransform):
    scale: Float

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        scale: Float,
    ):
        super().__init__(name_mapping)
        self.scale = scale
        self.transform_func = lambda x: {
            name_mapping[1][i]: x[name_mapping[0][i]] * self.scale
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: x[name_mapping[1][i]] / self.scale
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class OffsetTransform(BijectiveTransform):
    offset: Float

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        offset: Float,
    ):
        super().__init__(name_mapping)
        self.offset = offset
        self.transform_func = lambda x: {
            name_mapping[1][i]: x[name_mapping[0][i]] + self.offset
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: x[name_mapping[1][i]] - self.offset
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class LogitTransform(BijectiveTransform):
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
        self.transform_func = lambda x: {
            name_mapping[1][i]: 1 / (1 + jnp.exp(-x[name_mapping[0][i]]))
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: logit(x[name_mapping[1][i]])
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class SineTransform(BijectiveTransform):
    """
    Sine transformation

    The original parameter is expected to be in [-pi/2, pi/2]

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
        self.transform_func = lambda x: {
            name_mapping[1][i]: jnp.sin(x[name_mapping[0][i]])
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: jnp.arcsin(x[name_mapping[1][i]])
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class CosineTransform(BijectiveTransform):
    """
    Cosine transformation

    The original parameter is expected to be in [0, pi]

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
        self.transform_func = lambda x: {
            name_mapping[1][i]: jnp.cos(x[name_mapping[0][i]])
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: jnp.arccos(x[name_mapping[1][i]])
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class BoundToBound(BijectiveTransform):
    """
    Bound to bound transformation
    """

    original_lower_bound: Float[Array, " n_dim"]
    original_upper_bound: Float[Array, " n_dim"]
    target_lower_bound: Float[Array, " n_dim"]
    target_upper_bound: Float[Array, " n_dim"]

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        original_lower_bound: Float[Array, " n_dim"],
        original_upper_bound: Float[Array, " n_dim"],
        target_lower_bound: Float[Array, " n_dim"],
        target_upper_bound: Float[Array, " n_dim"],
    ):
        super().__init__(name_mapping)
        self.original_lower_bound = original_lower_bound
        self.original_upper_bound = original_upper_bound
        self.target_lower_bound = target_lower_bound
        self.target_upper_bound = target_upper_bound

        self.transform_func = lambda x: {
            name_mapping[1][i]: (x[name_mapping[0][i]] - self.original_lower_bound[i])
            * (self.target_upper_bound[i] - self.target_lower_bound[i])
            / (self.original_upper_bound[i] - self.original_lower_bound[i])
            + self.target_lower_bound[i]
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: (x[name_mapping[1][i]] - self.target_lower_bound[i])
            * (self.original_upper_bound[i] - self.original_lower_bound[i])
            / (self.target_upper_bound[i] - self.target_lower_bound[i])
            + self.original_lower_bound[i]
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class BoundToUnbound(BijectiveTransform):
    """
    Bound to unbound transformation
    """

    original_lower_bound: Float
    original_upper_bound: Float

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        original_lower_bound: Float,
        original_upper_bound: Float,
    ):

        super().__init__(name_mapping)
        self.original_lower_bound = jnp.atleast_1d(original_lower_bound)
        self.original_upper_bound = jnp.atleast_1d(original_upper_bound)

        self.transform_func = lambda x: {
            name_mapping[1][i]: logit(
                (x[name_mapping[0][i]] - self.original_lower_bound[i])
                / (self.original_upper_bound[i] - self.original_lower_bound[i])
            )
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: (
                self.original_upper_bound[i] - self.original_lower_bound[i]
            )
            / (1 + jnp.exp(-x[name_mapping[1][i]]))
            + self.original_lower_bound[i]
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class SingleSidedUnboundTransform(BijectiveTransform):
    """
    Unbound upper limit transformation

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    original_lower_bound: Float

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        original_lower_bound: Float,
    ):
        super().__init__(name_mapping)
        self.original_lower_bound = jnp.atleast_1d(original_lower_bound)

        self.transform_func = lambda x: {
            name_mapping[1][i]: jnp.log(
                x[name_mapping[0][i]] - self.original_lower_bound[i]
            )
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: jnp.exp(x[name_mapping[1][i]])
            + self.original_lower_bound[i]
            for i in range(len(name_mapping[1]))
        }


class PowerLawTransform(BijectiveTransform):
    """
    PowerLaw transformation
    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.
    """

    xmin: Float
    xmax: Float
    alpha: Float

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        xmin: Float,
        xmax: Float,
        alpha: Float,
    ):
        super().__init__(name_mapping)
        self.xmin = xmin
        self.xmax = xmax
        self.alpha = alpha
        if alpha == -1.0:
            self.transform_func = lambda x: {
                name_mapping[1][i]: self.xmin
                * jnp.exp(x[name_mapping[0][i]] * jnp.log(self.xmax / self.xmin))
                for i in range(len(name_mapping[0]))
            }
            self.inverse_transform_func = lambda x: {
                name_mapping[0][i]: (
                    jnp.log(x[name_mapping[1][i]] / self.xmin)
                    / jnp.log(self.xmax / self.xmin)
                )
                for i in range(len(name_mapping[1]))
            }
        else:
            alphap1 = 1.0 + self.alpha
            self.transform_func = lambda x: {
                name_mapping[1][i]: (
                    self.xmin**alphap1
                    + x[name_mapping[0][i]] * (self.xmax**alphap1 - self.xmin**alphap1)
                )
                ** (1.0 / alphap1)
                for i in range(len(name_mapping[0]))
            }
            self.inverse_transform_func = lambda x: {
                name_mapping[0][i]: (
                    (x[name_mapping[1][i]] ** alphap1 - self.xmin**alphap1)
                    / (self.xmax**alphap1 - self.xmin**alphap1)
                )
                for i in range(len(name_mapping[1]))
            }


@jaxtyped(typechecker=typechecker)
class CartesianToPolarTransform(BijectiveTransform):
    """
    Transformation from (x, y) to (theta, r).
    Parameters
    ----------
    parameter_name : str
            The name of the parameter to be transformed.
    """

    def __init__(
        self,
        parameter_name: str,
    ):
        super().__init__(
            name_mapping=(
                [f"{parameter_name}_x", f"{parameter_name}_y"],
                [f"{parameter_name}_theta", f"{parameter_name}_r"],
            )
        )
        self.transform_func = lambda x: {
            f"{parameter_name}_theta": jnp.arctan2(
                x[f"{parameter_name}_y"], x[f"{parameter_name}_x"]
            )
            + jnp.pi,
            f"{parameter_name}_r": jnp.sqrt(
                x[f"{parameter_name}_x"] ** 2 + x[f"{parameter_name}_y"] ** 2
            ),
        }
        self.inverse_transform_func = lambda x: {
            f"{parameter_name}_x": x[f"{parameter_name}_r"]
            * jnp.cos(x[f"{parameter_name}_theta"]),
            f"{parameter_name}_y": x[f"{parameter_name}_r"]
            * jnp.sin(x[f"{parameter_name}_theta"]),
        }


@jaxtyped(typechecker=typechecker)
class PeriodicTransform(BijectiveTransform):
    """
    Periodic transformation
    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        xmin: Float,
        xmax: Float,
    ):
        super().__init__(name_mapping)
        self.xmin = xmin
        self.xmax = xmax
        scaling = 2 * jnp.pi / (self.xmax - self.xmin)
        self.transform_func = lambda x: {
            f"{name_mapping[1][0]}": x[name_mapping[0][0]]
            * jnp.cos(scaling * (x[name_mapping[0][1]] - self.xmin)),
            f"{name_mapping[1][1]}": x[name_mapping[0][0]]
            * jnp.sin(scaling * (x[name_mapping[0][1]] - self.xmin)),
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][1]: self.xmin
            + (jnp.pi + jnp.arctan2(x[name_mapping[1][1]], x[name_mapping[1][0]]))
            / scaling,
            name_mapping[0][0]: jnp.sqrt(
                x[name_mapping[1][0]] ** 2 + x[name_mapping[1][1]] ** 2
            ),
        }


@jaxtyped(typechecker=typechecker)
class RayleighTransform(BijectiveTransform):
    """
    Transformation from Uniform(0, 1) to Rayleigh distribution with scale parameter sigma
    Parameters
    ----------
    parameter_name : str
            The name of the parameter to be transformed.
    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        sigma: Float,
    ):
        super().__init__(name_mapping)
        self.sigma = sigma
        self.transform_func = lambda x: {
            name_mapping[1][i]: sigma * jnp.sqrt(-2 * jnp.log(x[name_mapping[0][i]]))
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: jnp.exp(-((x[name_mapping[1][i]] / sigma) ** 2) / 2)
            for i in range(len(name_mapping[1]))
        }


def reverse_bijective_transform(
    original_transform: BijectiveTransform,
) -> BijectiveTransform:

    reversed_name_mapping = (
        original_transform.name_mapping[1],
        original_transform.name_mapping[0],
    )
    if isinstance(original_transform, ConditionalBijectiveTransform):
        reversed_transform = ConditionalBijectiveTransform(
            name_mapping=reversed_name_mapping,
            conditional_names=original_transform.conditional_names,
        )
    else:
        reversed_transform = BijectiveTransform(name_mapping=reversed_name_mapping)
    reversed_transform.transform_func = original_transform.inverse_transform_func
    reversed_transform.inverse_transform_func = original_transform.transform_func
    reversed_transform.__repr__ = lambda: f"Reversed{repr(original_transform)}"

    return reversed_transform
