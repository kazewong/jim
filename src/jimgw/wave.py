# Credit some part of the source code from bilby

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

KNOWN_POLS = "pcxybl"


class Polarization(eqx.Module):
    name: str
    """Object defining a given polarization mode, with utilities to produce
    corresponding tensor in an Earth centric frame.

    Arguments
    ---------
    name : str
        one of 'p' (plus), 'c' (cross), 'x' (vector x), 'y' (vector y), 'b'
        (breathing), or 'l' (longitudinal).
    """

    def __init__(self, name: str):
        self.name = name.lower()
        if self.name not in KNOWN_POLS:
            e = f"unknown mode '{self.name}'; must be one of: {KNOWN_POLS}"
            raise ValueError(e)

    def tensor_from_basis(
        self, x: Float[Array, " 3"], y: Float[Array, " 3"]
    ) -> Float[Array, " 3 3"]:
        """Constructor to obtain polarization tensor from waveframe basis
        defined by orthonormal vectors (x, y) in arbitrary Cartesian
        coordinates.
        """
        if self.name == "p":
            return jnp.einsum("i,j->ij", x, x) - jnp.einsum("i,j->ij", y, y)
        elif self.name == "c":
            return jnp.einsum("i,j->ij", x, y) + jnp.einsum("i,j->ij", y, x)
        elif self.name == "x":
            z = jnp.cross(x, y)
            return jnp.einsum("i,j->ij", x, z) + jnp.einsum("i,j->ij", z, x)
        elif self.name == "y":
            z = jnp.cross(x, y)
            return jnp.einsum("i,j->ij", y, z) + jnp.einsum("i,j->ij", z, y)
        elif self.name == "b":
            return jnp.einsum("i,j->ij", x, x) + jnp.einsum("i,j->ij", y, y)
        elif self.name == "l":
            z = jnp.cross(x, y)
            return jnp.einsum("i,j->ij", z, z)
        else:
            raise ValueError(f"unrecognized polarization {self.name}")

    def tensor_from_sky(
        self, ra: Float, dec: Float, psi: Float, gmst: Float
    ) -> Float[Array, " 3 3"]:
        """Computes {name} polarization tensor in celestial
        coordinates from sky location and orientation parameters.

        Arguments
        ---------
        ra : Float
            right ascension in radians.
        dec : Float
            declination in radians.
        psi : Float
            polarization angle in radians.
        gmst : Float
            Greenwhich mean standard time (GMST) in radians.

        Returns
        -------
        tensor : array
            3x3 polarization tensor.
        """
        gmst = jnp.mod(gmst, 2 * jnp.pi)
        phi = ra - gmst
        theta = jnp.pi / 2 - dec

        u = jnp.array(
            [
                jnp.cos(phi) * jnp.cos(theta),
                jnp.cos(theta) * jnp.sin(phi),
                -jnp.sin(theta),
            ]
        )
        v = jnp.array([-jnp.sin(phi), jnp.cos(phi), 0])
        m = -u * jnp.sin(psi) - v * jnp.cos(psi)
        n = -u * jnp.cos(psi) + v * jnp.sin(psi)

        return self.tensor_from_basis(m, n)
