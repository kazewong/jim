# Credit some part of the source code from bilby

import equinox as eqx
import jax.numpy as np
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
        outer_fmt = "i...,j...->ij..."
        if self.name == "p":
            return np.einsum(outer_fmt, x, x) - np.einsum(outer_fmt, y, y)
        elif self.name == "c":
            return np.einsum(outer_fmt, x, y) + np.einsum(outer_fmt, y, x)
        elif self.name == "x":
            z = np.cross(x, y)
            return np.einsum(outer_fmt, x, z) + np.einsum(outer_fmt, z, x)
        elif self.name == "y":
            z = np.cross(x, y)
            return np.einsum(outer_fmt, y, z) + np.einsum(outer_fmt, z, y)
        elif self.name == "b":
            return np.einsum(outer_fmt, x, x) + np.einsum(outer_fmt, y, y)
        elif self.name == "l":
            z = np.cross(x, y)
            return np.einsum(outer_fmt, z, z)
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
        gmst = np.mod(gmst, 2 * np.pi)
        phi = ra - gmst
        theta = np.pi / 2 - dec

        u = np.array(
            [
                np.cos(phi) * np.cos(theta),
                np.cos(theta) * np.sin(phi),
                -np.sin(theta),
            ]
        )
        v = np.array([-np.sin(phi), np.cos(phi), phi * 0.0])
        m = -u * np.sin(psi) - v * np.cos(psi)
        n = -u * np.cos(psi) + v * np.sin(psi)

        return self.tensor_from_basis(m, n)
