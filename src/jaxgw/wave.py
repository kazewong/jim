# Credit some part of the source code from bilby

import jax.numpy as jnp
from jaxgw.PE.constants import *

KNOWN_POLS = 'pcxybl'

class Polarization(object):
    """Object defining a given polarization mode, with utilities to produce
    corresponding tensor in an Earth centric frame.

    Arguments
    ---------
    name : str
        one of 'p' (plus), 'c' (cross), 'x' (vector x), 'y' (vector y), 'b'
        (breathing), or 'l' (longitudinal).
    """
    def __init__(self, name):
        self.name = name.lower()
        if self.name not in KNOWN_POLS:
            e = f"unknown mode '{self.name}'; must be one of: {KNOWN_POLS}"
            raise ValueError(e)

    @property
    def tensor_from_basis_constructor(self):
        """Constructor to obtain polarization tensor from waveframe basis
        defined by orthonormal vectors (x, y) in arbitrary Cartesian
        coordinates.
        """
        if self.name == 'p':
            def kernel(x, y):
                """Plus polarization from (x, y) waveframe basis elements.
                """
                return jnp.einsum('i,j->ij', x, x) - jnp.einsum('i,j->ij', y, y)
        elif self.name == 'c':
            def kernel(x, y):
                """Cross polarization from (x, y) waveframe basis elements.
                """
                return jnp.einsum('i,j->ij', x, y) + jnp.einsum('i,j->ij', y, x)
        elif self.name == 'x':
            def kernel(x, y):
                """Vector-x polarization from (x, y) waveframe basis elements.
                """
                z = jnp.cross(x, y)
                return jnp.einsum('i,j->ij', x, z) + jnp.einsum('i,j->ij', z, x)
        elif self.name == 'y':
            def kernel(x, y):
                """Vector-y polarization from (x, y) waveframe basis elements.
                """
                z = jnp.cross(x, y)
                return jnp.einsum('i,j->ij', y, z) + jnp.einsum('i,j->ij', z, y)
        elif self.name == 'b':
            def kernel(x, y):
                """Breathing polarization from (x, y) waveframe basis elements.
                """
                return jnp.einsum('i,j->ij', x, x) + jnp.einsum('i,j->ij', y, y)
        elif self.name == 'l':
            def kernel(x, y):
                """Longitudinal polarization from (x, y) waveframe basis elements.
                """
                z = jnp.cross(x, y)
                return jnp.einsum('i,j->ij', z, z)
        else:
            raise ValueError(f"unrecognized polarization {self.name}"
        return kernel

    @property
    def tensor_from_sky_constructor(self):
        """Constructor to obtain polarization tensor from sky location and
        orientation parameters.
        """
        kernel = self.tensor_from_sky_constructor
        def get_pol_tensor(ra, dec, psi, gmst):
            """Computes {name} polarization tensor in celestial
            coordinates from sky location and orientation parameters.

            Arguments
            ---------
            ra : float
                right ascension in radians.
            dec : float
                declination in radians.
            psi : float
                polarization angle in radians.
            gmst : float
                Greenwhich mean standard time (GMST) in radians.

            Returns
            -------
            tensor : array
                3x3 polarization tensor.
            """
            gmst = jnp.mod(gmst, 2*jnp.pi)
            phi = ra - gmst
            theta = jnp.pi / 2 - dec

            u = jnp.array([jnp.cos(phi) * jnp.cos(theta),
                           jnp.cos(theta) * jnp.sin(phi),
                           -jnp.sin(theta)])
            v = jnp.array([-jnp.sin(phi), jnp.cos(phi), 0])
            m = -u * jnp.sin(psi) - v * jnp.cos(psi)
            n = -u * jnp.cos(psi) + v * jnp.sin(psi)

            return kernel(m, n)
        get_pol_tensor.__doc__ = get_pol_tensor.__doc__.format(name=self.name)
        return get_pol_tensor
            




