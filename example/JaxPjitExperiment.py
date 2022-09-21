import jax
import jax.numpy as jnp
from jax.experimental import maps
from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit
from jax.scipy.special import logsumexp

import numpy as np

def dual_moon_pe(x):
    """
    Term 2 and 3 separate the distribution and smear it along the first and second dimension
    """
    term1 = 0.5 * ((jnp.linalg.norm(x) - 2) / 0.1) ** 2
    term2 = -0.5 * ((x[:1] + jnp.array([-3.0, 3.0])) / 0.8) ** 2
    term3 = -0.5 * ((x[1:2] + jnp.array([-3.0, 3.0])) / 0.6) ** 2
    return -(term1 - logsumexp(term2) - logsumexp(term3))

mesh_shape = (4, 1)
devices = np.asarray(jax.devices()).reshape(*mesh_shape)
# 'x', 'y' axis names are used here for simplicity
mesh = maps.Mesh(devices, ('x', 'y'))

input_data = jnp.array(np.random.uniform(size=(100000,5)))
