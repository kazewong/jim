from jaxtyping import Array
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
import jax.numpy as jnp
from abc import ABC

class Waveform(ABC):

    def __init__(self):
        return NotImplemented

    def __call__(self, axis: Array, params: Array) -> Array:
        return NotImplemented

class RippleIMRPhenomD(Waveform):

    f_ref: float

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(self, frequency: Array, params: dict) -> dict:
        output = {}
        ra = params['ra']
        dec = params['dec']
        theta = [params['M_c'], params['eta'], params['s1_z'], params['s2_z'], params['d_L'], 0, params['phase_c'], params['iota'], params['psi'], ra, dec]
        hp, hc = gen_IMRPhenomD_polar(frequency, theta, self.f_ref)
        output['p'] = hp
        output['c'] = hc
        return output


