import equinox as eqx
from jaxtyping import Array
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from ripple.waveforms.IMRPhenomD_utils import get_coeffs
from ripple import Mc_eta_to_ms
import jax.numpy as jnp

class Waveform(eqx.Module):

    def __init__(self):
        return NotImplemented

    def __call__(self, axis: Array, params: Array) -> Array:
        return NotImplemented

class RippleIMRPhenomD(Waveform):

    f_ref: float
    coeffs: Array

    def __init__(self, f_ref: float = 20.0, coeffs: Array = jnp.array([])):
        self.f_ref = f_ref
        self.coeffs = coeffs
    def __call__(self, frequency: Array, params: dict) -> Array:
        output = {}
        ra = params['ra']
        dec = params['dec']
        theta = [params['Mc'], params['eta'], params['s1z'], params['s2z'], params['distance'], 0, params['phic'], params['incl'], params['psi'], ra, dec]
        if len(self.coeffs) == 0:
            kappa = jnp.concatenate(Mc_eta_to_ms(jnp.array([params['Mc'], params['eta']])),theta[2:])
            self.coeffs = get_coeffs(kappa) 
        hp, hc = gen_IMRPhenomD_polar(frequency, theta, self.f_ref, self.coeffs)
        output['p'] = hp
        output['c'] = hc
        return output


