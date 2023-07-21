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

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(self, frequency: Array, params: dict, coeffs: Array = jnp.array([])) -> Array:
        output = {}
        ra = params['ra']
        dec = params['dec']
        theta = [params['Mc'], params['eta'], params['s1z'], params['s2z'], params['distance'], 0, params['phic'], params['incl'], params['psi'], ra, dec]
        if len(coeffs) == 0:
            kappa = Mc_eta_to_ms(jnp.array([params['Mc'], params['eta']]))
            kappa = jnp.concatenate((kappa,theta[2:]))
            coeffs = get_coeffs(kappa) 
        hp, hc = gen_IMRPhenomD_polar(frequency, theta, self.f_ref, coeffs)
        output['p'] = hp
        output['c'] = hc
        return output


