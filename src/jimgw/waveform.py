from jaxtyping import Array
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
from ripple.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
from ripple.waveforms.PPE_IMRPhenomPv2 import gen_PPE_IMRPhenomPv2_hphc
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
        ra = params["ra"]
        dec = params["dec"]
        theta = [
            params["M_c"],
            params["eta"],
            params["s1_z"],
            params["s2_z"],
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        hp, hc = gen_IMRPhenomD_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output


class RippleIMRPhenomPv2(Waveform):

    f_ref: float

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def __call__(self, frequency: Array, params: dict) -> Array:
        output = {}
        theta = [
            params["M_c"],
            params["eta"],
            params['s1_x'],
            params['s1_y'],
            params["s1_z"],
            params['s2_x'],
            params['s2_y'],
            params["s2_z"],
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        hp, hc = gen_IMRPhenomPv2_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output

class RipplePPEIMRPhenomPv2(Waveform):

    f_ref: float
    ppe_index: Array

    def __init__(self, ppe_index: Array = [0], f_ref: float = 20.0):
        self.f_ref = f_ref
        self.ppe_index = ppe_index

    def __call__(self, frequency: Array, params: dict) -> Array:
        output = {}
        theta = [
            params["M_c"],
            params["eta"],
            params['s1_x'],
            params['s1_y'],
            params["s1_z"],
            params['s2_x'],
            params['s2_y'],
            params["s2_z"],
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        ppes = jnp.zeros(15)
        for index in self.ppe_index:
            ppes = ppes.at[index].set(params["ppes" + str(index)])
        hp, hc = gen_PPE_IMRPhenomPv2_hphc(frequency, theta, ppes, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output
