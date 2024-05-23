from abc import ABC

import jax.numpy as jnp
from jaxtyping import Array, Float
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
from ripple.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
from ripple.waveforms.TaylorF2 import gen_TaylorF2_hphc
from ripple.waveforms.IMRPhenomD_NRTidalv2 import gen_IMRPhenomD_NRTidalv2_hphc

class Waveform(ABC):
    def __init__(self):
        return NotImplemented

    def __call__(
        self, axis: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        return NotImplemented


class RippleIMRPhenomD(Waveform):
    f_ref: float

    def __init__(self, f_ref: float = 20.0, **kwargs):
        self.f_ref = f_ref

    def __call__(
        self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                params["d_L"],
                0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomD_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"RippleIMRPhenomD(f_ref={self.f_ref})"


class RippleIMRPhenomPv2(Waveform):
    f_ref: float

    def __init__(self, f_ref: float = 20.0, **kwargs):
        self.f_ref = f_ref

    def __call__(
        self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_x"],
                params["s1_y"],
                params["s1_z"],
                params["s2_x"],
                params["s2_y"],
                params["s2_z"],
                params["d_L"],
                0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomPv2_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"RippleIMRPhenomPv2(f_ref={self.f_ref})"

class RippleTaylorF2(Waveform):

    f_ref: float
    use_lambda_tildes: bool

    def __init__(self, f_ref: float = 20.0, use_lambda_tildes: bool = False):
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes

    def __call__(self, frequency: Array, params: dict) -> dict:
        output = {}
        
        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]
        
        theta = [
            params["M_c"],
            params["eta"],
            params["s1_z"],
            params["s2_z"],
            first_lambda_param,
            second_lambda_param,
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        hp, hc = gen_TaylorF2_hphc(frequency, theta, self.f_ref, use_lambda_tildes=self.use_lambda_tildes)
        output["p"] = hp
        output["c"] = hc
        return output
    
    def __repr__(self):
        return f"RippleTaylorF2(f_ref={self.f_ref})"
    
class RippleIMRPhenomD_NRTidalv2(Waveform):

    f_ref: float
    use_lambda_tildes: bool

    def __init__(self, f_ref: float = 20.0, use_lambda_tildes: bool = False, no_taper: bool = False):
        """
        Initialize the waveform.

        Args:
            f_ref (float, optional): Reference frequency in Hz. Defaults to 20.0.
            use_lambda_tildes (bool, optional): Whether we sample over lambda_tilde and delta_lambda_tilde, as defined for instance in Equation (5) and Equation (6) of arXiv:1402.5156, rather than lambda_1 and lambda_2. Defaults to False.
            no_taper (bool, optional): Whether to remove the Planck taper in the amplitude of the waveform, which we use for relative binning runs. Defaults to False.
        """
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes
        self.no_taper = no_taper

    def __call__(self, frequency: Array, params: dict) -> dict:
        output = {}
        
        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]
        
        theta = [
            params["M_c"],
            params["eta"],
            params["s1_z"],
            params["s2_z"],
            first_lambda_param,
            second_lambda_param,
            params["d_L"],
            0,
            params["phase_c"],
            params["iota"],
        ]
        
        hp, hc = gen_IMRPhenomD_NRTidalv2_hphc(frequency, theta, self.f_ref, use_lambda_tildes=self.use_lambda_tildes, no_taper=self.no_taper)
        output["p"] = hp
        output["c"] = hc
        return output
    
    def __repr__(self):
        return f"RippleIMRPhenomD_NRTidalv2(f_ref={self.f_ref})"
    
waveform_preset = {
    "RippleIMRPhenomD": RippleIMRPhenomD,
    "RippleIMRPhenomPv2": RippleIMRPhenomPv2,
    "RippleTaylorF2": RippleTaylorF2,
    "RippleIMRPhenomD_NRTidalv2": RippleIMRPhenomD_NRTidalv2,
}
