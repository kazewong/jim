from jaxtyping import Array, Float
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
from ripple.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
from abc import ABC
import jax.numpy as jnp


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


waveform_preset = {
    "RippleIMRPhenomD": RippleIMRPhenomD,
    "RippleIMRPhenomPv2": RippleIMRPhenomPv2,
}
