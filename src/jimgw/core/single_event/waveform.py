from abc import ABC

import jax.numpy as jnp
from jaxtyping import Array, Float
from ripplegw import FDWaveform
from ripplegw.waveforms.IMRPhenomXAS import Polarization

class Waveform(ABC):
    def __init__(self):
        return NotImplemented

    def __call__(
        self, axis: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        return NotImplemented


class RippleIMRPhenomD(Waveform):
    f_ref: float
    _waveform: FDWaveform.IMRPhenomD.value

    def __init__(self, f_ref: float = 20.0, **kwargs):
        self.f_ref = f_ref
        self._waveform = FDWaveform.IMRPhenomD.value()

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
        result = self._waveform(frequency, theta, {"f_ref": self.f_ref})
        output["p"] = result[Polarization.P]
        output["c"] = result[Polarization.C]
        return output

    def __repr__(self):
        return f"RippleIMRPhenomD(f_ref={self.f_ref})"


class RippleIMRPhenomPv2(Waveform):
    f_ref: float
    _waveform: FDWaveform.IMRPhenomPv2.value

    def __init__(self, f_ref: float = 20.0, **kwargs):
        self.f_ref = f_ref
        self._waveform = FDWaveform.IMRPhenomPv2.value()

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
        result = self._waveform(frequency, theta, {"f_ref": self.f_ref})
        output["p"] = result[Polarization.P]
        output["c"] = result[Polarization.C]
        return output

    def __repr__(self):
        return f"RippleIMRPhenomPv2(f_ref={self.f_ref})"

class RippleTaylorF2(Waveform):
    f_ref: float
    use_lambda_tildes: bool
    _waveform: FDWaveform.TaylorF2.value

    def __init__(self, f_ref: float = 20.0, use_lambda_tildes: bool = False):
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes
        self._waveform = FDWaveform.TaylorF2.value()

    def __call__(
        self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}

        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]

        theta = jnp.array(
            [
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
        )
        result = self._waveform(
            frequency,
            theta,
            {"f_ref": self.f_ref, "use_lambda_tildes": self.use_lambda_tildes},
        )
        output["p"] = result[Polarization.P]
        output["c"] = result[Polarization.C]
        return output

    def __repr__(self):
        return f"RippleTaylorF2(f_ref={self.f_ref})"

class RippleIMRPhenomD_NRTidalv2(Waveform):
    f_ref: float
    use_lambda_tildes: bool
    no_taper: bool
    _waveform: FDWaveform.IMRPhenomD_NRTidalv2.value

    def __init__(
        self,
        f_ref: float = 20.0,
        use_lambda_tildes: bool = False,
        no_taper: bool = False,
    ):
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
        self._waveform = FDWaveform.IMRPhenomD_NRTidalv2.value()

    def __call__(
        self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}

        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]

        theta = jnp.array(
            [
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
        )

        result = self._waveform(
            frequency,
            theta,
            {
                "f_ref": self.f_ref,
                "use_lambda_tildes": self.use_lambda_tildes,
                "no_taper": self.no_taper,
            },
        )
        output["p"] = result[Polarization.P]
        output["c"] = result[Polarization.C]
        return output

    def __repr__(self):
        return f"RippleIMRPhenomD_NRTidalv2(f_ref={self.f_ref})"

class JaxNRSurHyb3dq8(Waveform):
    _waveform: FDWaveform.NRSurHyb3dq8_FD.value

    def __init__(self, target_frequency: Float[Array, " n_sample"], segment_length: float, sampling_rate: int = 4096, alpha_window: float = 0.1):
        self._waveform = FDWaveform.NRSurHyb3dq8_FD.value(target_frequency, segment_length=segment_length, sampling_rate=sampling_rate, alpha_window=alpha_window)

    def __call__(
        self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}
        Mc = params["M_c"]
        q = params["q"]
        eta = q / (1 + q) ** 2
        M_tot = Mc / (eta ** 0.6)
        theta = jnp.array(
            [
                M_tot,
                params["d_L"],
                1. / q,
                params["s1_z"],
                params["s2_z"],
            ]
        )
        result = self._waveform(frequency, theta, {})
        phi_c = params['phase_c']
        iota = params['iota']
        output["p"] = result[Polarization.P] * jnp.exp(1j * phi_c) * (1 / 2 * (1 + jnp.cos(iota) ** 2))
        output["c"] = -1j * result[Polarization.C] * jnp.exp(1j * phi_c) * jnp.cos(iota)
        return output

    def __repr__(self):
        return f"JaxNRSurHyb3dq8(segment_length={self._waveform.surrogate.segment_length}, sampling_rate={self._waveform.surrogate.sampling_rate})"
        

class JaxNRSur7dq4(Waveform):
    _waveform: FDWaveform.NRSur7dq4_FD.value

    def __init__(self, target_frequency: Float[Array, " n_sample"], segment_length: float, sampling_rate: int = 4096, alpha_window: float = 0.1):
        self._waveform = FDWaveform.NRSur7dq4_FD.value(target_frequency, segment_length=segment_length, sampling_rate=sampling_rate, alpha_window=alpha_window)

    def __call__(
        self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}
        Mc = params["M_c"]
        q = params["q"]
        eta = q/(1 + q) ** 2
        M_tot = Mc / (eta ** 0.6)
        theta = jnp.array(
            [
                M_tot,
                params["d_L"],
                1./q,
                params["s1_x"],
                params["s1_y"],
                params["s1_z"],
                params["s2_x"],
                params["s2_y"],
                params["s2_z"],
            ]
        )
        result = self._waveform(frequency, theta, {})
        
        phi_c = params['phase_c']
        iota = params['iota']
        output["p"] = result[Polarization.P] * jnp.exp(1j * phi_c) * (1 / 2 * (1 + jnp.cos(iota) ** 2))
        output["c"] = -1j * result[Polarization.C] * jnp.exp(1j * phi_c) * jnp.cos(iota) 
        return output

    def __repr__(self):
        return f"JaxNRSur7dq4(segment_length={self._waveform.surrogate.segment_length}, sampling_rate={self._waveform.surrogate.sampling_rate})"



waveform_preset = {
    "RippleIMRPhenomD": RippleIMRPhenomD,
    "RippleIMRPhenomPv2": RippleIMRPhenomPv2,
    "RippleTaylorF2": RippleTaylorF2,
    "RippleIMRPhenomD_NRTidalv2": RippleIMRPhenomD_NRTidalv2,
    "JaxNRSurHyb3dq8": JaxNRSurHyb3dq8,
    "JaxNRSur7dq4": JaxNRSur7dq4,
}
