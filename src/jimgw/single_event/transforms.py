from abc import ABC
from typing import Callable

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Float, Array, jaxtyped
from astropy.time import Time

from jimgw.single_event.detector import GroundBased2G
from jimgw.transforms import BijectiveTransform
from jimgw.single_event.utils import (
    Mc_q_to_m1_m2,
    m1_m2_to_Mc_q,
    q_to_eta,
    eta_to_q,
    ra_dec_to_zenith_azimuth,
    zenith_azimuth_to_ra_dec,
    euler_rotation,
)

@jaxtyped(typechecker=typechecker)
class ComponentMassesToChirpMassMassRatioTransform(BijectiveTransform):
    """
    Transform chirp mass and mass ratio to component masses

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.
    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        super().__init__(name_mapping)
        assert "m_1" in name_mapping[0] and "m_2" in name_mapping[0] and "M_c" in name_mapping[1] and "q" in name_mapping[1]

        def named_transform(x):
            Mc, q = m1_m2_to_Mc_q(x["m_1"], x["m_2"])
            return {"M_c": Mc, "q": q}

        self.transform_func = named_transform

        def named_inverse_transform(x):
            m1, m2 = Mc_q_to_m1_m2(x["M_c"], x["q"])
            return {"m_1": m1, "m_2": m2}

        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class MassRatioToSymmetricMassRatioTransform(BijectiveTransform):
    """
    Transform mass ratio to symmetric mass ratio

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        super().__init__(name_mapping)

        self.transform_func = lambda x: {
            name_mapping[1][0]: q_to_eta(x[name_mapping[0][0]])
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][0]: eta_to_q(x[name_mapping[1][0]])
        }


@jaxtyped(typechecker=typechecker)
class SkyFrameToDetectorFrameSkyPositionTransform(BijectiveTransform):
    """
    Transform sky frame to detector frame sky position

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    gmst: Float
    rotation: Float[Array, " 3 3"]
    rotation_inv: Float[Array, " 3 3"]

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        gps_time: Float,
        ifos: GroundBased2G,
    ):
        super().__init__(name_mapping)

        self.gmst = Time(gps_time, format="gps").sidereal_time("apparent", "greenwich").rad
        delta_x = ifos[0].vertex - ifos[1].vertex
        self.rotation = euler_rotation(delta_x)
        self.rotation_inv = jnp.linalg.inv(self.rotation)
        
        assert "ra" in name_mapping[0] and "dec" in name_mapping[0] and "zenith" in name_mapping[1] and "azimuth" in name_mapping[1]

        def named_transform(x):
            zenith, azimuth = ra_dec_to_zenith_azimuth(
                x["ra"], x["dec"], self.gmst, self.rotation
            )
            return {"zenith": zenith, "azimuth": azimuth}

        self.transform_func = named_transform

        def named_inverse_transform(x):
            ra, dec = zenith_azimuth_to_ra_dec(
                x["zenith"], x["azimuth"], self.gmst, self.rotation_inv
            )
            return {"ra": ra, "dec": dec}

        self.inverse_transform_func = named_inverse_transform
