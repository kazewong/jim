import jax.numpy as jnp
from jax import lax
from beartype import beartype as typechecker
from jaxtyping import Float, Array, jaxtyped, Bool
from astropy.time import Time

from jimgw.single_event.detector import GroundBased2G
from jimgw.transforms import (
    ConditionalBijectiveTransform,
    BijectiveTransform,
    NtoNTransform,
    reverse_bijective_transform,
)
from jimgw.single_event.utils import (
    Mc_m1_to_m2,
    m1_m2_to_Mc_q,
    Mc_q_to_m1_m2,
    m1_m2_to_Mc_eta,
    Mc_eta_to_m1_m2,
    q_to_eta,
    eta_to_q,
    ra_dec_to_zenith_azimuth,
    zenith_azimuth_to_ra_dec,
    euler_rotation,
    spin_to_cartesian_spin,
)


@jaxtyped(typechecker=typechecker)
class UniformComponentMassSecondaryMassQuantileToSecondaryMassTransform(
    ConditionalBijectiveTransform
):
    """ """

    q_min: Float
    q_max: Float
    M_c_min: Float
    M_c_max: Float
    m_1_turning_point_lower: Float
    m_1_turning_point_upper: Float
    regime_2_mass_ratio: Bool

    def __init__(
        self,
        q_min: Float,
        q_max: Float,
        M_c_min: Float,
        M_c_max: Float,
        m_1_min: Float,
        m_1_max: Float,
    ):
        name_mapping = (
            [
                "m_2_quantile",
            ],
            [
                "m_2",
            ],
        )
        conditional_names = [
            "m_1",
        ]
        super().__init__(name_mapping, conditional_names)

        self.q_min = q_min
        self.q_max = q_max
        self.M_c_min = M_c_min
        self.M_c_max = M_c_max

        assert (M_c_min >= 0.0) & (M_c_max > 0.0), "Chirp mass range has to be positive"
        assert (
            M_c_max > M_c_min
        ), "Upper bound on chirp mass has to be higher than the lower bound"
        assert (q_min >= 0.0) & (q_max > 0.0), "Mass ratio range has to be positive"
        assert (
            q_max > q_min
        ), "Upper bound on mass ratio has to be higher than the lower bound"
        assert q_max <= 1.0, "The mass ratio is defined to be less than 1"

        m_1_lower_bound = Mc_q_to_m1_m2(self.M_c_min, self.q_max)[0]
        m_1_upper_bound = Mc_q_to_m1_m2(self.M_c_max, self.q_min)[0]

        assert (
            m_1_min >= m_1_lower_bound
        ), f"Please increase the lower bound on m_1 to {m_1_lower_bound}"
        assert (
            m_1_max <= m_1_upper_bound
        ), f"Please decrease the upper bound on m_1 to {m_1_upper_bound}"

        m_1_turning_point_1 = Mc_q_to_m1_m2(self.M_c_min, self.q_min)[0]
        m_1_turning_point_2 = Mc_q_to_m1_m2(self.M_c_max, self.q_max)[0]

        def m2_range_regime_1(m_1: Float):
            lower_bound = Mc_m1_to_m2(self.M_c_min, m_1)
            upper_bound = self.q_max * m_1
            return [lower_bound, upper_bound]

        def m2_range_regime_3(m_1: Float):
            lower_bound = self.q_min * m_1
            upper_bound = Mc_m1_to_m2(self.M_c_max, m_1)
            return [lower_bound, upper_bound]

        if m_1_turning_point_2 >= m_1_turning_point_1:
            self.m_1_turning_point_lower = m_1_turning_point_1
            self.m_1_turning_point_upper = m_1_turning_point_2
            self.regime_2_mass_ratio = True
        else:
            self.m_1_turning_point_lower = m_1_turning_point_2
            self.m_1_turning_point_upper = m_1_turning_point_1
            self.regime_2_mass_ratio = False

        def m2_range_regime_2(m_1: Float):
            return lax.cond(
                self.regime_2_mass_ratio,
                lambda x: [self.q_min * x, self.q_max * x],
                lambda x: [
                    Mc_m1_to_m2(self.M_c_min, x),
                    Mc_m1_to_m2(self.M_c_max, x),
                ],
                m_1,
            )

        def m1_to_m2_range(m_1: Float):
            m2_range = lax.cond(
                m_1 < self.m_1_turning_point_lower,
                m2_range_regime_1,
                lambda x: lax.cond(
                    x <= self.m_1_turning_point_upper,
                    m2_range_regime_2,
                    m2_range_regime_3,
                    x,
                ),
                m_1,
            )
            return m2_range

        def named_transform(x):
            m2_range = m1_to_m2_range(x["m_1"])
            m_2 = (m2_range[1] - m2_range[0]) * x["m_2_quantile"] + m2_range[0]
            return {
                "m_2": m_2,
            }

        self.transform_func = named_transform

        def named_inverse_transform(x):
            m2_range = m1_to_m2_range(x["m_1"])
            m_2_quantile = (x["m_2"] - m2_range[0]) / (m2_range[1] - m2_range[0])
            return {
                "m_2_quantile": m_2_quantile,
            }

        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class PrecessingSpinToCartesianSpinTransform(NtoNTransform):
    """
    Spin to Cartesian spin transformation
    """

    freq_ref: Float

    def __init__(
        self,
        freq_ref: Float,
    ):
        name_mapping = (
            ["theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1", "a_2"],
            ["iota", "s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z"],
        )
        super().__init__(name_mapping)

        self.freq_ref = freq_ref

        def named_transform(x):
            iota, s1x, s1y, s1z, s2x, s2y, s2z = spin_to_cartesian_spin(
                x["theta_jn"],
                x["phi_jl"],
                x["tilt_1"],
                x["tilt_2"],
                x["phi_12"],
                x["a_1"],
                x["a_2"],
                x["M_c"],
                x["q"],
                self.freq_ref,
                x["phase_c"],
            )
            return {
                "iota": iota,
                "s1_x": s1x,
                "s1_y": s1y,
                "s1_z": s1z,
                "s2_x": s2x,
                "s2_y": s2y,
                "s2_z": s2z,
            }

        self.transform_func = named_transform


@jaxtyped(typechecker=typechecker)
class SphereSpinToCartesianSpinTransform(BijectiveTransform):
    """
    Spin to Cartesian spin transformation
    """

    def __init__(
        self,
        label: str,
    ):
        name_mapping = (
            [label + "_mag", label + "_theta", label + "_phi"],
            [label + "_x", label + "_y", label + "_z"],
        )
        super().__init__(name_mapping)

        def named_transform(x):
            mag, theta, phi = x[label + "_mag"], x[label + "_theta"], x[label + "_phi"]
            x = mag * jnp.sin(theta) * jnp.cos(phi)
            y = mag * jnp.sin(theta) * jnp.sin(phi)
            z = mag * jnp.cos(theta)
            return {
                label + "_x": x,
                label + "_y": y,
                label + "_z": z,
            }

        def named_inverse_transform(x):
            x, y, z = x[label + "_x"], x[label + "_y"], x[label + "_z"]
            mag = jnp.sqrt(x**2 + y**2 + z**2)
            theta = jnp.arccos(z / mag)
            phi = jnp.arctan2(y, x)
            return {
                label + "_mag": mag,
                label + "_theta": theta,
                label + "_phi": phi,
            }

        self.transform_func = named_transform
        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class SkyFrameToDetectorFrameSkyPositionTransform(BijectiveTransform):
    """
    Transform sky frame to detector frame sky position
    """

    gmst: Float
    rotation: Float[Array, " 3 3"]
    rotation_inv: Float[Array, " 3 3"]

    def __init__(
        self,
        gps_time: Float,
        ifos: list[GroundBased2G],
    ):
        name_mapping = (["ra", "dec"], ["zenith", "azimuth"])
        super().__init__(name_mapping)

        self.gmst = (
            Time(gps_time, format="gps").sidereal_time("apparent", "greenwich").rad
        )
        delta_x = ifos[0].vertex - ifos[1].vertex
        self.rotation = euler_rotation(delta_x)
        self.rotation_inv = jnp.linalg.inv(self.rotation)

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


@jaxtyped(typechecker=typechecker)
class GeocentricArrivalTimeToDetectorArrivalTimeTransform(
    ConditionalBijectiveTransform
):
    """
    Transform the geocentric arrival time to detector arrival time

    In the geocentric convention, the arrival time of the signal at the
    center of Earth is gps_time + t_c

    In the detector convention, the arrival time of the signal at the
    detecotr is gps_time + time_delay_from_geo_to_det + t_det

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    gmst: Float
    ifo: GroundBased2G
    tc_min: Float
    tc_max: Float

    def __init__(
        self,
        gps_time: Float,
        ifo: GroundBased2G,
        tc_min: Float,
        tc_max: Float,
    ):
        name_mapping = (["t_c"], ["t_det_unbounded"])
        conditional_names = ["ra", "dec"]
        super().__init__(name_mapping, conditional_names)

        self.gmst = (
            Time(gps_time, format="gps").sidereal_time("apparent", "greenwich").rad
        )
        self.ifo = ifo
        self.tc_min = tc_min
        self.tc_max = tc_max

        assert "t_c" in name_mapping[0] and "t_det_unbounded" in name_mapping[1]
        assert "ra" in conditional_names and "dec" in conditional_names

        def time_delay(ra, dec, gmst):
            return self.ifo.delay_from_geocenter(ra, dec, gmst)

        def named_transform(x):

            time_shift = time_delay(x["ra"], x["dec"], self.gmst)

            t_det = x["t_c"] + time_shift
            t_det_min = self.tc_min + time_shift
            t_det_max = self.tc_max + time_shift

            y = (t_det - t_det_min) / (t_det_max - t_det_min)
            t_det_unbounded = jnp.log(y / (1.0 - y))
            return {
                "t_det_unbounded": t_det_unbounded,
            }

        self.transform_func = named_transform

        def named_inverse_transform(x):

            time_shift = self.ifo.delay_from_geocenter(x["ra"], x["dec"], self.gmst)

            t_det_min = self.tc_min + time_shift
            t_det_max = self.tc_max + time_shift
            t_det = (t_det_max - t_det_min) / (
                1.0 + jnp.exp(-x["t_det_unbounded"])
            ) + t_det_min

            t_c = t_det - time_shift

            return {
                "t_c": t_c,
            }

        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
    ConditionalBijectiveTransform
):
    """
    Transform the geocentric arrival phase to detector arrival phase

    In the geocentric convention, the arrival phase of the signal at the
    center of Earth is phase_c / 2 (in ripple, phase_c is the orbital phase)

    In the detector convention, the arrival phase of the signal at the
    detecotr is phase_det = phase_c / 2 + arg R_det

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    gmst: Float
    ifo: GroundBased2G

    def __init__(
        self,
        gps_time: Float,
        ifo: GroundBased2G,
    ):
        name_mapping = (["phase_c"], ["phase_det"])
        conditional_names = ["ra", "dec", "psi", "iota"]
        super().__init__(name_mapping, conditional_names)

        self.gmst = (
            Time(gps_time, format="gps").sidereal_time("apparent", "greenwich").rad
        )
        self.ifo = ifo

        assert "phase_c" in name_mapping[0] and "phase_det" in name_mapping[1]
        assert (
            "ra" in conditional_names
            and "dec" in conditional_names
            and "psi" in conditional_names
            and "iota" in conditional_names
        )

        def _calc_R_det_arg(ra, dec, psi, iota, gmst):
            p_iota_term = (1.0 + jnp.cos(iota) ** 2) / 2.0
            c_iota_term = jnp.cos(iota)

            antenna_pattern = self.ifo.antenna_pattern(ra, dec, psi, gmst)
            p_mode_term = p_iota_term * antenna_pattern["p"]
            c_mode_term = c_iota_term * antenna_pattern["c"]

            return jnp.angle(p_mode_term - 1j * c_mode_term)

        def named_transform(x):
            R_det_arg = _calc_R_det_arg(
                x["ra"], x["dec"], x["psi"], x["iota"], self.gmst
            )
            phase_det = R_det_arg + x["phase_c"] / 2.0
            return {
                "phase_det": phase_det % (2.0 * jnp.pi),
            }

        self.transform_func = named_transform

        def named_inverse_transform(x):
            R_det_arg = _calc_R_det_arg(
                x["ra"], x["dec"], x["psi"], x["iota"], self.gmst
            )
            phase_c = -R_det_arg + x["phase_det"] * 2.0
            return {
                "phase_c": phase_c % (2.0 * jnp.pi),
            }

        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class DistanceToSNRWeightedDistanceTransform(ConditionalBijectiveTransform):
    """
    Transform the luminosity distance to network SNR weighted distance

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
            The name mapping between the input and output dictionary.

    """

    gmst: Float
    ifos: list[GroundBased2G]
    dL_min: Float
    dL_max: Float

    def __init__(
        self,
        gps_time: Float,
        ifos: list[GroundBased2G],
        dL_min: Float,
        dL_max: Float,
    ):
        name_mapping = (["d_L"], ["d_hat_unbounded"])
        conditional_names = ["M_c", "ra", "dec", "psi", "iota"]
        super().__init__(name_mapping, conditional_names)

        self.gmst = (
            Time(gps_time, format="gps").sidereal_time("apparent", "greenwich").rad
        )
        self.ifos = ifos
        self.dL_min = dL_min
        self.dL_max = dL_max

        assert "d_L" in name_mapping[0] and "d_hat_unbounded" in name_mapping[1]
        assert (
            "ra" in conditional_names
            and "dec" in conditional_names
            and "psi" in conditional_names
            and "iota" in conditional_names
            and "M_c" in conditional_names
        )

        def _calc_R_dets(ra, dec, psi, iota):
            p_iota_term = (1.0 + jnp.cos(iota) ** 2) / 2.0
            c_iota_term = jnp.cos(iota)
            R_dets2 = 0.0
            for ifo in self.ifos:
                antenna_pattern = ifo.antenna_pattern(ra, dec, psi, self.gmst)
                p_mode_term = p_iota_term * antenna_pattern["p"]
                c_mode_term = c_iota_term * antenna_pattern["c"]
                R_dets2 += p_mode_term**2 + c_mode_term**2

            return jnp.sqrt(R_dets2)

        def named_transform(x):
            d_L, M_c = (
                x["d_L"],
                x["M_c"],
            )
            R_dets = _calc_R_dets(x["ra"], x["dec"], x["psi"], x["iota"])

            scale_factor = 1.0 / jnp.power(M_c, 5.0 / 6.0) / R_dets
            d_hat = scale_factor * d_L

            d_hat_min = scale_factor * self.dL_min
            d_hat_max = scale_factor * self.dL_max

            y = (d_hat - d_hat_min) / (d_hat_max - d_hat_min)
            d_hat_unbounded = jnp.log(y / (1.0 - y))

            return {
                "d_hat_unbounded": d_hat_unbounded,
            }

        self.transform_func = named_transform

        def named_inverse_transform(x):
            d_hat_unbounded, M_c = (
                x["d_hat_unbounded"],
                x["M_c"],
            )
            R_dets = _calc_R_dets(x["ra"], x["dec"], x["psi"], x["iota"])

            scale_factor = 1.0 / jnp.power(M_c, 5.0 / 6.0) / R_dets

            d_hat_min = scale_factor * self.dL_min
            d_hat_max = scale_factor * self.dL_max

            d_hat = (d_hat_max - d_hat_min) / (
                1.0 + jnp.exp(-d_hat_unbounded)
            ) + d_hat_min
            d_L = d_hat / scale_factor
            return {
                "d_L": d_L,
            }

        self.inverse_transform_func = named_inverse_transform


def named_m1_m2_to_Mc_q(x):
    Mc, q = m1_m2_to_Mc_q(x["m_1"], x["m_2"])
    return {"M_c": Mc, "q": q}


def named_Mc_q_to_m1_m2(x):
    m1, m2 = Mc_q_to_m1_m2(x["M_c"], x["q"])
    return {"m_1": m1, "m_2": m2}


ComponentMassesToChirpMassMassRatioTransform = BijectiveTransform(
    (["m_1", "m_2"], ["M_c", "q"])
)
ComponentMassesToChirpMassMassRatioTransform.transform_func = named_m1_m2_to_Mc_q
ComponentMassesToChirpMassMassRatioTransform.inverse_transform_func = (
    named_Mc_q_to_m1_m2
)


def named_m1_m2_to_Mc_eta(x):
    Mc, eta = m1_m2_to_Mc_eta(x["m_1"], x["m_2"])
    return {"M_c": Mc, "eta": eta}


def named_Mc_eta_to_m1_m2(x):
    m1, m2 = Mc_eta_to_m1_m2(x["M_c"], x["eta"])
    return {"m_1": m1, "m_2": m2}


ComponentMassesToChirpMassSymmetricMassRatioTransform = BijectiveTransform(
    (["m_1", "m_2"], ["M_c", "eta"])
)
ComponentMassesToChirpMassSymmetricMassRatioTransform.transform_func = (
    named_m1_m2_to_Mc_eta
)
ComponentMassesToChirpMassSymmetricMassRatioTransform.inverse_transform_func = (
    named_Mc_eta_to_m1_m2
)


def named_q_to_eta(x):
    return {"eta": q_to_eta(x["q"])}


def named_eta_to_q(x):
    return {"q": eta_to_q(x["eta"])}


MassRatioToSymmetricMassRatioTransform = BijectiveTransform((["q"], ["eta"]))
MassRatioToSymmetricMassRatioTransform.transform_func = named_q_to_eta
MassRatioToSymmetricMassRatioTransform.inverse_transform_func = named_eta_to_q


ChirpMassMassRatioToComponentMassesTransform = reverse_bijective_transform(
    ComponentMassesToChirpMassMassRatioTransform
)
ChirpMassSymmetricMassRatioToComponentMassesTransform = reverse_bijective_transform(
    ComponentMassesToChirpMassSymmetricMassRatioTransform
)
SymmetricMassRatioToMassRatioTransform = reverse_bijective_transform(
    MassRatioToSymmetricMassRatioTransform
)
