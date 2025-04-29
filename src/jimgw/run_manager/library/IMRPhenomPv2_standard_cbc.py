from jimgw.run_manager.single_event_run import SingleEventRun
from typing import Literal

import jax.numpy as jnp

from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    UniformSpherePrior,
)
from jimgw.core.single_event.detector import H1, L1, V1
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2
from jimgw.core.transforms import BoundToUnbound, BijectiveTransform, NtoMTransform
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
)

detector_enum = {"H1": H1, "L1": L1, "V1": V1}


class IMRPhenomPv2StandardCBCRun(SingleEventRun):

    # Likelihood parameters

    gps: int  # GPS time of the event trigger
    segment_length: int  # Length of the segment
    post_trigger_length: int  # Length of segment after the trigger
    f_min: int  # Minimum frequency
    f_max: int  # Maximum frequency
    ifos: set[Literal["H1"], Literal["L1"], Literal["V1"]]
    f_ref: int  # Reference frequency

    # Prior parameters
    M_c_range: tuple[float, float]
    q_range: tuple[float, float]
    max_s1: float
    max_s2: float
    iota_range: tuple[float, float]
    dL_range: tuple[float, float]
    t_c_range: tuple[float, float]
    phase_c_range: tuple[float, float]
    psi_prior: tuple[float, float]
    ra_prior: tuple[float, float]
    dec_prior: tuple[float, float]

    @property
    def n_dims():
        return 15

    def __init__(
        self,
        gps: int,
        segment_length: int,
        post_trigger_length: int,
        ifos: set[Literal["H1"], Literal["L1"], Literal["V1"]],
        M_c_range: tuple[float, float],
        q_range: tuple[float, float],
        max_s1: float,
        max_s2: float,
        iota_range: tuple[float, float],
        dL_range: tuple[float, float],
        t_c_range: tuple[float, float],
        phase_c_range: tuple[float, float],
        psi_prior: tuple[float, float],
        ra_prior: tuple[float, float],
    ):

        self.likelihood = self.initialize_likelihood()
        self.prior = self.initialize_prior()
        self.likelihood_transforms = self.initialize_likelihood_transforms()
        self.sample_transforms = self.initialize_sample_transforms()

        self.gps = gps
        self.segment_length = segment_length
        self.post_trigger_length = post_trigger_length
        self.f_min = 20
        self.f_max = 1024
        self.ifos = ifos
        self.f_ref = 20

        self.M_c_range = M_c_range
        self.q_range = q_range
        self.max_s1 = max_s1
        self.max_s2 = max_s2
        self.iota_range = iota_range
        self.dL_range = dL_range
        self.t_c_range = t_c_range
        self.phase_c_range = phase_c_range
        self.psi_prior = psi_prior
        self.ra_prior = ra_prior

    def initialize_likelihood(self) -> TransientLikelihoodFD:
        # first, fetch a 4s segment centered on GW150914
        gps = self.gps
        start = gps - (self.segment_length - self.post_trigger_length)
        end = gps + self.post_trigger_length

        ifos = []
        for ifo in self.ifos:
            if ifo not in detector_enum:
                raise ValueError(f"Invalid detector: {ifo}")
            detector_enum[ifo].load_data(
                gps, start, end, self.f_min, self.f_max, psd_pad=16, tukey_alpha=0.2
            )
            ifos.append(detector_enum[ifo])

        waveform = RippleIMRPhenomPv2(f_ref=self.f_ref)

        likelihood = TransientLikelihoodFD(
            ifos=ifos,
            waveform=waveform,
            trigger_time=gps,
            duration=self.segment_length,
            post_trigger_duration=self.post_trigger_length,
        )

        return likelihood

    def initialize_prior(self) -> CombinePrior:

        # Mass prior
        Mc_prior = UniformPrior(
            self.M_c_range[0], self.M_c_range[1], parameter_names=["M_c"]
        )
        q_prior = UniformPrior(self.q_range[0], self.q_range[1], parameter_names=["q"])
        # Spin prior
        s1_prior = UniformSpherePrior(parameter_names=["s1"], max_mag=self.max_s1)
        s2_prior = UniformSpherePrior(parameter_names=["s2"], max_mag=self.max_s1)
        iota_prior = SinePrior(parameter_names=["iota"])
        # Extrinsic prior
        dL_prior = PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"])
        t_c_prior = UniformPrior(-0.05, 0.05, parameter_names=["t_c"])
        phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
        psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
        ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
        dec_prior = CosinePrior(parameter_names=["dec"])

        dL_prior = PowerLawPrior(
            self.dL_range[0],
            self.dL_range[1],
            2.0,
            parameter_names=["d_L"],
        )
        t_c_prior = UniformPrior(
            self.t_c_range[0], self.t_c_range[1], parameter_names=["t_c"]
        )
        phase_c_prior = UniformPrior(
            self.phase_c_range[0], self.phase_c_range[1], parameter_names=["phase_c"]
        )
        psi_prior = UniformPrior(
            self.psi_prior[0], self.psi_prior[1], parameter_names=["psi"]
        )
        ra_prior = UniformPrior(
            self.ra_prior[0], self.ra_prior[1], parameter_names=["ra"]
        )
        dec_prior = CosinePrior(parameter_names=["dec"])

        prior = [
            Mc_prior,
            q_prior,
            s1_prior,
            s2_prior,
            iota_prior,
            dL_prior,
            t_c_prior,
            phase_c_prior,
            psi_prior,
            ra_prior,
            dec_prior,
        ]

        return CombinePrior(prior)

    def initialize_likelihood_transforms(self) -> list[BijectiveTransform]:
        return [
            MassRatioToSymmetricMassRatioTransform,
            SphereSpinToCartesianSpinTransform("s1"),
            SphereSpinToCartesianSpinTransform("s2"),
        ]

    def initialize_sample_tranforms(self) -> list[NtoMTransform]:
        return [
            DistanceToSNRWeightedDistanceTransform(
            gps_time=self.gps, ifos=self.ifos, dL_min=self.dL_range[0], dL_max=self.dL_range[1]
            ),
            GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
            gps_time=self.gps, ifo=list(self.ifos)[0]
            ),
            GeocentricArrivalTimeToDetectorArrivalTimeTransform(
            tc_min=self.t_c_range[0], tc_max=self.t_c_range[1], gps_time=self.gps, ifo=list(self.ifos)[0]
            ),
            SkyFrameToDetectorFrameSkyPositionTransform(gps_time=self.gps, ifos=self.ifos),
            BoundToUnbound(
            name_mapping=(["M_c"], ["M_c_unbounded"]),
            original_lower_bound=self.M_c_range[0],
            original_upper_bound=self.M_c_range[1],
            ),
            BoundToUnbound(
            name_mapping=(["q"], ["q_unbounded"]),
            original_lower_bound=self.q_range[0],
            original_upper_bound=self.q_range[1],
            ),
            BoundToUnbound(
            name_mapping=(["s1_phi"], ["s1_phi_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=2 * jnp.pi,
            ),
            BoundToUnbound(
            name_mapping=(["s2_phi"], ["s2_phi_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=2 * jnp.pi,
            ),
            BoundToUnbound(
            name_mapping=(["iota"], ["iota_unbounded"]),
            original_lower_bound=self.iota_range[0],
            original_upper_bound=self.iota_range[1],
            ),
            BoundToUnbound(
            name_mapping=(["s1_theta"], ["s1_theta_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=jnp.pi,
            ),
            BoundToUnbound(
            name_mapping=(["s2_theta"], ["s2_theta_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=jnp.pi,
            ),
            BoundToUnbound(
            name_mapping=(["s1_mag"], ["s1_mag_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=self.max_s1,
            ),
            BoundToUnbound(
            name_mapping=(["s2_mag"], ["s2_mag_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=self.max_s2,
            ),
            BoundToUnbound(
            name_mapping=(["phase_det"], ["phase_det_unbounded"]),
            original_lower_bound=self.phase_c_range[0],
            original_upper_bound=self.phase_c_range[1],
            ),
            BoundToUnbound(
            name_mapping=(["psi"], ["psi_unbounded"]),
            original_lower_bound=self.psi_prior[0],
            original_upper_bound=self.psi_prior[1],
            ),
            BoundToUnbound(
            name_mapping=(["zenith"], ["zenith_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=jnp.pi,
            ),
            BoundToUnbound(
            name_mapping=(["azimuth"], ["azimuth_unbounded"]),
            original_lower_bound=0.0,
            original_upper_bound=2 * jnp.pi,
            ),
        ]
