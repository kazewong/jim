from jimgw.run_manager.run import Run
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



class IMRPhenomPv2StandardCBCRun(Run):

    gps: int # GPS time of the event trigger
    segment_length: int # Length of the segment
    post_trigger_length: int # Length of segment after the trigger
    f_min: int # Minimum frequency
    f_max: int # Maximum frequency
    ifos: set[Literal["H1"], Literal["L1"], Literal["V1"]]
    f_ref: int # Reference frequency

    @property
    def n_dims():
        return 15

    def __init__(self,
                 ):
        
        self.likelihood = self.initialize_likelihood()
        self.prior = self.initialize_prior()
        self.likelihood_transforms = self.initialize_likelihood_transforms()
        self.sample_transforms = self.initialize_sample_transforms()
    
    def initialize_likelihood(self) -> TransientLikelihoodFD:
        # first, fetch a 4s segment centered on GW150914
        gps = 1126259462.4
        start = gps - 2
        end = gps + 2
        fmin = 20.0
        fmax = 1024.0

        ifos = [H1, L1]

        H1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
        L1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

        waveform = RippleIMRPhenomPv2(f_ref=20)

        likelihood = TransientLikelihoodFD(
            [H1, L1], waveform=waveform, trigger_time=gps, duration=4, post_trigger_duration=2
        )
    
    def initialize_prior(self) -> CombinePrior:

        prior = []

        # Mass prior
        M_c_min, M_c_max = 10.0, 80.0
        q_min, q_max = 0.125, 1.0
        Mc_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
        q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])

        prior = prior + [Mc_prior, q_prior]

        # Spin prior
        s1_prior = UniformSpherePrior(parameter_names=["s1"])
        s2_prior = UniformSpherePrior(parameter_names=["s2"])
        iota_prior = SinePrior(parameter_names=["iota"])

        prior = prior + [
            s1_prior,
            s2_prior,
            iota_prior,
        ]

        # Extrinsic prior
        dL_prior = PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"])
        t_c_prior = UniformPrior(-0.05, 0.05, parameter_names=["t_c"])
        phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
        psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
        ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
        dec_prior = CosinePrior(parameter_names=["dec"])

        prior = prior + [
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
            DistanceToSNRWeightedDistanceTransform(gps_time=gps, ifos=ifos, dL_min=dL_prior.xmin, dL_max=dL_prior.xmax),
            GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=ifos[0]),
            GeocentricArrivalTimeToDetectorArrivalTimeTransform(tc_min=t_c_prior.xmin, tc_max=t_c_prior.xmax, gps_time=gps, ifo=ifos[0]),
            SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
            BoundToUnbound(name_mapping = (["M_c"], ["M_c_unbounded"]), original_lower_bound=M_c_min, original_upper_bound=M_c_max),
            BoundToUnbound(name_mapping = (["q"], ["q_unbounded"]), original_lower_bound=q_min, original_upper_bound=q_max),
            BoundToUnbound(name_mapping = (["s1_phi"], ["s1_phi_unbounded"]) , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
            BoundToUnbound(name_mapping = (["s2_phi"], ["s2_phi_unbounded"]) , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
            BoundToUnbound(name_mapping = (["iota"], ["iota_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
            BoundToUnbound(name_mapping = (["s1_theta"], ["s1_theta_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
            BoundToUnbound(name_mapping = (["s2_theta"], ["s2_theta_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
            BoundToUnbound(name_mapping = (["s1_mag"], ["s1_mag_unbounded"]) , original_lower_bound=0.0, original_upper_bound=0.99),
            BoundToUnbound(name_mapping = (["s2_mag"], ["s2_mag_unbounded"]) , original_lower_bound=0.0, original_upper_bound=0.99),
            BoundToUnbound(name_mapping = (["phase_det"], ["phase_det_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
            BoundToUnbound(name_mapping = (["psi"], ["psi_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
            BoundToUnbound(name_mapping = (["zenith"], ["zenith_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
            BoundToUnbound(name_mapping = (["azimuth"], ["azimuth_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
        ]