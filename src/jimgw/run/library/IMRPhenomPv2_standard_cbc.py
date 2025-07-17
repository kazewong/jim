from jimgw.run.single_event_run_definition import SingleEventRunDefinition

import jax.numpy as jnp

from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    UniformSpherePrior,
)

from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_detector_preset
from jimgw.core.single_event.likelihood import BaseTransientLikelihoodFD, ZeroLikelihood
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2
from jimgw.core.transforms import BoundToUnbound, BijectiveTransform, NtoMTransform
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
)

from typing import Optional, Sequence, Self
import yaml
import logging


class IMRPhenomPv2StandardCBCRunDefinition(SingleEventRunDefinition):
    M_c_range: tuple[float, float]
    q_range: tuple[float, float]
    max_s1: float
    max_s2: float
    iota_range: tuple[float, float]
    dL_range: tuple[float, float]
    # t_c_range: tuple[float, float]
    phase_c_range: tuple[float, float]
    psi_range: tuple[float, float]
    ra_range: tuple[float, float]
    dec_range: tuple[float, float]

    @property
    def n_dims(self):
        return 15

    def __init__(
        self,
        M_c_range: tuple[float, float],
        q_range: tuple[float, float],
        max_s1: float,
        max_s2: float,
        iota_range: tuple[float, float],
        dL_range: tuple[float, float],
        t_c_range: tuple[float, float],
        phase_c_range: tuple[float, float],
        psi_range: tuple[float, float],
        ra_range: tuple[float, float],
        dec_range: tuple[float, float],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.M_c_range = M_c_range
        self.q_range = q_range
        self.max_s1 = max_s1
        self.max_s2 = max_s2
        self.iota_range = iota_range
        self.dL_range = dL_range
        self.t_c_range = t_c_range
        self.phase_c_range = phase_c_range
        self.psi_range = psi_range
        self.ra_range = ra_range
        self.dec_range = dec_range

    def initialize_jim_objects(self):
        self.likelihood = self.initialize_likelihood(
            local_data_prefix=self.local_data_prefix
        )
        self.prior = self.initialize_prior()
        self.likelihood_transforms = self.initialize_likelihood_transforms()
        self.sample_transforms = self.initialize_sample_transforms()

    def initialize_likelihood(
        self, local_data_prefix: Optional[str] = None
    ) -> BaseTransientLikelihoodFD:
        logging.info("Initializing likelihood...")

        gps = self.gps
        start = gps - (self.segment_length - self.post_trigger_length)
        end = gps + self.post_trigger_length

        if self.local_data_prefix is None:
            logging.info("No local data provided, using GWOSC data.")
            psd_start = gps - 2048
            psd_end = gps + 2048
            for ifo in self.ifos:
                if ifo.name not in [
                    detector.name for detector in list(get_detector_preset().values())
                ]:
                    raise ValueError(f"Invalid detector: {ifo}")
                ifo_data = Data.from_gwosc(ifo.name, start, end)
                ifo.set_data(ifo_data)
                ifo_psd = Data.from_gwosc(ifo.name, psd_start, psd_end)
                psd_fftlength = ifo_data.duration * ifo_data.sampling_frequency
                ifo.set_psd(ifo_psd.to_psd(nperseg=psd_fftlength))
        else:
            logging.info(f"Using local data from {local_data_prefix}.")
            # TODO: Load local data from a file, and the PSD correspondingly.
            for ifo in self.ifos:
                if ifo.name not in [
                    detector.name for detector in list(get_detector_preset().values())
                ]:
                    raise ValueError(f"Invalid detector: {ifo}")
                ifo_data = Data.from_file(f"{local_data_prefix}{ifo.name}_data.npz")
                ifo.set_data(ifo_data)
                ifo_psd = PowerSpectrum.from_file(
                    f"{local_data_prefix}{ifo.name}_psd.npz"
                )
                ifo.set_psd(ifo_psd)

        waveform = RippleIMRPhenomPv2(f_ref=self.f_ref)

        likelihood = BaseTransientLikelihoodFD(
            detectors=self.ifos,
            waveform=waveform,
            trigger_time=gps,
            f_min=self.f_min,
            f_max=self.f_max,
        )

        return likelihood

    def initialize_prior(self) -> CombinePrior:
        logging.info("Initializing prior...")
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
            self.psi_range[0], self.psi_range[1], parameter_names=["psi"]
        )
        ra_prior = UniformPrior(
            self.ra_range[0], self.ra_range[1], parameter_names=["ra"]
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

    def initialize_likelihood_transforms(self) -> Sequence[NtoMTransform]:
        logging.info("Initializing likelihood transforms...")
        return [
            MassRatioToSymmetricMassRatioTransform,
            SphereSpinToCartesianSpinTransform("s1"),
            SphereSpinToCartesianSpinTransform("s2"),
        ]

    def initialize_sample_transforms(self) -> Sequence[BijectiveTransform]:
        logging.info("Initializing sample transforms...")
        return [
            DistanceToSNRWeightedDistanceTransform(
                gps_time=self.gps,
                ifos=self.ifos,
                dL_min=self.dL_range[0],
                dL_max=self.dL_range[1],
            ),
            GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
                gps_time=self.gps, ifo=self.ifos[0]
            ),
            GeocentricArrivalTimeToDetectorArrivalTimeTransform(
                tc_min=self.t_c_range[0],
                tc_max=self.t_c_range[1],
                gps_time=self.gps,
                ifo=self.ifos[0],
            ),
            SkyFrameToDetectorFrameSkyPositionTransform(
                gps_time=self.gps, ifos=self.ifos
            ),
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
                original_lower_bound=self.psi_range[0],
                original_upper_bound=self.psi_range[1],
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

    def serialize(self, path: str = "./") -> dict:
        run_dict = super().serialize(path)
        run_dict.update(
            {
                "definition_name": "IMRPhenomPv2StandardCBC",
                "M_c_range": list(self.M_c_range),
                "q_range": list(self.q_range),
                "max_s1": self.max_s1,
                "max_s2": self.max_s2,
                "iota_range": list(self.iota_range),
                "dL_range": list(self.dL_range),
                "t_c_range": list(self.t_c_range),
                "phase_c_range": list(self.phase_c_range),
                "psi_range": list(self.psi_range),
                "ra_range": list(self.ra_range),
                "dec_range": list(self.dec_range),
            }
        )
        with open(path, "w") as f:
            yaml.dump(run_dict, f, default_flow_style=False, sort_keys=False)
        print(f"Run serialized to {path}")

        return run_dict

    @classmethod
    def deserialize(cls, path: str) -> Self:
        print(f"Deserializing run from {path}")
        with open(path, "r") as f:
            run_dict = yaml.safe_load(f)
        run = cls(
            seed=run_dict["seed"],
            gps=run_dict["gps"],
            segment_length=run_dict["segment_length"],
            post_trigger_length=run_dict["post_trigger_length"],
            f_min=run_dict["f_min"],
            f_max=run_dict["f_max"],
            ifos=set(run_dict["ifos"]),
            f_ref=run_dict["f_ref"],
            M_c_range=tuple(run_dict["M_c_range"]),
            q_range=tuple(run_dict["q_range"]),
            max_s1=run_dict["max_s1"],
            max_s2=run_dict["max_s2"],
            iota_range=tuple(run_dict["iota_range"]),
            dL_range=tuple(run_dict["dL_range"]),
            t_c_range=tuple(run_dict["t_c_range"]),
            phase_c_range=tuple(run_dict["phase_c_range"]),
            psi_range=tuple(run_dict["psi_range"]),
            ra_range=tuple(run_dict["ra_range"]),
            dec_range=tuple(run_dict["dec_range"]),
        )
        run.load_flowMC_params(run_dict)
        run.load_single_event_params(run_dict)
        return run


class TestIMRPhenomPv2StandardCBCRunDefinition(IMRPhenomPv2StandardCBCRunDefinition):
    """
    A test run with zero likelihood
    """

    def __init__(self):
        super().__init__(
            seed=123130941092,
            gps=1126259462.4,
            segment_length=4,
            post_trigger_length=2,
            f_min=20,
            f_max=2000,
            ifos={"H1", "L1"},
            f_ref=20,
            M_c_range=(1.0, 100.0),
            q_range=(1.0, 100.0),
            max_s1=0.99,
            max_s2=0.99,
            iota_range=(0.0, jnp.pi),
            dL_range=(1.0, 10000.0),
            t_c_range=(-0.05, 0.05),
            phase_c_range=(0.0, 2 * jnp.pi),
            psi_range=(0.0, jnp.pi),
            ra_range=(0.0, 2 * jnp.pi),
            dec_range=(-jnp.pi / 2, jnp.pi / 2),
        )
        self.likelihood = ZeroLikelihood()
