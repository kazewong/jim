import pytest
import numpy as np
from jimgw.core.single_event.likelihood import BaseTransientLikelihoodFD
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from jimgw.core.single_event.data import Data

class TestBaseTransientLikelihoodFD:
    """
    Organized tests for BaseTransientLikelihoodFD using real detector and waveform implementations.
    """

    @pytest.fixture
    def GW150912_likelihood(self) -> BaseTransientLikelihoodFD:
        """
        Fixture to set up a realistic BaseTransientLikelihoodFD instance using GWOSC data and power spectral density.
        """
        gps = 1126259462.4
        start = gps - 2
        end = gps + 2
        psd_start = gps - 2048
        psd_end = gps + 2048
        fmin = 20.0
        fmax = 1024.0

        # Initialize detectors and set data/PSD
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            data = Data.from_gwosc(ifo.name, start, end)
            ifo.set_data(data)
            psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
            psd_fftlength = data.duration * data.sampling_frequency
            ifo.set_psd(psd_data.to_psd(nperseg=psd_fftlength))

        waveform = RippleIMRPhenomD(f_ref=20.0)
        likelihood = BaseTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )
        return likelihood

    def test_likelihood_initialization(self, GW150912_likelihood: BaseTransientLikelihoodFD):
        """
        Test initialization and attributes of BaseTransientLikelihoodFD with realistic setup.
        """
        likelihood = GW150912_likelihood
        assert isinstance(likelihood, BaseTransientLikelihoodFD)
        assert np.allclose(likelihood.frequencies, [20.0, (20.0 + 1024.0) / 2, 1024.0])
        assert likelihood.trigger_time == 1126259462.4
        assert hasattr(likelihood, "gmst")

    def test_likelihood_evaluation(self, GW150912_likelihood: BaseTransientLikelihoodFD):
        """
        Test the evaluation of the likelihood with realistic parameters.
        """
        likelihood = GW150912_likelihood
        # Example parameters for testing
        params = {
            "M_c": 30.0,
            "eta": 0.249,
            "s1_z": 0.0,
            "s2_z": 0.0,
            "d_L": 400.0,
            "phase_c": 0.0,
            "t_c": 0.0,
            "iota": 0.0,
            "ra": 1.375,
            "dec": -1.2108,
            "gmst": likelihood.gmst,
            "psi": 0.0,
        }
        log_likelihood = likelihood.evaluate(params, {})
        assert np.isfinite(log_likelihood), "Log likelihood should be finite"