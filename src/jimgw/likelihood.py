from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from abc import ABC, abstractmethod
from typing import Tuple
from jimgw.waveform import Waveform
from jimgw.detector import Detector


class LikelihoodBase(ABC):
    """Base class for likelihoods.
    Note that this likelihood class should work for a somehwat general class of problems.
    In light of that, this class would be somewhat abstract, but the idea behind it is this
    handles two main components of a likelihood: the data and the model.

    It should be able to take the data and model and evaluate the likelihood for a given set of parameters.

    """

    @property
    def model(self):
        """The model for the likelihood.
        """
        return self._model

    @property
    def data(self):
        """The data for the likelihood.
        """
        return self._data

    @abstractmethod
    def evalutate(self, params) -> float:
        """Evaluate the likelihood for a given set of parameters.
        """
        raise NotImplementedError

class TransientLikelihoodFD(LikelihoodBase):

    detectors: list[Detector]
    waveform: Waveform

    def __init__(self,
        detectors: list[Detector],
        waveform: Waveform
    ) -> None:
        self.detectors = detectors
        self.waveform = waveform

    def evaluate(self, params) -> float:
        """Evaluate the likelihood for a given set of parameters.
        """
        raise NotImplementedError



class LogLikelihoodTransientFD(object):
    """Object to construct a frequency-domain JAX-based log-likelihood function
    for transient gravitational-wave signals detected by ground-based
    detectors with stationary Gaussian noise.

    Arguments
    ---------
    waveform :
        frequency-domain waveform function, must accept an array of frequencies
        and a set of parameter values, like
        `waveform(frequencies, [param1, param2, ...])`.
    heterodyne : bool
        whether to approximate likelihood through a heteredoyne.   
    earth_rotation : bool
        whether to include Earth's rotation in the antenna pattern.
    """
    def __init__(self, waveform, heterodyne=False, earth_rotation=False):
        self.waveform = waveform
        self.heterodyne = heterodyne
        # whether to include Earth's rotation in the antenna pattern
        # TODO: implement automatic defaults based on IFO names
        self.earth_rotation = earth_rotation
        self.data = {}
        self.psds = {}

    @property
    def ifos(self):
        """Names of interferometers to analyze.
        """
        return list(self.data.keys())

    def add_data(self, ifo, data, **kws):
        """Add frequency-domain strain data for a given detector.

        Arguments
        ---------
        ifo : str
            interferometer name, e.g., 'H1' for LIGO Hanford.
        data : array,FrequencySeries
            frequency-domain strain data.
        """
        if isinstance(data, FrequencySeries):
            self.data[ifo] = data
        else:
            self.data[ifo] = FrequencySeries(data, **kws)

    def add_psd(self, ifo, psd, **kws):
        """Add power spectral density (PSD) for the noise of a given detector.

        Arguments
        ---------
        ifo : str
            interferometer name, e.g., 'H1' for LIGO Hanford.
        psd : array,FrequencySeries
            power spectrum data.
        """
        if isinstance(psd, FrequencySeries):
            self.psd[ifo] = psd
        else:
            self.psd[ifo] = FrequencySeries(psd, **kws)

