from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from abc import ABC

class LikelihoodBase(ABC):
    """Base class for likelihoods.

    """

    def __init__(self, 
                waveform,
                detectors,
                heterodyne: bool = False):
        self.waveform = waveform
        self.heterodyne = heterodyne
        # whether to include Earth's rotation in the antenna pattern



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

