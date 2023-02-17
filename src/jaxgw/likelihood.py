import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries

class LogLikelihoodTransientFD(object):
    """Object to construct a frequency-domain JAX-based log-likelihood function
    for transient gravitational-wave signals detected by ground-based
    detectors with stationary Gaussian noise.

    Arguments
    ---------
    waveform :
        frequency-domain waveform function, must accept an array of frequencies
        and a set of parameter values, like
        `waveform(frequencies, [param1, param2, ...])`
    """
    def __init__(self, waveform, heterodyne=True):
        self.waveform = waveform
        self.heterodyne = heterodyne
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
            self.data[ifo] - FrequencySeries(data, **kws)

    def add_psd(self, ifo, data, **kws):
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
            self.psd[ifo] - FrequencySeries(psd, **kws)

