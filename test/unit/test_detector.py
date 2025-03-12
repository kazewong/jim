from jimgw.single_event.detector import H1
from jimgw.single_event.data import Data, PowerSpectrum
import jax.numpy as jnp
import numpy as np
from copy import deepcopy
import pytest

class TestDataInterface:

    def setup_method(self):
        # create some dummy data
        self.f_samp = 2048
        self.duration = 4
        self.epoch = 2.
        self.name = 'Dummy'
        delta_t = 1 / self.f_samp
        n_time = int(self.duration / delta_t)
        self.data = Data(td=np.ones(n_time), delta_t=delta_t,
                         name=self.name, epoch=self.epoch)

        # create a dummy PSD spanning [20, 512] Hz
        delta_f = 1 / self.duration
        self.psd_band = (20, 512)
        psd_min, psd_max = self.psd_band
        freqs = np.arange(int(psd_max / delta_f)) * delta_f
        freqs_psd = freqs[freqs >= psd_min]
        self.psd = PowerSpectrum(np.ones_like(freqs_psd),
                                 frequencies=freqs_psd,
                                 name=self.name)

    def test_data(self):
        """Test data manipulation.
        """
        # check basic attributes
        assert self.data.name == 'Dummy'
        assert self.data.epoch == self.epoch
        assert self.data.duration == self.duration
        assert self.data.delta_t == 1 / self.f_samp
        assert len(self.data.td) == int(self.f_samp * self.duration)
        # by default, the Data class should have pre-assigned a Tukey window
        assert len(self.data.window) == len(self.data.td)
        # the boolean should be true if data are present
        assert bool(self.data)

        # check FFTing
        # initially the FD data should be zero, and but its length should
        # be correct and match np.fft.rfftfreq
        assert not self.data.has_fd
        assert np.all(self.data.fd == 0)
        fftfreq = np.fft.rfftfreq(len(self.data.td), self.data.delta_t)
        assert len(self.data.fd) == len(fftfreq)
        assert self.data.n_freq == len(fftfreq)

        # now, requesting a frequency slice should trigger an FFT computation,
        # the result of which will be stured in data.fd; this should be the
        # same as calling data.fft() with the default window
        data_copy = deepcopy(self.data)
        # manually compute the FFT with the right dimensions to compare
        fftdata = np.fft.rfft(self.data.td*self.data.window)*self.data.delta_t

        # check that frequency slice does the right thing
        fmin, fmax = self.psd_band
        data_slice, freq_slice = self.data.frequency_slice(fmin, fmax)
        # note that the FFT requires float64 or it might be off
        freq_mask = (fftfreq >= fmin) & (fftfreq <= fmax)
        assert np.allclose(self.data.fd, fftdata)
        assert np.allclose(data_slice, fftdata[freq_mask])
        assert np.allclose(freq_slice, fftfreq[freq_mask])

        # check that calling data.fft() does the same thing
        assert not data_copy.has_fd
        data_copy.fft()
        assert np.allclose(data_copy.fd, fftdata)
        data_slice1, freq_slice1 = data_copy.frequency_slice(fmin, fmax)
        assert np.allclose(data_slice, data_slice1)
        assert np.allclose(freq_slice, freq_slice1)

        # finally check that we can produce a Welch PSD
        psd = self.data.to_psd(nperseg=self.data.n_time//2)
        # assert np.allclose(psd.frequencies, self.data.frequencies)
        # assert psd.n_freq == self.data.n_freq
    
    # def test_user_provide_data(self):

    #     detector = H1
    #     with pytest.raises(AssertionError):
    #         detector.data.frequency_slice(20., 2048.)

    #     detector.set_data(data)
    #     detector.set_psd(psd)
    #     assert detector.data.frequency_slice(0., 0.5)
    #     with pytest.raises(AssertionError):
    #         assert detector.data.frequency_slice(0., 2048.)