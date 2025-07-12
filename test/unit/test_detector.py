import jax
import jax.numpy as jnp
from copy import deepcopy
from scipy.signal import welch
from jimgw.single_event.data import Data, PowerSpectrum


class TestDataInterface:
    def setup_method(self):
        # create some dummy data
        self.f_samp = 2048
        self.duration = 4
        self.epoch = 2.0
        self.name = "Dummy"
        delta_t = 1 / self.f_samp
        n_time = int(self.duration / delta_t)
        self.data = Data(
            td=jnp.ones(n_time), delta_t=delta_t, name=self.name, epoch=self.epoch
        )

        # create a dummy PSD spanning [20, 512] Hz
        delta_f = 1 / self.duration
        self.psd_band = (20, 512)
        psd_min, psd_max = self.psd_band
        freqs = jnp.arange(int(psd_max / delta_f)) * delta_f
        freqs_psd = freqs[freqs >= psd_min]
        self.psd = PowerSpectrum(
            jnp.ones_like(freqs_psd), frequencies=freqs_psd, name=self.name
        )

    def test_data(self):
        """Test data manipulation."""
        # check basic attributes
        assert self.data.name == "Dummy"
        assert self.data.epoch == self.epoch
        assert self.data.duration == self.duration
        assert self.data.delta_t == 1 / self.f_samp
        assert len(self.data.td) == int(self.f_samp * self.duration)
        # by default, the Data class should have pre-assigned a Tukey window
        assert len(self.data.window) == len(self.data.td)
        # the boolean should be true if data are present
        assert bool(self.data)

        # check FFTing
        # initially the FD data should be zero, but its length should
        # be correct and match jnp.fft.rfftfreq
        assert not self.data.has_fd
        assert jnp.all(self.data.fd == 0)
        fftfreq = jnp.fft.rfftfreq(len(self.data.td), self.data.delta_t)
        assert len(self.data.fd) == len(fftfreq)
        assert self.data.n_freq == len(fftfreq)

        # now, requesting a frequency slice should trigger an FFT computation,
        # the result of which will be stored in data.fd; this should be the
        # same as calling data.fft() with the default window
        data_copy = deepcopy(self.data)
        # manually compute the FFT with the right dimensions to compare
        fftdata = jnp.fft.rfft(self.data.td * self.data.window) * self.data.delta_t

        # check that frequency slice does the right thing
        fmin, fmax = self.psd_band
        data_slice, freq_slice = self.data.frequency_slice(fmin, fmax)
        # note that the FFT requires float64 or it might be off
        freq_mask = (fftfreq >= fmin) & (fftfreq <= fmax)
        assert jnp.allclose(self.data.fd, fftdata)
        assert jnp.allclose(data_slice, fftdata[freq_mask])
        assert jnp.allclose(freq_slice, fftfreq[freq_mask])

        # check that calling data.fft() does the same thing
        assert not data_copy.has_fd
        data_copy.fft()
        assert jnp.allclose(data_copy.fd, fftdata)
        data_slice1, freq_slice1 = data_copy.frequency_slice(fmin, fmax)
        assert jnp.allclose(data_slice, data_slice1)
        assert jnp.allclose(freq_slice, freq_slice1)

    def test_psd(self):
        """Test PSD manipulation."""
        # check basic attributes of dummy PSD
        assert self.psd.name == "Dummy"
        assert self.psd.n_freq == len(self.psd.frequencies)
        assert jnp.all(self.psd.frequencies >= self.psd_band[0])
        assert jnp.all(self.psd.frequencies <= self.psd_band[1])

        # check PSD frequency slice
        sliced_psd, freq_slice = self.psd.frequency_slice(*self.psd_band)
        assert jnp.allclose(sliced_psd, self.psd.values)
        assert jnp.allclose(freq_slice, self.psd.frequencies)

        # finally check that we can a Welch PSD from data
        nperseg = self.data.n_time // 2
        psd_auto = self.data.to_psd(nperseg=nperseg)
        freq_manual, psd_manual = welch(self.data.td, fs=self.f_samp, nperseg=nperseg)
        assert jnp.allclose(psd_auto.frequencies, freq_manual)
        assert jnp.allclose(psd_auto.values, psd_manual)

        # check interpolation of PSD to data frequency grid
        psd_interp = self.psd.interpolate(self.data.frequencies)
        assert isinstance(psd_interp, PowerSpectrum)

        # check drawing frequency domain data from PSD
        fd_data = self.psd.simulate_data(jax.random.PRNGKey(0))

        # the variance of the simulated data should equal PSD / (4 * delta_f)
        target_var = self.psd.values / (4 * self.psd.delta_f)
        assert jnp.allclose(jnp.var(fd_data.real), target_var, rtol=1e-1)
        assert jnp.allclose(jnp.var(fd_data.imag), target_var, rtol=1e-1)

        # the integral of the PSD should equal the variance of the TD data
        fd_data_white = fd_data / jnp.sqrt(self.psd.values / 2 / self.psd.delta_t)
        td_data_white = jnp.fft.irfft(fd_data_white) / self.psd.delta_t
        assert jnp.allclose(jnp.var(td_data_white), 1, rtol=1e-1)

    # def test_user_provide_data(self):

    #     detector = H1
    #     with pytest.raises(AssertionError):
    #         detector.data.frequency_slice(20., 2048.)

    #     detector.set_data(data)
    #     detector.set_psd(psd)
    #     assert detector.data.frequency_slice(0., 0.5)
    #     with pytest.raises(AssertionError):
    #         assert detector.data.frequency_slice(0., 2048.)
