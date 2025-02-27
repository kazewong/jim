from jimgw.single_event.detector import H1
from jimgw.single_event.data import Data, PowerSpectrum
import jax.numpy as jnp
import pytest

class TestDataInterface:

    # def test_import_data_from_gwosc():
    #     raise NotImplementedError
    
    def test_user_provide_data(self):
        n_time = 2048

        fake_data = Data(td=jnp.arange(n_time), delta_t = 1)
        fake_psd = PowerSpectrum(values = jnp.ones(n_time//2), frequencies=jnp.arange(n_time//2))

        detector = H1
        with pytest.raises(AssertionError):
            detector.data.frequency_slice(20., 2048.)

        detector.set_data(fake_data)
        detector.set_psd(fake_psd)
        assert detector.data.frequency_slice(0., 0.5)
        with pytest.raises(AssertionError):
            assert detector.data.frequency_slice(0., 2048.)