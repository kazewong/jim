from jimgw.single_event.detector import H1
from jimgw.single_event.data import Data, PowerSpectrum
import jax.numpy as jnp
import jax


class TestDataInterface:

    def test_construct_detector():
        raise NotImplementedError
    
    def test_import_data_from_gwosc():
        raise NotImplementedError
    
    def test_user_provide_data():
        n_time = 2048

        fake_data = Data(td=jnp.arange(n_time), delta_t = 1)
        fake_psd = PowerSpectrum(values = jnp.ones(n_time//2), frequencies=jnp.arange(n_time//2))

        detector = H1()

    