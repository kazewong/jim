# Import packages
from typing import List, Tuple
import lalsimulation as lalsim 
import jax.numpy as jnp
import jax
import numpy as np
jax.config.update('jax_enable_x64', True)

psd_func_dict = {
    'H1': lalsim.SimNoisePSDaLIGOZeroDetHighPower,
    'L1': lalsim.SimNoisePSDaLIGOZeroDetHighPower,
    'V1': lalsim.SimNoisePSDAdvVirgo,
}

def generate_noise(seed: int, f_sampling: int = 2048, duration: int = 4, f_min: float = 30., ifos: List = ['H1', 'L1']):


    # define sampling rate and duration

    delta_t = 1/f_sampling
    tlen = int(round(duration / delta_t))

    freqs = np.fft.rfftfreq(tlen, delta_t)
    delta_f = freqs[1] - freqs[0]

    # we will want to pad low frequencies; the function below applies a
    # prescription to do so smoothly, but this is not really needed: you
    # could just set all values below `fmin` to a constant.
    def pad_low_freqs(f, psd_ref):
        return psd_ref + psd_ref*(f_min-f)*jnp.exp(-(f_min-f))/3

    psd_dict = {}
    for ifo in ifos:
        psd = np.zeros(len(freqs))
        for i,f in enumerate(freqs):
            if f >= f_min:
                psd[i] = psd_func_dict[ifo](f)
            else:
                psd[i] = pad_low_freqs(f, psd_func_dict[ifo](f_min))
        psd_dict[ifo] = jnp.array(psd,dtype=jnp.float64)

    rng_key = jax.random.PRNGKey(seed)
    rng_keys = jax.random.split(rng_key)

    noise_fd_dict = {}
    for ifo, psd in psd_dict.items():
        rng_keys = jax.random.split(rng_keys[0], 3)
        var = psd / (4.*delta_f)  # this is the variance of LIGO noise given the definition of the likelihood function
        noise_real = jax.random.normal(rng_keys[1],shape=(len(psd),))*jnp.sqrt(var)
        noise_imag = jax.random.normal(rng_keys[2],shape=(len(psd),))*jnp.sqrt(var)
        noise_fd_dict[ifo] = noise_real + 1j*noise_imag

    return freqs, psd_dict, noise_fd_dict