# Import packages
from typing import List, Tuple
import jax.numpy as jnp
import jax
import numpy as np

# This is needed for the noise generation to have enough precision to work
jax.config.update("jax_enable_x64", True)


def generate_fd_noise(
    seed: int,
    f_sampling: int = 2048,
    duration: int = 4,
    f_min: float = 30.0,
    psd_funcs: dict = {
        "H1": None,
    },
):
    """
    Generate frequency domain noise for a given set of detectors or specific PSD.
    """
    # define sampling rate and duration
    delta_t = 1 / f_sampling
    tlen = int(round(duration / delta_t))

    freqs = np.fft.rfftfreq(tlen, delta_t)
    delta_f = freqs[1] - freqs[0]

    # we will want to pad low frequencies; the function below applies a
    # prescription to do so smoothly, but this is not really needed: you
    # could just set all values below `fmin` to a constant.
    def pad_low_freqs(f, psd_ref):
        return psd_ref + psd_ref * (f_min - f) * jnp.exp(-(f_min - f)) / 3

    psd_dict = {}
    for ifo in psd_funcs.keys():
        psd = np.zeros(len(freqs))
        for i, f in enumerate(freqs):
            if f >= f_min:
                psd[i] = psd_funcs[ifo](f)
            else:
                psd[i] = pad_low_freqs(f, psd_funcs[ifo](f_min))
        psd_dict[ifo] = jnp.array(psd, dtype=jnp.float64)

    rng_key = jax.random.PRNGKey(seed)
    rng_keys = jax.random.split(rng_key)

    noise_fd_dict = {}
    for ifo, psd in psd_dict.items():
        rng_keys = jax.random.split(rng_keys[0], 3)
        # this is the variance of LIGO noise given the definition of the likelihood function
        var = psd / (4.0 * delta_f)
        noise_real = jax.random.normal(rng_keys[1], shape=(len(psd),)) * jnp.sqrt(var)
        noise_imag = jax.random.normal(rng_keys[2], shape=(len(psd),)) * jnp.sqrt(var)
        noise_fd_dict[ifo] = noise_real + 1j * noise_imag

    return freqs, psd_dict, noise_fd_dict


def generate_td_noise(
    seed: int,
    f_sampling: int = 2048,
    duration: int = 4,
    f_min: float = 30.0,
    psd_funcs: dict = {
        "H1": None,
    },
):
    """
    Generate time domain noise for a given set of detectors or specific PSD.
    """

    delta_t = 1 / f_sampling
    tlen = int(round(duration / delta_t))
    ts = jnp.linspace(0, duration, tlen)

    _, psd_dict, noise_fd_dict = generate_fd_noise(
        seed, duration=duration, f_sampling=f_sampling, psd_funcs=psd_funcs, f_min=f_min
    )

    noise_td_dict = {}
    for ifo, psd in noise_fd_dict.items():
        noise_td_dict[ifo] = jnp.fft.irfft(noise_fd_dict[ifo]) * f_sampling

    return ts, noise_td_dict
