import jax.numpy as jnp
from jax.scipy.special import i0e, logsumexp
from jax import jit
from jaxtyping import Float, Array


@jit
def inner_product(
    h1: Float[Array, " n_sample"],
    h2: Float[Array, " n_sample"],
    frequency: Float[Array, " n_sample"],
    psd: Float[Array, " n_sample"],
) -> Float:
    """
        Evaluating the inner product of two waveforms h1 and h2 with the psd.

    Do psd interpolation outside the inner product loop to speed up the evaluation

        Parameters
        ----------
        h1 : Float[Array, "n_sample"]
                First waveform. Can be complex.
        h2 : Float[Array, "n_sample"]
                Second waveform. Can be complex.
        frequency : Float[Array, "n_sample"]
                Frequency array.
        psd : Float[Array, "n_sample"]
                Power spectral density.

        Returns
        -------
        Float
                Inner product of h1 and h2 with the psd.
    """
    # psd_interp = jnp.interp(frequency, psd_frequency, psd)
    df = frequency[1] - frequency[0]
    integrand = jnp.conj(h1) * h2 / psd
    return 4.0 * jnp.real(jnp.trapz(integrand, dx=df))


@jit
def m1m2_to_Mq(m1: Float, m2: Float):
    """
    Transforming the primary mass m1 and secondary mass m2 to the Total mass M
    and mass ratio q.

    Parameters
        ----------
        m1 : Float
                Primary mass.
        m2 : Float
                Secondary mass.

        Returns
        -------
        M_tot : Float
                Total mass.
        q : Float
                Mass ratio.
    """
    M_tot = jnp.log(m1 + m2)
    q = jnp.log(m2 / m1) - jnp.log(1 - m2 / m1)
    return M_tot, q


@jit
def Mq_to_m1m2(trans_M_tot: Float, trans_q: Float):
    """
    Transforming the Total mass M and mass ratio q to the primary mass m1 and
    secondary mass m2.

    Parameters
    ----------
    M_tot : Float
            Total mass.
    q : Float
            Mass ratio.

    Returns
    -------
    m1 : Float
            Primary mass.
    m2 : Float
            Secondary mass.
    """
    M_tot = jnp.exp(trans_M_tot)
    q = 1.0 / (1 + jnp.exp(-trans_q))
    m1 = M_tot / (1 + q)
    m2 = m1 * q
    return m1, m2


@jit
def Mc_q_to_m1m2(Mc: Float, q: Float) -> tuple[Float, Float]:
    """
    Transforming the chirp mass Mc and mass ratio q to the primary mass m1 and
    secondary mass m2.

    Parameters
    ----------
    Mc : Float
            Chirp mass.
    q : Float
            Mass ratio.

    Returns
    -------
    m1 : Float
            Primary mass.
    m2 : Float
            Secondary mass.
    """
    eta = q / (1 + q) ** 2
    M_tot = Mc / eta ** (3.0 / 5)
    m1 = M_tot / (1 + q)
    m2 = m1 * q
    return m1, m2


def ra_dec_to_theta_phi(ra: Float, dec: Float, gmst: Float) -> tuple[Float, Float]:
    """
    Transforming the right ascension ra and declination dec to the polar angle
    theta and azimuthal angle phi.

    Parameters
    ----------
    ra : Float
            Right ascension.
    dec : Float
            Declination.
    gmst : Float
            Greenwich mean sidereal time.

    Returns
    -------
    theta : Float
            Polar angle.
    phi : Float
            Azimuthal angle.
    """
    phi = ra - gmst
    theta = jnp.pi / 2 - dec
    return theta, phi


def log_i0(x):
    """
    A numerically stable method to evaluate log of
    a modified Bessel function of order 0.
    It is used in the phase-marginalized likelihood.

    Parameters
    ==========
    x: array-like
        Value(s) at which to evaluate the function

    Returns
    =======
    array-like:
        The natural logarithm of the bessel function
    """
    return jnp.log(i0e(x)) + x


def original_likelihood(params, h_sky, detectors, freqs, align_time, **kwargs):
    log_likelihood = 0.
    df = freqs[1] - freqs[0]
    for detector in detectors:
        h_dec = (
            detector.fd_response(freqs, h_sky, params) * align_time
        )
        match_filter_SNR = (
            4
            * jnp.sum(
                (jnp.conj(h_dec) * detector.data) / detector.psd * df
            ).real
        )
        optimal_SNR = (
            4
            * jnp.sum(
                jnp.conj(h_dec) * h_dec / detector.psd * df
            ).real
        )
        log_likelihood += match_filter_SNR - optimal_SNR / 2

    return log_likelihood


def phase_marginalized_likelihood(params, h_sky, detectors, freqs, align_time, **kwargs):
    log_likelihood = 0.
    complex_d_inner_h = 0.
    df = freqs[1] - freqs[0]
    for detector in detectors:
        h_dec = (
            detector.fd_response(freqs, h_sky, params) * align_time
        )
        complex_d_inner_h += (
            4
            * jnp.sum(
                (jnp.conj(h_dec) * detector.data) / detector.psd * df
            )
        )
        optimal_SNR = (
            4
            * jnp.sum(
                jnp.conj(h_dec) * h_dec / detector.psd * df
            ).real
        )
        log_likelihood += -optimal_SNR / 2

    log_likelihood += log_i0(jnp.absolute(complex_d_inner_h))

    return log_likelihood


def time_marginalized_likelihood(params, h_sky, detectors, freqs, align_time, **kwargs):
    log_likelihood = 0.
    df = freqs[1] - freqs[0]
    # using <h|d> instead of <d|h>
    complex_h_inner_d = jnp.zeros_like(freqs)
    for detector in detectors:
        h_dec = (
            detector.fd_response(freqs, h_sky, params) * align_time
        )
        complex_h_inner_d += (
            4 * h_dec * jnp.conj(detector.data) / detector.psd * df
        )
        optimal_SNR = (
            4
            * jnp.sum(
                jnp.conj(h_dec) * h_dec / detector.psd * df
            ).real
        )
        log_likelihood += -optimal_SNR / 2

    # padding the complex_d_inner_h before feeding to the fft
    # lower and higher frequency padding
    pad_low = jnp.arange(0, freqs[0], df)
    pad_high = jnp.arange(freqs[1], kwargs['sampling_rate'], df)
    complex_h_inner_d = jnp.concatenate((pad_low, complex_h_inner_d, pad_high))
    fft_h_inner_d = jnp.fft.fft(complex_h_inner_d)
    # abusing the fftfreq to get the corresponding tc array
    tc_array = jnp.fft.fftfreq(n=len(fft_h_inner_d), d=df)

    # fetch the range of valid tc
    tc_range = kwargs['tc_range']
    # set the values to -inf when it is outside the tc range
    # so that they will disappear after the logsumexp
    fft_h_inner_d = jnp.where(
        tc_array > tc_range[0] and tc_array < tc_range[1],
        fft_h_inner_d,
        jnp.zeros_like(fft_h_inner_d) - jnp.inf
    )

    # using the logsumexp to marginalize over the tc prior range
    log_likelihood += logsumexp(fft_h_inner_d) - jnp.log(len(fft_h_inner_d))

    return log_likelihood


def phase_time_marginalized_likelihood(params, h_sky, detectors, freqs, align_time, **kwargs):
    log_likelihood = 0.
    df = freqs[1] - freqs[0]
    # using <h|d> instead of <d|h>
    complex_h_inner_d = jnp.zeros_like(freqs)
    for detector in detectors:
        h_dec = (
            detector.fd_response(freqs, h_sky, params) * align_time
        )
        complex_h_inner_d += (
            4 * h_dec * jnp.conj(detector.data) / detector.psd * df
        )
        optimal_SNR = (
            4
            * jnp.sum(
                jnp.conj(h_dec) * h_dec / detector.psd * df
            ).real
        )
        log_likelihood += -optimal_SNR / 2

    # padding the complex_d_inner_h before feeding to the fft
    # lower and higher frequency padding
    pad_low = jnp.arange(0, freqs[0], df)
    pad_high = jnp.arange(freqs[1], kwargs['sampling_rate'], df)
    complex_h_inner_d = jnp.concatenate((pad_low, complex_h_inner_d, pad_high))
    fft_h_inner_d = jnp.fft.fft(complex_h_inner_d)
    # abusing the fftfreq to get the corresponding tc array
    tc_array = jnp.fft.fftfreq(n=len(fft_h_inner_d), d=df)

    # fetch the range of valid tc
    tc_range = kwargs['tc_range']
    # set the values to -inf when it is outside the tc range
    # so that they will disappear after the logsumexp
    log_i0_abs_fft = jnp.where(
        tc_array > tc_range[0] and tc_array < tc_range[1],
        log_i0(jnp.absolute(fft_h_inner_d)),
        jnp.zeros_like(fft_h_inner_d) - jnp.inf
    )

    # using the logsumexp to marginalize over the tc prior range
    log_likelihood += logsumexp(log_i0_abs_fft) - jnp.log(len(fft_h_inner_d))

    return log_likelihood
