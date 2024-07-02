import jax.numpy as jnp
from jax import jit
from jax.scipy.integrate import trapezoid
from jax.scipy.special import i0e
from jaxtyping import Array, Float


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
    return 4.0 * jnp.real(trapezoid(integrand, dx=df))


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

@jit
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


@jit
def theta_phi_to_ra_dec(theta: Float, phi: Float, gmst: Float) -> tuple[Float, Float]:
    """
    Transforming the polar angle and azimuthal angle to right ascension and declination.

    Parameters
    ----------
    theta : Float
            Polar angle.
    phi : Float
            Azimuthal angle.
    gmst : Float
            Greenwich mean sidereal time.

    Returns
    -------
    ra : Float
            Right ascension.
    dec : Float
            Declination.
    """
    ra = phi + gmst
    dec = jnp.pi / 2 - theta
    return ra, dec


@jit
def euler_rotation(delta_x: tuple[Float, Float, Float]):
    """
    Calculate the rotation matrix mapping the vector (0, 0, 1) to delta_x
    while preserving the origin of the azimuthal angle.

    This is decomposed into three Euler angles, alpha, beta, gamma, which rotate
    about the z-, y-, and z- axes respectively.

    Copied and modified from bilby-cython/geometry.pyx
    """
    norm = jnp.power(delta_x[0] * delta_x[0] + delta_x[1] * delta_x[1] + delta_x[2] * delta_x[2], 0.5)
    cos_beta = delta_x[2] / norm
    sin_beta = jnp.power(1 - cos_beta**2, 0.5)

    alpha = jnp.atan2(- delta_x[1] * cos_beta, delta_x[0])
    gamma = jnp.atan2(delta_x[1], delta_x[0])

    cos_alpha = jnp.cos(alpha)
    sin_alpha = jnp.sin(alpha)
    cos_gamma = jnp.cos(gamma)
    sin_gamma = jnp.sin(gamma)

    rotation = jnp.empty((3, 3))

    rotation[0][0] = cos_alpha * cos_beta * cos_gamma - sin_alpha * sin_gamma
    rotation[1][0] = cos_alpha * cos_beta * sin_gamma + sin_alpha * cos_gamma
    rotation[2][0] = -cos_alpha * sin_beta
    rotation[0][1] = -sin_alpha * cos_beta * cos_gamma - cos_alpha * sin_gamma
    rotation[1][1] = -sin_alpha * cos_beta * sin_gamma + cos_alpha * cos_gamma
    rotation[2][1] = sin_alpha * sin_beta
    rotation[0][2] = sin_beta * cos_gamma
    rotation[1][2] = sin_beta * sin_gamma
    rotation[2][2] = cos_beta

    return rotation


@jit
def zenith_azimuth_to_theta_phi(zenith: Float, azimuth: Float, delta_x: tuple[Float, Float, Float]) -> tuple[Float, Float]:
    """
    Transforming the azimuthal angle and zenith angle in Earth frame to the polar angle and azimuthal angle in sky frame.

    Copied and modified from bilby-cython/geometry.pyx

    Parameters
    ----------
    zenith : Float
            Zenith angle.
    azimuth : Float
            Azimuthal angle.
    delta_x : Float
            The vector pointing from the first detector to the second detector.

    Returns
    -------
    theta : Float
            Polar angle.
    phi : Float
            Azimuthal angle.
    """
    sin_azimuth = jnp.sin(azimuth)
    cos_azimuth = jnp.cos(azimuth)
    sin_zenith = jnp.sin(zenith)
    cos_zenith = jnp.cos(zenith)

    rotation = euler_rotation(delta_x)

    theta = jnp.acos(rotation[2][0] * sin_zenith * cos_azimuth + rotation[2][1] * sin_zenith * sin_azimuth + rotation[2][2] * cos_zenith
    )
    phi = jnp.fmod(
            jnp.atan2(
                rotation[1][0] * sin_zenith * cos_azimuth
                + rotation[1][1] * sin_zenith * sin_azimuth
                + rotation[1][2] * cos_zenith,
                rotation[0][0] * sin_zenith * cos_azimuth
                + rotation[0][1] * sin_zenith * sin_azimuth
                + rotation[0][2] * cos_zenith)
            + 2 * jnp.pi,
            (2 * jnp.pi)
            )
    return theta, phi


@jit
def azimuth_zenith_to_ra_dec(azimuth: Float, zenith: Float, geocent_time: Float, ifos: list) -> tuple[Float, Float]:
    """
    Transforming the azimuthal angle and zenith angle in Earth frame to right ascension and declination.

    Parameters
    ----------
    azimuth : Float
            Azimuthal angle.
    zenith : Float
            Zenith angle.

    Copied and modified from bilby/gw/utils.py

    Returns
    -------
    ra : Float
            Right ascension.
    dec : Float
            Declination.
    """
    delta_x = ifos[0].vertex - ifos[1].vertex
    theta, phi = zenith_azimuth_to_theta_phi(zenith, azimuth, delta_x)
    gmst = greenwich_mean_sidereal_time(geocent_time)
    ra, dec = theta_phi_to_ra_dec(theta, phi, gmst)
    ra = ra % (2 * jnp.pi)
    return ra, dec


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
