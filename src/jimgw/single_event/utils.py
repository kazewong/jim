import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from jax.scipy.special import i0e
from jaxtyping import Array, Float


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
    

def spin_to_spin(
    thetaJN: Float, 
    phiJL: Float, 
    theta1: Float, 
    theta2: Float, 
    phi12: Float, 
    chi1: Float, 
    chi2: Float, 
    M_c: Float, 
    eta: Float,
    fRef: Float, 
    phiRef: Float
) -> tuple[Float, Float, Float, Float, Float, Float, Float]:
"""
    Transforming the spin parameters
    
    Parameters:
    -------
    thetaJN: Float
        Zenith angle between the total angular momentum and the line of sight
    phiJL: Float
        Difference between total and orbital angular momentum azimuthal angles
    theta1: Float
        Zenith angle between the spin and orbital angular momenta for the primary object
    theta2: Float
        Zenith angle between the spin and orbital angular momenta for the secondary object
    phi12: Float
        Difference between the azimuthal angles of the individual spin vector projections
        onto the orbital plane
    chi1: Float
        Primary object aligned spin:
    chi2: Float
        Secondary object aligned spin:
    M_c: Float
        The chirp mass
    eta: Float
        The symmetric mass ratio
    fRef: Float
        The reference frequency
    phiRef: Float
        Binary phase at a reference frequency
        
    Returns:
    -------
    iota: Float
        Zenith angle between the orbital angular momentum and the line of sight
    S1x: Float
        The x-component of the primary spin
    S1y: Float
        The y-component of the primary spin
    S1z: Float
        The z-component of the primary spin
    S2x: Float
        The x-component of the secondary spin
    S2y: Float
        The y-component of the secondary spin
    S2z: Float
        The z-component of the secondary spin
"""
    def rotate_y(angle, vec):
        """
        Rotate the vector (x, y, z) about y-axis
        """
        cos_angle = jnp.cos(angle)
        sin_angle = jnp.sin(angle)
        rotation_matrix = np.array([
            [cos_angle, 0, sin_angle],
            [0, 1, 0],
            [-sin_angle, 0, cos_angle]
        ])
        rotated_vec = jnp.dot(rotation_matrix, vec)
        return rotated_vec
    
    def rotate_z(angle, vec):
        """
        Rotate the vector (x, y, z) about z-axis
        """
        cos_angle = jnp.cos(angle)
        sin_angle = jnp.sin(angle)
        rotation_matrix = jnp.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
        rotated_vec = jnp.dot(rotation_matrix, vec)
        return rotated_vec
    
    LNh = jnp.array([0., 0., 1.])
    
    s1hat = jnp.array([
        jnp.sin(theta1)*jnp.cos(phiRef), 
        jnp.sin(theta1)*jnp.sin(phiRef), 
        jnp.cos(theta1)
    ])
    s2hat = jnp.array([
        jnp.sin(theta2) * jnp.cos(phi12+phiRef), 
        jnp.sin(theta2) * jnp.sin(phi12+phiRef), 
        jnp.cos(theta2)
    ])
    
    temp = (1 / eta / 2 - 1)
    q = temp - (temp ** 2 - 1) ** 0.5
    m1, m2 = Mc_q_to_m1m2(M_c, q)
    v0 = jnp.cbrt((m1+m2) * MTsun_SI * jnp.pi * fRef)
    
    Lmag = ((m1+m2)*(m1+m2)*eta/v0) * (1.0 + v0*v0*(1.5 + eta/6.0))
    s1 = m1 * m1 * chi1 * s1hat
    s2 = m2 * m2 * chi2 * s2hat
    J = s1 + s2 + jnp.array([0., 0., Lmag])

    Jhat = J / jnp.linalg.norm(J)
    theta0 = jnp.arccos(Jhat[2])
    phi0 = jnp.arctan2(Jhat[1], Jhat[0])

    # Rotation 1:
    s1hat = rotate_z(-phi0, s1hat)
    s2hat = rotate_z(-phi0, s2hat)

    # Rotation 2:
    LNh = rotate_y(-theta0, LNh)
    s1hat = rotate_y(-theta0, s1hat)
    s2hat = rotate_y(-theta0, s2hat)
    
    # Rotation 3:
    LNh = rotate_z(phiJL - jnp.pi, LNh)
    s1hat = rotate_z(phiJL - jnp.pi, s1hat)
    s2hat = rotate_z(phiJL - jnp.pi, s2hat)
    
    # Compute iota
    N = jnp.array([0.0, jnp.sin(thetaJN), jnp.cos(thetaJN)])
    iota = jnp.arccos(jnp.dot(N, LNh))
    
    thetaLJ = jnp.arccos(LNh[2])
    phiL = jnp.arctan2(LNh[1], LNh[0])
    
    # Rotation 4:
    s1hat = rotate_z(-phiL, s1hat)
    s2hat = rotate_z(-phiL, s2hat)
    N = rotate_z(-phiL, N)
    
    # Rotation 5:
    s1hat = rotate_y(-thetaLJ, s1hat)
    s2hat = rotate_y(-thetaLJ, s2hat)
    N = rotate_y(-thetaLJ, N)

    # Rotation 6:
    phiN = jnp.arctan2(N[1], N[0])
    s1hat = rotate_z(jnp.pi/2.-phiN-phiRef, s1hat)
    s2hat = rotate_z(jnp.pi/2.-phiN-phiRef, s2hat)

    S1 = s1hat * chi1
    S2 = s2hat * chi2
    return iota, S1[0], S1[1], S1[2], S2[0], S2[1], S2[2]


def euler_rotation(delta_x: Float[Array, " 3"]):
    """
    Calculate the rotation matrix mapping the vector (0, 0, 1) to delta_x
    while preserving the origin of the azimuthal angle.

    This is decomposed into three Euler angles, alpha, beta, gamma, which rotate
    about the z-, y-, and z- axes respectively.

    Copied and modified from bilby-cython/geometry.pyx
    """
    norm = jnp.power(
        delta_x[0] * delta_x[0] + delta_x[1] * delta_x[1] + delta_x[2] * delta_x[2], 0.5
    )
    cos_beta = delta_x[2] / norm
    sin_beta = jnp.power(1 - cos_beta**2, 0.5)

    alpha = jnp.atan2(-delta_x[1] * cos_beta, delta_x[0])
    gamma = jnp.atan2(delta_x[1], delta_x[0])

    cos_alpha = jnp.cos(alpha)
    sin_alpha = jnp.sin(alpha)
    cos_gamma = jnp.cos(gamma)
    sin_gamma = jnp.sin(gamma)

    rotation = jnp.array(
        [
            [
                cos_alpha * cos_beta * cos_gamma - sin_alpha * sin_gamma,
                -sin_alpha * cos_beta * cos_gamma - cos_alpha * sin_gamma,
                sin_beta * cos_gamma,
            ],
            [
                cos_alpha * cos_beta * sin_gamma + sin_alpha * cos_gamma,
                -sin_alpha * cos_beta * sin_gamma + cos_alpha * cos_gamma,
                sin_beta * sin_gamma,
            ],
            [-cos_alpha * sin_beta, sin_alpha * sin_beta, cos_beta],
        ]
    )

    return rotation


def zenith_azimuth_to_theta_phi(
    zenith: Float, azimuth: Float, delta_x: Float[Array, " 3"]
) -> tuple[Float, Float]:
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

    theta = jnp.acos(
        rotation[2][0] * sin_zenith * cos_azimuth
        + rotation[2][1] * sin_zenith * sin_azimuth
        + rotation[2][2] * cos_zenith
    )
    phi = jnp.fmod(
        jnp.atan2(
            rotation[1][0] * sin_zenith * cos_azimuth
            + rotation[1][1] * sin_zenith * sin_azimuth
            + rotation[1][2] * cos_zenith,
            rotation[0][0] * sin_zenith * cos_azimuth
            + rotation[0][1] * sin_zenith * sin_azimuth
            + rotation[0][2] * cos_zenith,
        )
        + 2 * jnp.pi,
        (2 * jnp.pi),
    )
    return theta, phi


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


def zenith_azimuth_to_ra_dec(
    zenith: Float, azimuth: Float, gmst: Float, delta_x: Float[Array, " 3"]
) -> tuple[Float, Float]:
    """
    Transforming the azimuthal angle and zenith angle in Earth frame to right ascension and declination.

    Parameters
    ----------
    zenith : Float
            Zenith angle.
    azimuth : Float
            Azimuthal angle.
    gmst : Float
            Greenwich mean sidereal time.
    delta_x : Float
            The vector pointing from the first detector to the second detector.

    Copied and modified from bilby/gw/utils.py

    Returns
    -------
    ra : Float
            Right ascension.
    dec : Float
            Declination.
    """
    theta, phi = zenith_azimuth_to_theta_phi(zenith, azimuth, delta_x)
    ra, dec = theta_phi_to_ra_dec(theta, phi, gmst)
    ra = ra % (2 * jnp.pi)
    return ra, dec


def log_i0(x: Float[Array, " n"]) -> Float[Array, " n"]:
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
