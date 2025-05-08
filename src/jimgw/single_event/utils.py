import jax.numpy as np
from jax.scipy.integrate import trapezoid
from jaxtyping import Array, Float, Complex
from typing import Optional

from jimgw.constants import MTSUN


def complex_inner_product(
    h1: Float[Array, " n_freq"],
    h2: Float[Array, " n_freq"],
    psd: Float[Array, " n_freq"],
    frequency: Optional[Float[Array, " n_freq"]] = None,
    df: Optional[Float] = None,
) -> Complex:
    """
    Compute the complex inner product of two waveforms h1 and h2 
    with the given power spectral density (PSD).
    The first waveform, h1, is complex conjugated.

    Args:
        h1 (Float[Array, "n_sample"]): First waveform. Can be complex.
        h2 (Float[Array, "n_sample"]): Second waveform. Can be complex.
        psd (Float[Array, "n_sample"]): Power spectral density.
        frequency (Float[Array, "n_sample"]): Frequency array.
        df (Float): Frequency spacing. If None, it is calculated from the frequency array.

    Returns:
        Float: Noise-weighted inner product of h1 and h2 with given the PSD.
    """
    integrand = np.conj(h1) * h2 / psd
    return 4.0 * trapezoid(integrand, x=frequency, dx=df)


def inner_product(
    h1: Float[Array, " n_freq"],
    h2: Float[Array, " n_freq"],
    psd: Float[Array, " n_freq"],
    frequency: Optional[Float[Array, " n_freq"]] = None,
    df: Optional[Float] = None,
) -> Float:
    """
    Compute the noise-weighted inner product of two waveforms h1 and h2 
    with the given power spectral density (PSD).

    Args:
        h1 (Float[Array, "n_sample"]): First waveform. Can be complex.
        h2 (Float[Array, "n_sample"]): Second waveform. Can be complex.
        psd (Float[Array, "n_sample"]): Power spectral density.
        frequency (Float[Array, "n_sample"]): Frequency array.
        df (Float): Frequency spacing. If None, it is calculated from the frequency array.

    Returns:
        Float: Noise-weighted inner product of h1 and h2 with given the PSD.
    """
    return complex_inner_product(h1, h2, psd, frequency, df).real


def m1_m2_to_M_q(m1: Float, m2: Float) -> tuple[Float, Float]:
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
    M_tot = m1 + m2
    q = m2 / m1
    return M_tot, q


def M_q_to_m1_m2(M_tot: Float, q: Float) -> tuple[Float, Float]:
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
    m1 = M_tot / (1 + q)
    m2 = m1 * q
    return m1, m2


def m1_m2_to_Mc_q(m1: Float, m2: Float) -> tuple[Float, Float]:
    """
    Transforming the primary mass m1 and secondary mass m2 to the chirp mass M_c
    and mass ratio q.

    Parameters
    ----------
    m1 : Float
            Primary mass.
    m2 : Float
            Secondary mass.

    Returns
    -------
    M_c : Float
            Chirp mass.
    q : Float
            Mass ratio.
    """
    M_tot = m1 + m2
    eta = m1 * m2 / M_tot**2
    M_c = M_tot * eta ** (3.0 / 5)
    q = m2 / m1
    return M_c, q


def Mc_q_to_m1_m2(M_c: Float, q: Float) -> tuple[Float, Float]:
    """
    Transforming the chirp mass M_c and mass ratio q to the primary mass m1 and
    secondary mass m2.

    Parameters
    ----------
    M_c : Float
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
    M_tot = M_c / eta ** (3.0 / 5)
    m1 = M_tot / (1 + q)
    m2 = m1 * q
    return m1, m2


def m1_m2_to_M_eta(m1: Float, m2: Float) -> tuple[Float, Float]:
    """
    Transforming the primary mass m1 and secondary mass m2 to the total mass M
    and symmetric mass ratio eta.

    Parameters
    ----------
    m1 : Float
            Primary mass.
    m2 : Float
            Secondary mass.

    Returns
    -------
    M : Float
            Total mass.
    eta : Float
            Symmetric mass ratio.
    """
    M_tot = m1 + m2
    eta = m1 * m2 / M_tot**2
    return M_tot, eta


def M_eta_to_m1_m2(M_tot: Float, eta: Float) -> tuple[Float, Float]:
    """
    Transforming the total mass M and symmetric mass ratio eta to the primary mass m1
    and secondary mass m2.

    Parameters
    ----------
    M : Float
            Total mass.
    eta : Float
            Symmetric mass ratio.

    Returns
    -------
    m1 : Float
            Primary mass.
    m2 : Float
            Secondary mass.
    """
    m1 = M_tot * (1 + np.sqrt(1 - 4 * eta)) / 2
    m2 = M_tot * (1 - np.sqrt(1 - 4 * eta)) / 2
    return m1, m2


def m1_m2_to_Mc_eta(m1: Float, m2: Float) -> tuple[Float, Float]:
    """
    Transforming the primary mass m1 and secondary mass m2 to the chirp mass M_c
    and symmetric mass ratio eta.

    Parameters
    ----------
    m1 : Float
            Primary mass.
    m2 : Float
            Secondary mass.

    Returns
    -------
    M_c : Float
            Chirp mass.
    eta : Float
            Symmetric mass ratio.
    """
    M = m1 + m2
    eta = m1 * m2 / M**2
    M_c = M * eta ** (3.0 / 5)
    return M_c, eta


def Mc_eta_to_m1_m2(M_c: Float, eta: Float) -> tuple[Float, Float]:
    """
    Transforming the chirp mass M_c and symmetric mass ratio eta to the primary mass m1
    and secondary mass m2.

    Parameters
    ----------
    M_c : Float
            Chirp mass.
    eta : Float
            Symmetric mass ratio.

    Returns
    -------
    m1 : Float
            Primary mass.
    m2 : Float
            Secondary mass.
    """
    M = M_c / eta ** (3.0 / 5)
    m1 = M * (1 + np.sqrt(1 - 4 * eta)) / 2
    m2 = M * (1 - np.sqrt(1 - 4 * eta)) / 2
    return m1, m2


def q_to_eta(q: Float) -> Float:
    """
    Transforming the chirp mass M_c and mass ratio q to the symmetric mass ratio eta.

    Parameters
    ----------
    M_c : Float
            Chirp mass.
    q : Float
            Mass ratio.

    Returns
    -------
    eta : Float
            Symmetric mass ratio.
    """
    eta = q / (1 + q) ** 2
    return eta


def eta_to_q(eta: Float) -> Float:
    """
    Transforming the symmetric mass ratio eta to the mass ratio q.

    Copied and modified from bilby/gw/conversion.py

    Parameters
    ----------
    eta : Float
            Symmetric mass ratio.

    Returns
    -------
    q : Float
            Mass ratio.
    """
    temp = 1 / eta / 2 - 1
    return temp - (temp**2 - 1) ** 0.5


def euler_rotation(delta_x: Float[Array, " 3"]) -> Float[Array, " 3 3"]:
    """
    Calculate the rotation matrix mapping the vector (0, 0, 1) to delta_x
    while preserving the origin of the azimuthal angle.

    This is decomposed into three Euler angles, alpha, beta, gamma, which rotate
    about the z-, y-, and z- axes respectively.

    Copied and modified from bilby-cython/geometry.pyx
    """
    norm = np.linalg.vector_norm(delta_x)

    cos_beta = delta_x[2] / norm
    sin_beta = np.sqrt(1 - cos_beta**2)

    alpha = np.arctan2(-delta_x[1] * cos_beta, delta_x[0])
    gamma = np.arctan2(delta_x[1], delta_x[0])

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    rotation = np.array(
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


def angle_rotation(
    zenith: Float, azimuth: Float, rotation: Float[Array, " 3 3"]
) -> tuple[Float, Float]:
    """
    Transforming the azimuthal angle and zenith angle in Earth frame
    to the polar angle and azimuthal angle in sky frame.

    Modified from bilby-cython/geometry.pyx.

    Parameters
    ----------
    zenith : Float
            Zenith angle.
    azimuth : Float
            Azimuthal angle.
    rotation : Float[Array, " 3 3"]
            The rotation matrix.

    Returns
    -------
    theta : Float
            Polar angle.
    phi : Float
            Azimuthal angle.
    """
    sky_loc_vec = np.array(
        [
            np.sin(zenith) * np.cos(azimuth),
            np.sin(zenith) * np.sin(azimuth),
            np.cos(zenith),
        ]
    )
    rotated_vec = np.einsum("ij,j...->i...", rotation, sky_loc_vec)

    theta = np.acos(rotated_vec[2])
    phi = np.fmod(
        np.arctan2(rotated_vec[1], rotated_vec[0]) + 2 * np.pi,
        2 * np.pi,
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
    dec = np.pi / 2 - theta
    ra = ra % (2 * np.pi)
    return ra, dec


def zenith_azimuth_to_ra_dec(
    zenith: Float, azimuth: Float, gmst: Float, rotation: Float[Array, " 3 3"]
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
    rotation : Float[Array, " 3 3"]
            The rotation matrix.

    Copied and modified from bilby/gw/utils.py

    Returns
    -------
    ra : Float
            Right ascension.
    dec : Float
            Declination.
    """
    theta, phi = angle_rotation(zenith, azimuth, rotation)
    ra, dec = theta_phi_to_ra_dec(theta, phi, gmst)
    return ra, dec


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
    theta = np.pi / 2 - dec
    phi = (phi + 2 * np.pi) % (2 * np.pi)
    return theta, phi


def ra_dec_to_zenith_azimuth(
    ra: Float, dec: Float, gmst: Float, rotation: Float[Array, " 3 3"]
) -> tuple[Float, Float]:
    """
    Transforming the right ascension and declination to the zenith angle and azimuthal angle.

    Parameters
    ----------
    ra : Float
            Right ascension.
    dec : Float
            Declination.
    gmst : Float
            Greenwich mean sidereal time.
    rotation : Float[Array, " 3 3"]
            The rotation matrix.

    Returns
    -------
    zenith : Float
            Zenith angle.
    azimuth : Float
            Azimuthal angle.
    """
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    zenith, azimuth = angle_rotation(theta, phi, rotation)
    return zenith, azimuth


def rotate_y(angle: Float, vec: Float[Array, " 3"]) -> Float[Array, " 3"]:
    """
    Rotate the vector (x, y, z) about y-axis

    Parameters
    ----------
    angle : Float
        Angle in radians.
    vec : Float[Array, " 3"]
        Vector to be rotated.
    Returns
    -------
    rotated_vec : Float[Array, " 3"]
        Rotated vector.
    -------
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array(
        [[cos_angle, 0, sin_angle], [0, 1, 0], [-sin_angle, 0, cos_angle]]
    )
    rotated_vec = np.dot(rotation_matrix, vec)
    return rotated_vec


def rotate_z(angle: Float, vec: Float[Array, " 3"]) -> Float[Array, " 3"]:
    """
    Rotate the vector (x, y, z) about z-axis

    Parameters
    ----------
    angle : Float
        Angle in radians.
    vec : Float[Array, " 3"]
        Vector to be rotated.
    Returns
    -------
    rotated_vec : Float[Array, " 3"]
        Rotated vector.
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array(
        [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]]
    )
    rotated_vec = np.dot(rotation_matrix, vec)
    return rotated_vec


def Lmag_2PN(m1: Float, m2: Float, v0: Float) -> Float:
    """
    Compute the magnitude of the orbital angular momentum
    to 2 post-Newtonian orders.

    Parameters
    ----------
    m1 : Float
        Primary mass.
    m2 : Float
        Secondary mass.
    v0 : Float
        Relative velocity at the reference frequency.
    Returns
    -------
    Lmag : Float
        Magnitude of the orbital angular momentum.
    """
    eta = m1 * m2 / (m1 + m2) ** 2
    LN = (m1 + m2) * (m1 + m2) * eta / v0
    L_2PN = 1.5 + eta / 6.0
    return LN * (1.0 + v0 * v0 * L_2PN)


def spin_angles_to_cartesian_spin(
    theta_jn: Float,
    phi_jl: Float,
    tilt_1: Float,
    tilt_2: Float,
    phi_12: Float,
    chi_1: Float,
    chi_2: Float,
    M_c: Float,
    q: Float,
    fRef: Float,
    phiRef: Float,
) -> tuple[Float, Float, Float, Float, Float, Float, Float]:
    """
    Transforming the spin parameters

    The code is based on the approach used in LALsimulation:
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html

    Parameters:
    -------
    theta_jn: Float
        Zenith angle between the total angular momentum and the line of sight
    phi_jl: Float
        Difference between total and orbital angular momentum azimuthal angles
    tilt_1: Float
        Zenith angle between the spin and orbital angular momenta for the primary object
    tilt_2: Float
        Zenith angle between the spin and orbital angular momenta for the secondary object
    phi_12: Float
        Difference between the azimuthal angles of the individual spin vector projections
        onto the orbital plane
    chi_1: Float
        Primary object aligned spin:
    chi_2: Float
        Secondary object aligned spin:
    M_c: Float
        The chirp mass
    q: Float
        The mass ratio
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

    # Starting frame: LNh along the z-axis
    # S1hat on the x-z plane
    LNh = np.array([0.0, 0.0, 1.0])

    # Define the spin vectors in the LNh frame
    s1hat = np.array(
        [
            np.sin(tilt_1) * np.cos(phiRef),
            np.sin(tilt_1) * np.sin(phiRef),
            np.cos(tilt_1),
        ]
    )
    s2hat = np.array(
        [
            np.sin(tilt_2) * np.cos(phi_12 + phiRef),
            np.sin(tilt_2) * np.sin(phi_12 + phiRef),
            np.cos(tilt_2),
        ]
    )

    m1, m2 = Mc_q_to_m1_m2(M_c, q)
    v0 = np.cbrt((m1 + m2) * MTSUN * np.pi * fRef)

    # Define S1, S2, and J
    Lmag = Lmag_2PN(m1, m2, v0)
    s1 = m1 * m1 * chi_1 * s1hat
    s2 = m2 * m2 * chi_2 * s2hat
    J = s1 + s2 + np.array([0.0, 0.0, Lmag])

    # Normalize J, and find theta0 and phi0 (the angles in starting frame)
    Jhat = J / np.linalg.norm(J)
    theta0 = np.arccos(Jhat[2])
    phi0 = np.arctan2(Jhat[1], Jhat[0])

    # Rotation 1: Rotate about z-axis by -phi0
    s1hat = rotate_z(-phi0, s1hat)
    s2hat = rotate_z(-phi0, s2hat)

    # Rotation 2: Rotate about y-axis by -theta0
    LNh = rotate_y(-theta0, LNh)
    s1hat = rotate_y(-theta0, s1hat)
    s2hat = rotate_y(-theta0, s2hat)

    # Rotation 3: Rotate about z-axis by -phi_jl
    LNh = rotate_z(phi_jl - np.pi, LNh)
    s1hat = rotate_z(phi_jl - np.pi, s1hat)
    s2hat = rotate_z(phi_jl - np.pi, s2hat)

    # Compute iota
    N = np.array([0.0, np.sin(theta_jn), np.cos(theta_jn)])
    iota = np.arccos(np.dot(N, LNh))

    thetaLJ = np.arccos(LNh[2])
    phiL = np.arctan2(LNh[1], LNh[0])

    # Rotation 4: Rotate about z-axis by -phiL
    s1hat = rotate_z(-phiL, s1hat)
    s2hat = rotate_z(-phiL, s2hat)
    N = rotate_z(-phiL, N)

    # Rotation 5: Rotate about y-axis by -thetaLJ
    s1hat = rotate_y(-thetaLJ, s1hat)
    s2hat = rotate_y(-thetaLJ, s2hat)
    N = rotate_y(-thetaLJ, N)

    # Rotation 6:
    phiN = np.arctan2(N[1], N[0])
    s1hat = rotate_z(np.pi / 2.0 - phiN - phiRef, s1hat)
    s2hat = rotate_z(np.pi / 2.0 - phiN - phiRef, s2hat)

    S1 = s1hat * chi_1
    S2 = s2hat * chi_2
    return iota, S1[0], S1[1], S1[2], S2[0], S2[1], S2[2]


def cartesian_spin_to_spin_angles(
    iota: Float,
    S1x: Float,
    S1y: Float,
    S1z: Float,
    S2x: Float,
    S2y: Float,
    S2z: Float,
    M_c: Float,
    q: Float,
    fRef: Float,
    phiRef: Float,
) -> tuple[Float, Float, Float, Float, Float, Float, Float]:
    """
    Transforming the cartesian spin parameters to the spin angles

    The code is based on the approach used in LALsimulation:
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html

    Parameters:
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
    M_c: Float
        The chirp mass
    q: Float
        The mass ratio
    fRef: Float
        The reference frequency
    phiRef: Float
        The binary phase at the reference frequency

    Returns:
    -------
    theta_jn: Float
    Zenith angle between the total angular momentum and the line of sight
    phi_jl: Float
        Difference between total and orbital angular momentum azimuthal angles
    tilt_1: Float
        Zenith angle between the spin and orbital angular momenta for the primary object
    tilt_2: Float
        Zenith angle between the spin and orbital angular momenta for the secondary object
    phi_12: Float
        Difference between the azimuthal angles of the individual spin vector projections
        onto the orbital plane
    chi_1: Float
        Primary object aligned spin:
    chi_2: Float
        Secondary object aligned spin:
    """
    # Starting frame: LNh along the z-axis
    LNh = np.array([0.0, 0.0, 1.0])

    # Define the dimensionless component spin vectors and magnitudes
    s1_vec = np.array([S1x, S1y, S1z])
    s2_vec = np.array([S2x, S2y, S2z])
    chi_1 = np.linalg.norm(s1_vec)
    chi_2 = np.linalg.norm(s2_vec)

    # Define the spin unit vectors in the LNh frame
    s1hat = np.where(chi_1 > 0, s1_vec / chi_1, np.zeros_like(s1_vec))
    s2hat = np.where(chi_2 > 0, s2_vec / chi_2, np.zeros_like(s2_vec))

    # Azimuthal and polar angles of the spin vectors
    phi1 = np.arctan2(s1hat[1], s1hat[0])
    phi2 = np.arctan2(s2hat[1], s2hat[0])

    phi_12 = phi2 - phi1

    phi_12 = (phi_12 + 2 * np.pi) % (2 * np.pi)  # Ensure 0 <= phi_12 < 2pi

    tilt_1 = np.arccos(s1hat[2])
    tilt_2 = np.arccos(s2hat[2])

    # Get angles in the J-N frame
    m1, m2 = Mc_q_to_m1_m2(M_c, q)
    v0 = np.cbrt((m1 + m2) * MTSUN * np.pi * fRef)

    # Define S1, S2, J
    S1 = m1 * m1 * s1_vec
    S2 = m2 * m2 * s2_vec

    Lmag = Lmag_2PN(m1, m2, v0)
    J = S1 + S2 + Lmag * LNh

    # Normalize J
    Jhat = J / np.linalg.norm(J)

    thetaJL = np.arccos(Jhat[2])
    phiJ = np.arctan2(Jhat[1], Jhat[0])

    # Azimuthal angle from phase angle
    phi0 = 0.5 * np.pi - phiRef
    # Line-of-sight vector in L-frame
    N = np.array(
        [np.sin(iota) * np.cos(phi0), np.sin(iota) * np.sin(phi0), np.cos(iota)]
    )

    # Inclination w.r.t. J
    theta_jn = np.arccos(np.dot(Jhat, N))

    # Rotate from L-frame to J-frame
    N = rotate_z(-phiJ, N)
    N = rotate_y(-thetaJL, N)

    LNh = rotate_z(-phiJ, LNh)
    LNh = rotate_y(-thetaJL, LNh)

    phiN = np.arctan2(N[1], N[0])
    LNh = rotate_z(0.5 * np.pi - phiN, LNh)

    phi_jl = np.arctan2(LNh[1], LNh[0])
    phi_jl = (phi_jl + 2 * np.pi) % (2 * np.pi)  # Ensure 0 <= phi_jl < 2pi

    return theta_jn, phi_jl, tilt_1, tilt_2, phi_12, chi_1, chi_2
