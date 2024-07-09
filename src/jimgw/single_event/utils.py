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

def rotate_y(angle: Float, x: Float, y: Float, z: Float) -> tuple[Float, Float, Float]:
    """
    Rotate the vector (x, y, z) about y-axis
    """
    x_new = x * jnp.cos(angle) + z * jnp.sin(angle)
    z_new = - (x * jnp.sin(angle)) + z * jnp.cos(angle)
    return x_new, y, z_new


def rotate_z(angle: Float, x: Float, y: Float, z: Float) -> tuple[Float, Float, Float]:
    """
    Rotate the vector (x, y, z) about z-axis
    """
    x_new = x * jnp.cos(angle) - y * jnp.sin(angle)
    y_new = x * jnp.sin(angle) + y * jnp.cos(angle)
    return x_new, y_new, z
    

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

    LNhx = 0.
    LNhy = 0.
    LNhz = 1.

    s1hatx = jnp.sin(theta1)*jnp.cos(phiRef)
    s1haty = jnp.sin(theta1)*jnp.sin(phiRef)
    s1hatz = jnp.cos(theta1)
    s2hatx = jnp.sin(theta2) * jnp.cos(phi12+phiRef)
    s2haty = jnp.sin(theta2) * jnp.sin(phi12+phiRef)
    s2hatz = jnp.cos(theta2)
  
    temp = (1 / eta / 2 - 1)
    q = temp - (temp ** 2 - 1) ** 0.5
    m1, m2 = Mc_q_to_m1m2(M_c, q)
    MTsun_SI = 4.925490947641266978197229498498379006e-6
    v0 = jnp.cbrt((m1+m2) * MTsun_SI * jnp.pi * fRef)
  
    Lmag = ((m1+m2)*(m1+m2)*eta/v0) * (1.0 + v0*v0*(1.5 + eta/6.0))
    s1x = m1 * m1 * chi1 * s1hatx
    s1y = m1 * m1 * chi1 * s1haty
    s1z = m1 * m1 * chi1 * s1hatz
    s2x = m2 * m2 * chi2 * s2hatx
    s2y = m2 * m2 * chi2 * s2haty
    s2z = m2 * m2 * chi2 * s2hatz
    Jx = s1x + s2x
    Jy = s1y + s2y
    Jz = Lmag + s1z + s2z
  

    Jnorm = jnp.sqrt( Jx*Jx + Jy*Jy + Jz*Jz)
    Jhatx = Jx / Jnorm
    Jhaty = Jy / Jnorm
    Jhatz = Jz / Jnorm
    theta0 = jnp.arccos(Jhatz)
    phi0 = jnp.arctan2(Jhaty, Jhatx)

    s1hatx, s1haty, s1hatz = rotate_z(-phi0, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_z(-phi0, s2hatx, s2haty, s2hatz)
  
    LNhx, LNhy, LNhz = rotate_y(-theta0, LNhx, LNhy, LNhz)
    s1hatx, s1haty, s1hatz = rotate_y(-theta0, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_y(-theta0, s2hatx, s2haty, s2hatz)
    
    LNhx, LNhy, LNhz = rotate_z(phiJL - jnp.pi, LNhx, LNhy, LNhz)
    s1hatx, s1haty, s1hatz = rotate_z(phiJL - jnp.pi, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_z(phiJL - jnp.pi, s2hatx, s2haty, s2hatz)

    Nx=0.0
    Ny=jnp.sin(thetaJN)
    Nz=jnp.cos(thetaJN)
    iota=jnp.arccos(Nx*LNhx+Ny*LNhy+Nz*LNhz)
  
    thetaLJ = jnp.arccos(LNhz)
    phiL = jnp.arctan2(LNhy, LNhx)
  
    s1hatx, s1haty, s1hatz = rotate_z(-phiL, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_z(-phiL, s2hatx, s2haty, s2hatz)
    Nx, Ny, Nz = rotate_z(-phiL, Nx, Ny, Nz)
    
    s1hatx, s1haty, s1hatz = rotate_y(-thetaLJ, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_y(-thetaLJ, s2hatx, s2haty, s2hatz)
    Nx, Ny, Nz = rotate_y(-thetaLJ, Nx, Ny, Nz)

    phiN = jnp.arctan2(Ny, Nx)
    s1hatx, s1haty, s1hatz = rotate_z(jnp.pi/2.-phiN-phiRef, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_z(jnp.pi/2.-phiN-phiRef, s2hatx, s2haty, s2hatz)

    S1x = s1hatx*chi1
    S1y = s1haty*chi1
    S1z = s1hatz*chi1
    S2x = s2hatx*chi2
    S2y = s2haty*chi2
    S2z = s2hatz*chi2
    
    return iota, S1x, S1y, S1z, S2x, S2y, S2z

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
