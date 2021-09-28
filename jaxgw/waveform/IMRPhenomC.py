"""
Implementation of IMRPhenomC waveform following 1005.3306
"""
import jax.numpy as jnp
from jax import jit
from jaxgw.utils import *

@jit
def Lorentzian(x, x0, gamma):
    return (gamma**2/((x-x0)**2+gamma**2/4))

@jit
def getPhenomCoef(M, eta, chi):
    psi_coef = jnp.array([[3715/756, -920.9, 492.1, 135, 6742, -1053, -1.34*1e4], \
               [-16*jnp.pi + 113*chi/3, 1.702*1e4, -9566, -2182, -1.214*1e5, 2.075*1e4, 2.386*1e5], \
               [15293365/508032 - 405*chi**2/8, -1.254*1e5, 7.507*1e4, 1.338*1e4, 8.735*1e5, -1.657*1e5, -1.694*1e6], \
               [0, -8.898*1e5, 6.31*1e5, 5.068*1e4, 5.981*1e6, -1.415*1e6, -1.128*1e7], \
               [0, 8.696*1e5, -6.71*1e5, -3.008*1e4, -5.838*1e6, 1.514*1e6, 1.089*1e7]])
    
    mu_coef = jnp.array([[1-4.455*(1-chi)**0.217+3.521*(1-chi)**0.26, 0.6437, 0.827, -0.2706, -0.05822, -3.935, -7.092], \
              [(1-0.63*(1-chi)**0.3)/2, 0.1469, -0.1228, -0.02609, -0.0249, 0.1701, 2.325], \
              [(1-0.63*(1-chi)**0.3)*((1-chi)**0.45)/4, -0.4098, -0.03523, 0.1008, 1.829, -0.02017, -2.87], \
              [0.3236 + 0.04894*chi + 0.01346*chi**2, -0.1331, -0.08172, 0.1451, -0.2714, 0.1279, 4.922]])
    psi = psi_coef[:,0] + eta * (psi_coef[:,1] + psi_coef[:,2]*chi + psi_coef[:,3]*chi**2)\
                        + eta**2 * (psi_coef[:,4] + psi_coef[:,5]*chi)\
                        + eta**3 * psi_coef[:,6]
    f1, f2, sigma, f3 = (mu_coef[:,0] + eta * (mu_coef[:,1] + mu_coef[:,2]*chi + mu_coef[:,3]*chi**2)\
                        + eta**2 * (mu_coef[:,4] + mu_coef[:,5]*chi)\
                        + eta**3 * mu_coef[:,6]) / (jnp.pi * M)

    return psi, f1, f2, sigma, f3

@jit
def IMRPhenomB(f,params):


    f = f[:,None]

    local_m1 = params['mass_1']*Msun
    local_m2 = params['mass_2']*Msun
    local_d = params['luminosity_distance']*Mpc
    local_spin1 = params['a_1']
    local_spin2 = params['a_2']

    M_tot = local_m1+local_m2
    eta = local_m1*local_m2/(local_m1+local_m2)**2
    chi_eff = (local_spin1*local_m1 + local_spin2*local_m2)/M_tot
    M_chirp = eta**(3./5)*M_tot
    PNcoef = (jnp.pi*M_tot*f)**(1./3)

    epsilon1 = 1.4547*chi_eff - 1.8897
    epsilon2 = -1.8153*chi_eff + 1.6557
    alpha2 = -323./224 + 451.*eta/168
    alpha3 = (27./8 - 11.*eta/6)*chi_eff

    psi, f1, f2, sigma, f3 = getPhenomCoef(M_tot, eta, chi_eff)

    Afactor_inspiral = (1 + alpha2*PNcoef**2+ alpha3*PNcoef**3)
    Afactor_merger = (1 + epsilon1*PNcoef+ epsilon2*PNcoef**2)
    omega_merger = Afactor_inspiral/Afactor_merger
    omega_ringdown = Afactor_merger/Lorentzian(f2,f2,sigma)


    phase = 2*jnp.pi*f*params['geocent_time'] - params['phase']
    phase += 3./(128*eta*PNcoef**5) * (1+ jnp.sum(psi*PNcoef**jnp.array([2,3,4,6,7]),axis=1)[:,None])

    A_overall = M_chirp**(5./6)/local_d*f1**(-7./6)
    A_inspiral = (f/f1)**(-7./6) * Afactor_inspiral 
    A_merger =  omega_merger * (f/f1)**(-2./3) * Afactor_merger
    A_ringdown = omega_ringdown * Lorentzian(f, f2, sigma)

    amplitude = A_overall * (A_inspiral * jnp.heaviside(f1-f,0) \
                          +  A_merger * jnp.heaviside(f-f1,1) * jnp.heaviside(f2-f,0) \
                          +  A_ringdown * jnp.heaviside(f-f2,1))# * jnp.heaviside(f3-f,0))



    totalh = amplitude*jnp.exp(-1j*phase)
    hp = totalh * (1/2*(1+jnp.cos(params['theta_jn'])**2)*jnp.cos(2*params['psi']))
    hc = totalh * jnp.cos(params['theta_jn'])*jnp.sin(2*params['psi'])

    return {'plus':hp,'cross':hc}
