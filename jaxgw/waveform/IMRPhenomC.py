"""
Implementation of IMRPhenomC waveform following 1005.3306
"""
import jax.numpy as jnp
from jax import jit
from jaxgw.utils import *

euler_gamma = 0.577215664901532860606512090082
MR_sun = 1.476625061404649406193430731479084713e3

def Lorentzian(x, x0, gamma):
    return (gamma**2/((x-x0)**2+gamma**2/4))

def smoothing_plus(f,f0,d):
    return (1+jnp.tanh(4*(f-f0)/d))/2

def smoothing_minus(f,f0,d):
    return (1-jnp.tanh(4*(f-f0)/d))/2

@jit
def PNAmplitudeAndPhasing(f,m1,m2,chi1,chi2,t0,phase0):

    x = (jnp.pi*f)**(2./3) # I assume in the m in 3.12 is the m from harmonics instead of mass

    eta = m1*m2/(m1+m2)**2
    chi_eff = (m1*chi1+m2*chi2)/(m1+m2)

# Taylor T4 expansion coefficient from A3 for 3.6, needed for amplitude in fourier space

    T4_alpha_power_index = x[:,None]**(jnp.array([0,2,3,4,5,6,7])/2)

    T4_alpha_coeff = jnp.array([1, \

               (-7.43/3.36 - 11.*eta/4.),\

               (4.*jnp.pi - 11.3*chi_eff/1.2 + 19.*eta*(chi1+chi2)/6.),\

               (3.4103/1.8144 + 5*chi_eff**2 + eta*(13.661/2.016 - chi1*chi2/8.) + 5.9*eta**2/1.8),\

               (-jnp.pi*(41.59/6.72 + 189.*eta/8.) - chi_eff*(31.571/1.008 - 116.5*eta/2.4) +\
               (chi1+chi2)*(21.863*eta/1.008 - 79.*eta**2/6.) - 3*chi_eff**3/4. +\
               9.*eta*chi_eff*chi1*chi2/4.),\

               (164.47322263/1.39708800 - 17.12*euler_gamma/1.05 +\
               16.*jnp.pi**2/3 +\
               eta*(45.1*jnp.pi**2/4.8 - 561.98689/2.17728) +\
               5.41*eta**2/8.96 - 5.605*eta**3/2.592 - 80.*jnp.pi*chi_eff/3. +\
               eta*(chi1+chi2)*(20.*jnp.pi/3. - 113.5*chi_eff/3.6) +\
               chi_eff**2*(64.153/1.008 - 45.7*eta/3.6) -\
               chi1*chi2*(7.87*eta/1.44 - 30.37*eta**2/1.44)),\


               (-jnp.pi*(4.415/4.032 - 358.675*eta/6.048 - 91.495*eta**2/1.512) -\
               chi_eff*(252.9407/2.7216 - 845.827*eta/6.048 + 415.51*eta**2/8.64) +\
               (chi1+chi2)*(158.0239*eta/5.4432 - 451.597*eta**2/6.048 + 20.45*eta**3/4.32 +\
               107.*eta*chi_eff**2/6. - 5.*eta**2*chi1*chi2/24.) +\
               12.*jnp.pi*chi_eff**2 - chi_eff**3*(150.5/2.4 + eta/8.) +\
               chi_eff*chi1*chi2*(10.1*eta/2.4 + 3.*eta**2/8.))])

    T4_alpha_log =  - 8.56*jnp.log(16.*x)/1.05*x**3 

    T4_alpha = 64.*eta/5. *x**5* (jnp.sum(T4_alpha_power_index*T4_alpha_coeff,axis=1)+T4_alpha_log)

# Taylor T4 amplitude coefficient from A5

    T4_A_power_index = x[:,None]**(jnp.array([0,2,3,4,5,6])/2)

    T4_A_coeff = jnp.array([1,\

           ((-107. + 55.*eta)/42.),\

           (2.*jnp.pi - 4.*chi_eff/3. + 2.*eta*(chi1+chi2)/3.),\

           (-2.173/1.512 - eta*(10.69/2.16 - 2.*chi1*chi2) + 2.047*eta**2/1.512),\

           (-10.7*jnp.pi/2.1 + eta*(3.4*jnp.pi/2.1-24.*1j)),\

           (270.27409/6.46800 - 8.56*euler_gamma/1.05 +\
           2.*jnp.pi**2/3. +\
           eta*(4.1*jnp.pi**2/9.6 - 27.8185/3.3264) -\
           20.261*eta**2/2.772 + 11.4635*eta**3/9.9792 +\
           4.28*(1j*jnp.pi)/1.05)])

    T4_A_log = (4.28*(-jnp.log(16.*x))/1.05)*x**3

    T4_A =  8.*eta*jnp.sqrt(jnp.pi/5.)*x*(jnp.sum(T4_A_power_index*T4_A_coeff,axis=1)+T4_A_log)

# Taylor F2 Phasing coefficient from A4

    F2_alpha_power_index = jnp.pi*f[:,None]**(jnp.array([0,2,3,4,5,6,7])/3.)

    F2_alpha_coeff = jnp.array([1,\

            ((3715./756.) + (55.*eta/9.0)),\

            (-16.0*jnp.pi + (113./3.)*chi_eff - 38.*eta*(chi1+chi2)/3.),\

            ((152.93365/5.08032) - 50.*chi_eff**2 + eta*(271.45/5.04 + 1.25*chi1*chi2) + \
             3085.*eta**2/72.),\

            ((jnp.pi*(386.45/7.56 - 65.*eta/9.) - \
             chi_eff*(735.505/2.268 + 130.*eta/9.) + (chi1+chi2)*(1285.0*eta/8.1 + 170.*eta**2/9.) -\
             10.*chi_eff**3/3. + 10.*eta*chi_eff*(chi1*chi2))), \

            (11583.231236531/4.694215680 - 640.0*jnp.pi**2/3. - \
             6848.0*euler_gamma/21. + \
             eta*(2255.*jnp.pi**2/12. - 15737.765635/3.048192) + \
             76.055*eta**2/1.728 - (127.825*eta**3/1.296) + \
             2920.*jnp.pi*chi_eff/3. - (175. - 1490.*eta)*chi_eff**2/3. - \
             (1120.*jnp.pi/3. - 1085.*chi_eff/3.)*eta*(chi1+chi2) + \
             (269.45*eta/3.36 - 2365.*eta**2/6.)*chi1*chi2), \

            (jnp.pi*(770.96675/2.54016 + 378.515*eta/1.512 - 740.45*eta**2/7.56) - \
             chi_eff*(20373.952415/3.048192 + 1509.35*eta/2.24 - 5786.95*eta**2/4.32) + \
             (chi1+chi2)*(4862.041225*eta/1.524096 + 1189.775*eta**2/1.008 - \
             717.05*eta**3/2.16 - 830.*eta*chi_eff**2/3. + 35.*eta**2*chi1*chi2/3.) - \
             560.*jnp.pi*chi_eff**2 + 20.*jnp.pi*eta*chi1*chi2 + \
             chi_eff**3*(945.55/1.68 - 85.*eta) + chi_eff*chi1*chi2*(396.65*eta/1.68 + 255.*eta**2))])

    F2_alpha_log = F2_alpha_coeff[4]*jnp.log(jnp.pi*f)*(jnp.pi*f)**(5./3) + (- 684.8*jnp.log(64.*jnp.pi*f)/6.3)*(jnp.pi*f)**2

    F2_alpha = 3.0/(128.0 * eta)*(jnp.pi*f)**(-5./3)*(jnp.sum(F2_alpha_power_index*F2_alpha_coeff,axis=1)+F2_alpha_log)

    amplitude = T4_A*jnp.sqrt(jnp.pi/(3./2*jnp.sqrt(x)*T4_alpha))
    phase = 2*jnp.pi*f*t0 - phase0 - jnp.pi/4 + F2_alpha

    return amplitude, phase
#    return T4_alpha,T4_A,F2_alpha#amplitude, phase


alpha_coef = jnp.array([[-2.417 * 1e-3, -1.093 * 1e-3, -1.917 * 1e-2, 7.267 * 1e-2, -2.504 * 1e-1],\
                        [5.962 * 1e-1, -5.6 * 1e-2, 1.52 * 1e-1, -2.97, 1.312 * 1e1],\
                        [-3.283 * 1e1, 8.859, 2.931 * 1e1, 7.954 * 1e1, -4.349 * 1e2],\
                        [1.619 * 1e2, -4.702 * 1e1, -1.751 * 1e2, -3.225 * 1e2, 1.587 * 1e3],\
                        [-6.32 * 1e2, 2.463 * 1e2, 1.048 * 1e3, 3.355 * 1e2, -5.115 * 1e3],\
                        [-4.809 * 1e1, -3.643 * 1e2, -5.215 * 1e2, 1.87 * 1e3, 7.354 * 1e2]])

gamma_coef = jnp.array([4.149, -4.07, -8.752 * 1e1, -4.897 * 1e1, 6.665 * 1e2])

delta_coef = jnp.array([[-5.472 * 1e-2, 2.094 * 1e-2, 3.554 * 1e-1, 1.151 * 1e-1, 9.64 * 1e-1], \
                    [-1.235, 3.423*1e-1, 6.062, 5.949, -1.069*1e1]])

phase_PM_order = jnp.array([-5./3,-3./3,-1./3,0,2./3,1])


@jit
def getPhenomCoef(eta, chi):
    eta_chi = jnp.array([chi,chi**2,eta*chi,eta,eta**2])
    alpha = jnp.sum(alpha_coef*eta_chi,axis=1)
    gamma = jnp.sum(gamma_coef*eta_chi)
    delta = jnp.sum(delta_coef*eta_chi,axis=1)
    return alpha, gamma, delta 


@jit
def getFinalSpin(eta,chi_eff):
    finspin = chi_eff - 0.129*chi_eff**2*eta -0.384*eta**2*chi_eff -2.686*eta*chi_eff \
             + 2.*jnp.sqrt(3.)*eta  -3.454*eta**2 + 2.353*eta**3
    return finspin

@jit
def IMRPhenomC(f,params):
    """
    The amplitude and phase are generated first in unitless Mf space, then scaled with total mass and distance.
    """

    local_m1 = params['mass_1']*Msun
    local_m2 = params['mass_2']*Msun
    local_d = params['luminosity_distance']*Mpc
    local_spin1 = params['a_1']
    local_spin2 = params['a_2']


    M_tot = local_m1+local_m2
    eta = local_m1*local_m2/(local_m1+local_m2)**2
    M_chirp = M_tot*eta**(3./5)
    chi_eff = (local_spin1*local_m1 + local_spin2*local_m2)/M_tot
    f = f*M_tot

    final_spin = getFinalSpin(eta,chi_eff)
    f_rd = 1./(2*jnp.pi*(Msun))*(1.5251 - 1.1568*(1-final_spin)**0.1292)*Msun
    decay_time = 0.7 + 1.4187*(1-final_spin)**(-0.499) # Q in the paper
    
# Constructing phase of the waveform

    A_PN, phase_PN = PNAmplitudeAndPhasing(f,local_m1,local_m2,local_spin1,local_spin2,params['geocent_time'],params['phase'])
    alpha, gamma, delta = getPhenomCoef(eta,chi_eff)


    phase_PM = 1./eta*jnp.sum(alpha*f[:,None]**phase_PM_order,axis=1)

    beta_1 = 1./eta*jnp.sum(alpha*f_rd**phase_PM_order)
    beta_2 = 1./eta*jnp.sum(jnp.array([-5./3,-3./3,-1./3,2./3,1])*alpha[jnp.array([0,1,2,4,5])]*f_rd**jnp.array([-8./3,-6./3,-4./3,-1./3,0]))
    phase_RD = beta_1 + beta_2*f

    phase = phase_PN*smoothing_minus(f,0.1*f_rd,0.005) + phase_PM*smoothing_plus(f,0.1*f_rd,0.005)*smoothing_minus(f,f_rd,0.005) + phase_RD*smoothing_plus(f,f_rd,0.005)
    #phase = phase_PM*smoothing_plus(f,0.1*f_rd,0.005)*smoothing_minus(f,f_rd,0.005) + phase_RD*smoothing_plus(f,f_rd,0.005)

# Constructing amplitude of the waveform

    A_PM = A_PN + gamma*(f)**(5./6)

    A_RD = delta[0]*Lorentzian(f,f_rd,delta[1]*f_rd/decay_time)*f**(-7./6)

    amplitude = A_PM *smoothing_minus(f,0.98*f_rd,0.015) + A_RD*smoothing_plus(f,0.98*f_rd,0.015)

    amplitude_factor = 2. * jnp.sqrt(5. / (64.*jnp.pi)) * M_tot**2 / local_d
    totalh = (amplitude*amplitude_factor*jnp.exp(1j*phase))#* amplitude_factor
    hp = totalh * (1/2*(1+jnp.cos(params['theta_jn'])**2)*jnp.cos(2*params['psi']))
    hc = totalh * jnp.cos(params['theta_jn'])*jnp.sin(2*params['psi'])

    return {'plus':hp,'cross':hc}
