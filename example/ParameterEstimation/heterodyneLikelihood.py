from cmath import phase
import numpy as np
import jax.numpy as jnp
import jax

from ripple.waveforms import IMRPhenomD, IMRPhenomD_utils
import matplotlib.pyplot as plt
from ripple import ms_to_Mc_eta

from scipy.interpolate import interp1d

# Get a frequency domain waveform
# source parameters

m1_msun = 20.0 # In solar masses
m2_msun = 19.0
chi1 = 0.5 # Dimensionless spin
chi2 = -0.5
tc = 0.0 # Time of coalescence in seconds
phic = 0.0 # Time of coalescence
dist_mpc = 440 # Distance to source in Mpc
inclination = 0.0 # Inclination Angle
polarization_angle = 0.2 # Polarization angle

# The PhenomD waveform model is parameterized with the chirp mass and symmetric mass ratio
Mc, eta = ms_to_Mc_eta(jnp.array([m1_msun, m2_msun]))

# These are the parametrs that go into the waveform generator
# Note that JAX does not give index errors, so if you pass in the
# the wrong array it will behave strangely
theta_ripple = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle])
theta_ripple_vec = np.array(jnp.repeat(theta_ripple[None,:],100000,axis=0)*np.random.normal(loc=1,scale=0.001,size=(100000,9)))
theta_ripple_vec[theta_ripple_vec[:,1]>0.25,1] = 0.25


# Now we need to generate the frequency grid
f_l = 24
f_u = 1024
del_f = 10
fs = jnp.arange(f_l, f_u, del_f)

# And finally lets generate the waveform!
hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(fs, theta_ripple)

@jax.jit
def waveform_gen(theta):
    return IMRPhenomD.gen_IMRPhenomD_polar(fs, theta)

waveform_gen_vec = jax.vmap(waveform_gen)

# Choosing binning scheme

def max_phase_diff(f, f_low, f_high, chi=1):
    gamma = np.arange(-5,6,1)/3.
    f = np.repeat(f[:,None],len(gamma),axis=1)
    f_star = np.repeat(f_low, len(gamma))
    f_star[gamma >= 0] = f_high
    return 2*np.pi*chi*np.sum((f/f_star)**gamma*np.sign(gamma),axis=1)

f_fine = np.linspace(f_l, f_u, 10000)
phase_diff_array = max_phase_diff(f_fine,f_l,f_u,chi=1)
bin_f = interp1d(phase_diff_array, f_fine)
n_bin = 1001
f_bins = np.array([])
for i in np.linspace(phase_diff_array[0], phase_diff_array[-1], n_bin):
    f_bins = np.append(f_bins,bin_f(i))
f_bins_center = (f_bins[:-1] + f_bins[1:])/2

# Compute coefficients from reference waveform

# IMRPhenomD_jit = jax.vmap(jax.jit(IMRPhenomD.gen_IMRPhenomD_polar),(0,None),0)

data = IMRPhenomD.gen_IMRPhenomD_polar(f_fine, theta_ripple)[0]
bin_coef = []
theta_ref = jnp.array([Mc, 0.23, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle])
h_ref = IMRPhenomD.gen_IMRPhenomD_polar(f_fine, theta_ref)[0]
h_ref_bin_center = IMRPhenomD.gen_IMRPhenomD_polar(f_bins_center, theta_ref)[0]
h_ref_bin_low = IMRPhenomD.gen_IMRPhenomD_polar(f_bins[:-1], theta_ref)[0]
A0_array = []
A1_array = []

data_prod = np.array(data*h_ref.conj())
for i in range(len(f_bins)-1):
    print(i)
    f_index = np.where((f_fine >= f_bins[i]) & (f_fine < f_bins[i+1]))[0]
    A0_array.append(np.sum(data_prod[f_index]))
    A1_array.append(np.sum(data_prod[f_index]*(f_fine[f_index]-f_bins_center[i])))

A0_array = jnp.array(A0_array)
A1_array = jnp.array(A1_array)

# run time evaluation of inner product

theta_test = jnp.array([Mc, 0.22, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle])
h_test_fine = IMRPhenomD.gen_IMRPhenomD_polar(f_fine, theta_test)[0]
h_test_bin_center = IMRPhenomD.gen_IMRPhenomD_polar(f_bins_center, theta_test)[0]
h_test_bin_low = IMRPhenomD.gen_IMRPhenomD_polar(f_bins[:-1], theta_test)[0]
true_SNR = jnp.sum(data*h_test_fine.conj())

r0 = h_test_bin_center/h_ref_bin_center
r1 = (h_test_bin_low/h_ref_bin_low - r0)/(f_bins[:-1]-f_bins_center)

bin_SNR = np.sum(A0_array*r0.conj() + A1_array*r1.conj())

print(bin_SNR, true_SNR)


