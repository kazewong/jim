from cmath import phase
from webbrowser import get
import numpy as np
import jax.numpy as jnp

from ripple.waveforms import IMRPhenomD, IMRPhenomD_utils
import matplotlib.pyplot as plt
from ripple import ms_to_Mc_eta




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


# Now we need to generate the frequency grid
f_l = 24
f_u = 1024
del_f = 10
fs = jnp.arange(f_l, f_u, del_f)

# And finally lets generate the waveform!
hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(fs, theta_ripple)


from jaxgw.PE.detector_preset import *
from jaxgw.PE.detector_projection_new import make_detector_response

H1 = get_H1()
L1 = get_L1()
H1_response = make_detector_response(H1[0], H1[1])
L1_response = make_detector_response(L1[0], L1[1])
H1_response(fs, hp_ripple, hc_ripple, 0.2, 0.3, 0.,0.5)

