from lal import MSUN_SI, PC_SI, MTSUN_SI
import lalsimulation as lalsim
import numpy as np

from jaxgw.gw.waveform.TaylorF2 import TaylorF2

mass_1 = 30.
mass_2 = 30.
luminosity_distance = 410.
f0 = 20.
max_f = 2048
delta_f = 1./8
spin = 0.

injection_parameters = dict(
	mass_1=mass_1, mass_2=mass_2, spin_1=0.0, spin_2=0.0, luminosity_distance=luminosity_distance, 	phase_c=0, t_c=0, theta_jn=0.4, psi=2.659,)

 
waveform1 = lalsim.SimInspiralTaylorF2(0., delta_f, mass_1*MSUN_SI, mass_2* MSUN_SI, 0., 0., f0, max_f, 0.,luminosity_distance* 1e6*PC_SI,{})
frequency = waveform1.f0 + np.arange(len(waveform1.data.data)) * waveform1.deltaF
waveform3 = TaylorF2(frequency,injection_parameters)