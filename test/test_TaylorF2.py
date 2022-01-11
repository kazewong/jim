from lal import MSUN_SI, PC_SI, MTSUN_SI
import lalsimulation as lalsim
import numpy as np

from jaxgw.gw.waveform.TaylorF2 import TaylorF2
from bilby.gw.utils import greenwich_mean_sidereal_time

mass_1 = 30.
mass_2 = 30.
luminosity_distance = 410.
f0 = 20.
max_f = 2048
delta_f = 1./8
spin = 0.02

injection_parameters = dict(
	mass_1=mass_1, mass_2=mass_2, spin_1=spin, spin_2=spin, luminosity_distance=luminosity_distance, phase_c=0, t_c=0,\
	theta_jn=0.4, psi=2.659,f_ref = 50)

 
waveform1 = lalsim.SimInspiralTaylorF2(0., delta_f, mass_1*MSUN_SI, mass_2* MSUN_SI, 0., 0., f0, max_f, 50,luminosity_distance* 1e6*PC_SI,{})
waveform2 = lalsim.SimInspiralChooseFDWaveform(mass_1*MSUN_SI, mass_2*MSUN_SI, 0, 0, 0, 0, 0, 0, luminosity_distance*1e6*PC_SI,0.4,0,0,0,0,1./8,40,2048,50,{},5)
frequency = waveform1.f0 + np.arange(len(waveform1.data.data)) * waveform1.deltaF
waveform3 = TaylorF2(frequency,injection_parameters)

import bilby

duration = 32
sampling_frequency = 2 * 1024

# Fixed arguments passed into the source model. The analysis starts at 40 Hz.
waveform_arguments = dict(waveform_approximant='TaylorF2',
                          reference_frequency=50., minimum_frequency=40.0)

# Create the waveform_generator using a LAL Binary Neutron Star source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments)

waveform4 = waveform_generator.frequency_domain_source_model(frequency, mass_1, mass_2, luminosity_distance, 0, 0, 0, 0, 0, 0, 0.4, 0, 0, 0)