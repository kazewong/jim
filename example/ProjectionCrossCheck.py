import bilby
from gwpy.timeseries import TimeSeries

logger = bilby.core.utils.logger
outdir = "outdir"
label = "GW190425"

trigger_time = 1240215503.0
detectors = ["L1", "V1"]
maximum_frequency = 512
minimum_frequency = 20
roll_off = 0.4  # Roll off duration of tukey window in seconds, default is 0.4s
duration = 128  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

psd_duration = 1024
psd_start_time = start_time - psd_duration
psd_end_time = start_time

ifo_list = bilby.gw.detector.InterferometerList([])
for det in detectors:
    logger.info("Downloading analysis data for ifo {}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    data = TimeSeries.fetch_open_data(det, start_time, end_time)
    ifo.strain_data.set_from_gwpy_timeseries(data)

    logger.info("Downloading psd data for ifo {}".format(det))
    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)
    psd_alpha = 2 * roll_off / duration
    psd = psd_data.psd(
        fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median"
    )
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value
    )
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency
    ifo_list.append(ifo)

logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)


priors = bilby.gw.prior.BBHPriorDict(filename="./GW190425.prior")

priors['tilt_1'] = bilby.gw.prior.DeltaFunction(peak=0.0,name='tilt_1',latex_label='$\\theta_1$', unit=None)
priors['tilt_2'] = bilby.gw.prior.DeltaFunction(peak=0.0,name='tilt_2',latex_label='$\\theta_2$', unit=None)
priors['phi_12'] = bilby.gw.prior.DeltaFunction(peak=0.0,name='phi_12',latex_label='$\\phi_{12}$', unit=None)
priors['phi_jl'] = bilby.gw.prior.DeltaFunction(peak=0.0,name='phi_jl',latex_label='$\\phi_{JL}$', unit=None)
priors["geocent_time"] = bilby.core.prior.Uniform(
    trigger_time - 0.1, trigger_time + 0.1, name="geocent_time"
)


waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments={
        "waveform_approximant": "IMRPhenomD",
        "reference_frequency": 20,
    },
)

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list,
    waveform_generator,
    priors=priors,
    time_marginalization=True,
    phase_marginalization=False,
    distance_marginalization=True,
)

import lal
import lalsimulation as lalsim
import numpy as np
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from ripple import Mc_eta_to_ms
from jaxgw.PE.detector_preset import * 
from jaxgw.PE.heterodyneLikelihood import make_heterodyne_likelihood_mutliple_detector
from jaxgw.PE.detector_projection import make_detector_response
def get_lal_waveform(f,theta):
    Mc, eta, a_1, a_2, distance, t_c, phi_0, theta_jn, psi = theta 
    m1,m2 = Mc_eta_to_ms([Mc,eta])
    del_f = f[1]-f[0]
    f_l = f[0]
    f_u = f[-1]
    f_ref = f[0]

    m1_kg = m1 * lal.MSUN_SI
    m2_kg = m2 * lal.MSUN_SI
    distance = distance * 1e6 * lal.PC_SI
    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")

    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        float(m1_kg),
        float(m2_kg),
        0.0,
        0.0,
        a_1,
        0.0,
        0.0,
        a_2,
        float(distance),
        theta_jn,
        0.0,
        0.0,
        0.0,
        0.0,
        del_f,
        f_l,
        f_u,
        f_ref,
        None,
        approximant,
    )
    freqs = np.arange(len(hp.data.data)) * del_f
    mask_lal = (freqs >= f_l) & (freqs < f_u)
    return hp, hc, freqs, mask_lal



L1 = get_L1()
L1_response = make_detector_response(L1[0], L1[1])
V1 = get_V1()
V1_response = make_detector_response(V1[0], V1[1])

f = waveform_generator.frequency_array
f = f[f>40]
parameters = priors.sample()
Mc = parameters['chirp_mass']
eta = parameters['mass_ratio']/(1+parameters['mass_ratio'])**2
a_1 = parameters['a_1']
a_2 = parameters['a_2']
distance = parameters['luminosity_distance']
phi_0 = parameters['phase']
psi = parameters['psi']
theta_jn = parameters['theta_jn']
dec = parameters['dec']
ra = parameters['ra']

ripple_params = np.array([Mc, eta, a_1, a_2, distance, 0, phi_0, theta_jn, psi,  dec, ra])
m1,m2 = Mc_eta_to_ms([Mc, eta])
ripple_h = gen_IMRPhenomD_polar(f, ripple_params)
bilby_h = waveform_generator.frequency_domain_strain(parameters)
lal_bilby_h = bilby.gw.source.lal_binary_black_hole(f, m1, m2, distance, a_1, 0, 0, a_2,0, 0, theta_jn, phi_0)
lal_h = get_lal_waveform(f,ripple_params[:9])


from ripple import get_eff_pads, get_match_arr
from ripple.waveforms import IMRPhenomD, IMRPhenomD_utils
import matplotlib.pyplot as plt
import lalsimulation as lalsim
from ripple import ms_to_Mc_eta
def plot_waveforms():
    # Get a frequency domain waveform
    # source parameters
    m1_msun = 49.0
    m2_msun = 48.0
    chi1 = [0, 0, 0.5]
    chi2 = [0, 0, 0.5]
    tc = 0.0
    phic = 0.0
    dist_mpc = 440
    inclination = 0.123
    phi_ref = 0
    polarization_angle = 0.0

    Mc, eta = ms_to_Mc_eta(jnp.array([m1_msun, m2_msun]))

    theta_ripple = np.array(
        [Mc, eta, chi1[2], chi2[2], dist_mpc, tc, phic, -inclination, polarization_angle]
    )

    theta = np.array([m1_msun, m2_msun, chi1[2], chi2[2]])
    f_l = 20
    f_u = 1024
    del_f = 0.01
    fs = np.arange(f_l, f_u, del_f)

    coeffs = IMRPhenomD_utils.get_coeffs(theta)
    _, _, f3, f4, _, _ = IMRPhenomD_utils.get_transition_frequencies(
        theta, coeffs[5], coeffs[6]
    )

    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")

    f_ref = f_l
    m1_kg = m1_msun * lal.MSUN_SI
    m2_kg = m2_msun * lal.MSUN_SI
    distance = dist_mpc * 1e6 * lal.PC_SI

    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1_kg,
        m2_kg,
        chi1[0],
        chi1[1],
        chi1[2],
        chi2[0],
        chi2[1],
        chi2[2],
        distance,
        inclination,
        phi_ref,
        0,
        0.0,
        0.0,
        del_f,
        f_l,
        f_u,
        f_ref,
        None,
        approximant,
    )
    freqs = np.arange(len(hp.data.data)) * del_f
    mask_lal = (freqs >= f_l) & (freqs < f_u)

    hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(fs, theta_ripple)

    plt.figure(figsize=(15, 5))
    plt.plot(
        freqs[mask_lal], hp.data.data[mask_lal].real, label="hp lalsuite", alpha=0.3
    )

    plt.plot(
        fs,
        hp_ripple.real,
        label="hp ripple",
        alpha=0.3,
    )

    plt.axvline(x=f3, ls="--")
    plt.axvline(x=f4, ls="--")
    plt.legend()
    plt.xlim(0, 300)
    plt.xlabel("Frequency")
    plt.ylabel("hf")


