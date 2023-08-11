import lal
import lalsimulation as lalsim
from lal import MSUN_SI, PC_SI
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import corner
import matplotlib.pyplot as plt
from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1
from jimgw.likelihood import TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform
from ripple import ms_to_Mc_eta
import jax.numpy as jnp
import jax
from astropy.time import Time
from tap import Tap
import json
import sxs
import re
import os
from scipy import signal
from tqdm import tqdm

def list_files_in_directory(directory_path):
    try:
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        return files
    except OSError:
        return []

def save_dict_to_txt(data, filename):
    with open(filename, 'w') as f:
        for key, value in data.items():
            f.write(f"{key}: {value}\n")

def load_dict_from_txt(filename):
    loaded_data = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            key, value_str = line.strip().split(': ')
            value = np.array([float(v) for v in value_str.strip('[]').split()])
            loaded_data[key] = value
    return loaded_data

def extract_info(input_string):
    pattern = r'SXS_BBH_(\d+)_Res(\d+)\.h5'
    match = re.match(pattern, input_string)
    
    if match:
        number = int(match.group(1))
        resolution =int(match.group(2))
        return number, resolution
    else:
        return None
def format(lst):
    for element in lst:
        element[1] = str(element[1])
        element[0] = str(element[0])
        if len(element[0]) == 1:
            element[0] = '000'+element[0]
        elif len(element[0]) == 2:
            element[0] = '00'+element[0]
        elif len(element[0]) == 3:
            element[0] = '0'+element[0]
    return lst

poss_NR = list_files_in_directory('/mnt/ceph/users/misi/lvcnr-lfs/SXS')
NR_files = []
for file in poss_NR:
    file = extract_info(file)
    elements = [file[0], file[1]]
    if elements[1] % 2 == 1:
        NR_files.append(elements)
NR_files = sorted(NR_files, key = lambda x: (x[0], -x[1]))

ind = [[1,5]]
for i in NR_files:
    if i[0] != ind[-1][0]:
        ind.append(i)
NR_files = ind
NR_files = format(NR_files)
NR_files = NR_files[:1]

with tqdm(total=len(NR_files)) as pbar:
    for numbers in NR_files:
        try:
            path = '/mnt/ceph/users/misi/lvcnr-lfs/SXS/SXS_BBH_'+numbers[0]+'_Res'+numbers[1]+'.h5'

            metadata = sxs.load("SXS:BBH:"+numbers[0]+"/Lev/metadata.json", download=None)
            #Loading metadata

            q = metadata.reference_mass_ratio
            chi1 = metadata.reference_dimensionless_spin1[2]
            chi2 = metadata.reference_dimensionless_spin2[2]

            mass1 = 40*q/(1+q)
            mass2 = 40/(1+q)

            #Initializing waveform parameters
            distance = 400 * 1e6*PC_SI
            dist_mpc = 400
            tc = 0
            phic = 0
            inclination = 0.4
            polarization_angle = 0
            ra =0
            dec = 0
            phiRef = 0.

            duration = 2
            deltaT = 1./4096
            fStart = 30
            fRef = -1
            # Compute spins in the LAL frame
            s1x, s1y, s1z, s2x, s2y, s2z = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(fRef, mass1+mass2, path)

            params = lal.CreateDict()
            lalsim.SimInspiralWaveformParamsInsertNumRelData(params, path)

            #Generate hp and hc
            hp, hc = lalsim.SimInspiralChooseTDWaveform(mass1 * MSUN_SI, mass2 * MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z,distance, inclination, phiRef, np.pi/2, 0., 0., deltaT,fStart, fRef, params, approximant=lalsim.NR_hdf5)


            #Fetch injection parameters and inject signal

            seed = 1234
            Mc, eta = ms_to_Mc_eta(jnp.array([mass1, mass2]))
            f_ref = 30.0
            trigger_time = 1126259462.4
            post_trigger_duration = 2
            epoch = duration - post_trigger_duration
            gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad

            waveform = RippleIMRPhenomD(f_ref=f_ref)
            prior = Uniform(
                xmin = [10, 0.125, -1., -1., 0., -0.05, 0., -1, 0., 0.,-1.],
                xmax = [80., 1., 1., 1., 2000., 0.05, 2*jnp.pi, 1., jnp.pi, 2*jnp.pi, 1.],
                naming = ["M_c", "q", "s1_z", "s2_z", "d_L", "t_c", "phase_c", "cos_iota", "psi", "ra", "sin_dec"],
                transforms = {"q": ("eta", lambda q: q/(1+q)**2),
                            "cos_iota": ("iota",lambda cos_iota: jnp.arccos(jnp.arcsin(jnp.sin(cos_iota/2*jnp.pi))*2/jnp.pi)),
                            "sin_dec": ("dec",lambda sin_dec: jnp.arcsin(jnp.arcsin(jnp.sin(sin_dec/2*jnp.pi))*2/jnp.pi))} # sin and arcsin are periodize cos_iota and sin_dec
            )
            true_param = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle, ra, dec])
            true_param = prior.add_name(true_param, with_transform=True)
            detector_param = {"ra": ra, "dec": dec, "gmst": gmst, "psi": polarization_angle, "epoch": epoch, "t_c": tc}

            print('Injecting signal')

            # h_sky = waveform(freqs, true_param)
            freqs = np.fft.rfftfreq(len(hp.data.data), deltaT)
            h_sky = {'p': np.fft.rfft(hp.data.data)[freqs>20]*deltaT*jnp.exp(4j*freqs[freqs>20]), 'c': np.fft.rfft(hc.data.data)[freqs>20]*deltaT*jnp.exp(4j*freqs[freqs>20])}
            freqs = freqs[freqs>20]
            key, subkey = jax.random.split(jax.random.PRNGKey(seed+1234))
            H1.inject_signal(subkey, freqs, h_sky, detector_param)
            key, subkey = jax.random.split(key)
            L1.inject_signal(subkey, freqs, h_sky, detector_param)
            key, subkey = jax.random.split(key)
            V1.inject_signal(subkey, freqs, h_sky, detector_param)

            likelihood = TransientLikelihoodFD([H1, L1], waveform, trigger_time, duration, post_trigger_duration)
            mass_matrix = jnp.eye(11)
            mass_matrix = mass_matrix.at[1,1].set(1e-3)
            mass_matrix = mass_matrix.at[5,5].set(1e-3)
            local_sampler_arg = {"step_size": mass_matrix*3e-3}

            print('Running sampler')
            jim = Jim(likelihood, 
                    prior,
                    n_loop_training=20,
                    n_loop_production = 10,
                    n_local_steps=300,
                    n_global_steps=300,
                    n_chains=1000,
                    n_epochs=300,
                    learning_rate = 0.001,
                    momentum = 0.9,
                    batch_size = 50000,
                    use_global=True,
                    keep_quantile=0.,
                    train_thinning = 40,
                    local_sampler_arg = local_sampler_arg,
                    seed = seed,
                    )


            key, subkey = jax.random.split(jax.random.PRNGKey(1234))
            jim.sample(subkey)
            chains = jim.get_samples(training=True)
            chains = np.array(chains)
                        
            figure = corner.corner(chains.reshape(-1,11), labels = prior.naming, max_n_ticks = 4, plot_datapoints=False, quantiles=(0.16, 0.5, 0.84), show_titles = True)
            figure.set_size_inches(17, 17)
            figure.suptitle("Visualize PE run", fontsize = 33)
            plt.show(block=False)
            plt.savefig('/mnt/home/averhaeghe/ceph/PE/new_coeffs/NR_'+str(numbers[0])+'.png')
            est = {}
            for i in range(len(prior.naming)):
                est[prior.naming[i]] = corner.quantile(chains[:,:,i],(0.16,0.5,0.84))
            save_dict_to_txt(est,'/mnt/home/averhaeghe/ceph/PE/new_coeffs/NR_'+numbers[0]+'_param.txt')
        except RuntimeError:
            NR_files.remove(numbers)
            print('Removed file '+ path)
        pbar.update(1)


