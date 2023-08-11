import time
from jimgw.jim import Jim
from jimgw.detector import GroundBased2G
from jimgw.detector import H1 as H1_class
from jimgw.likelihood import TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform
import jax.numpy as jnp
import jax
import numpy as np
from scipy.interpolate import interp1d
jax.config.update('jax_enable_x64', True)
import corner
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

DEG_TO_RAD = jnp.pi / 180.

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
start = gps - 2
end = gps + 2
fmin = 20.
fmax = 1024.
key = 0

ifos = ['H1', 'L1']


# Now we load the data
def read_complex_matrix_from_file(file_path):
    complex_matrix = []

    with open(file_path, 'r') as file:
        for line in file:
            row_elements = line.strip().split()  # Split the line by whitespace
            complex_row = [complex(element.replace('(', '').replace(')', '')) for element in row_elements]
            complex_matrix.append(complex_row)

    return complex_matrix

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

catalog_list = []
for i in range(1):
    for j in range(0, 1):
        for k in range(0, 10): 
            for l in range(1, 10):
                catalog_list.append(str(i)+str(j)+str(k)+str(l))
catalog_list = ['0001']
                
#with tqdm(total=len(catalog_list)) as pbar:
for string in catalog_list:
    print('Starting data collection')
    path = "/mnt/home/averhaeghe/ceph/NR_waveforms_noisy/NR_"+ string + ".txt"
    data = read_complex_matrix_from_file(path)

    print('Managed to read the data')

    freqs = [item[0].real for item in data]
    signal = [item[1] for item in data]

    print('Stored the data succesfully')

    freqs = [freq for freq in freqs if freqs.index(freq)%3==0]
    freqs = jnp.array(freqs)
    print('time to get the PSD')
    signal = [sig for sig in signal if signal.index(sig)%3==0]
    signal = jnp.array(signal)
    psd = H1_class.load_psd(freqs)


    H1 = GroundBased2G('H1',
    latitude = (46 + 27. / 60 + 18.528 / 3600) * DEG_TO_RAD,
    longitude = -(119 + 24. / 60 + 27.5657 / 3600) * DEG_TO_RAD,
    xarm_azimuth = 125.9994 * DEG_TO_RAD,
    yarm_azimuth = 215.9994 * DEG_TO_RAD,
    xarm_tilt = -6.195e-4,
    yarm_tilt = 1.25e-5,
    elevation = 142.554,
    mode='pc')

    H1.set_data(freqs, signal, psd)

    print('Data collection done')

    likelihood = TransientLikelihoodFD([H1], RippleIMRPhenomD(), gps, 4, 2)
    prior = Uniform(
        xmin = [10, 0.125, -1., -1., 0., -0.05, 0., -1, 0., 0.,-1.],
        xmax = [80., 1., 1., 1., 2000., 0.05, 2*jnp.pi, 1., jnp.pi, 2*jnp.pi, 1.],
        naming = ["M_c", "q", "s1_z", "s2_z", "d_L", "t_c", "phase_c", "cos_iota", "psi", "ra", "sin_dec"],
        transforms = {"q": ("eta", lambda q: q/(1+q)**2),
                    "cos_iota": ("iota",lambda cos_iota: jnp.arccos(jnp.arcsin(jnp.sin(cos_iota/2*jnp.pi))*2/jnp.pi)),
                    "sin_dec": ("dec",lambda sin_dec: jnp.arcsin(jnp.arcsin(jnp.sin(sin_dec/2*jnp.pi))*2/jnp.pi))} # sin and arcsin are periodize cos_iota and sin_dec
    )

    mass_matrix = jnp.eye(11)
    mass_matrix = mass_matrix.at[1,1].set(1e-3)
    mass_matrix = mass_matrix.at[5,5].set(1e-3)
    local_sampler_arg = {"step_size": mass_matrix*3e-3}

    jim = Jim(likelihood, 
            prior,
            n_loop_training=10,
            n_loop_production = 10,
            n_local_steps=300,
            n_global_steps=300,
            n_chains=150,
            n_epochs=300,
            learning_rate = 0.001,
            momentum = 0.9,
            batch_size = 50000,
            use_global=True,
            keep_quantile=0.,
            train_thinning = 40,
            local_sampler_arg = local_sampler_arg,
            )

    #print(jim.maximize_likelihood((prior.xmin, prior.xmax)))

    if True:
            key, subkey = jax.random.split(jax.random.PRNGKey(1234))
            jim.sample(subkey)
            chains = jim.get_samples(training=True)
            chains = np.array(chains)
            figure = corner.corner(
            chains.reshape(-1, 11), labels = prior.naming, max_n_ticks = 4, plot_datapoints=False, quantiles=(0.16, 0.5, 0.84), show_titles = True)
            figure.set_size_inches(17, 17)
            figure.suptitle("Visualize PE run", fontsize = 33)
            plt.show(block=False)
            plt.savefig("/mnt/home/averhaeghe/ceph/debugging/PE____.png")
    #pbar.update(1)


path = "/mnt/home/averhaeghe/ceph/NR_waveforms/param_0001.txt"
intr_param =read_json_file(path)
q = intr_param["eta"]**(-1)-2+jnp.sqrt((intr_param["eta"]**(-1)-2)**2-4)
q /= 2
#intr_param = np.array([intr_param['chirp_mass'], q, intr_param['chi1'], intr_param['chi2']])
 

# MLE_est = jim.maximize_likelihood((prior.xmin, prior.xmax))
# log_MLE = likelihood.evaluate(prior.transform(MLE_est),{})
#likelihood.evaluate(prior.transform(jim.maximize_likelihood((prior.xmin, prior.xmax))),{})
# log_act = likelihood.evaluate(prior.transform(jnp.array([intr_param['chirp_mass'], q, intr_param['chi1'], intr_param['chi2'], 440, 0, 0, 1, 0, 0, 0])),{})
