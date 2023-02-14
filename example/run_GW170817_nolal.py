import numpy as np
import jax.numpy as jnp
import jax

#from lal import GreenwichMeanSiderealTime
from astropy.time import Time
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d


from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from jaxgw.PE.detector_preset import * 
from jaxgw.PE.heterodyneLikelihood import make_heterodyne_likelihood_mutliple_detector
from jaxgw.PE.detector_projection import make_detector_response

from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *

minimum_frequency = 23
maximum_frequency = 1792

trigger_time = 1187008882.43
duration = 128
post_trigger_duration = 2
epoch = duration - post_trigger_duration
#gmst = GreenwichMeanSiderealTime(trigger_time)
gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
f_ref = 20#minimum_frequency

H1_frequency, H1_data_re, H1_data_im = np.genfromtxt('../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_H1_fd_strain.txt').T
H1_data = H1_data_re + 1j*H1_data_im
H1_psd_frequency, H1_psd = np.genfromtxt('../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_H1_psd.txt').T

H1_data = H1_data[(H1_frequency>minimum_frequency)*(H1_frequency<maximum_frequency)]
H1_psd = H1_psd[(H1_frequency>minimum_frequency)*(H1_frequency<maximum_frequency)]
H1_frequency = H1_frequency[(H1_frequency>minimum_frequency)*(H1_frequency<maximum_frequency)]

L1_frequency, L1_data_re, L1_data_im = np.genfromtxt('../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_L1_fd_strain.txt').T
L1_data = L1_data_re + 1j*L1_data_im
L1_psd_frequency, L1_psd = np.genfromtxt('../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_L1_psd.txt').T

L1_data = L1_data[(L1_frequency>minimum_frequency)*(L1_frequency<maximum_frequency)]
L1_psd = L1_psd[(L1_frequency>minimum_frequency)*(L1_frequency<maximum_frequency)]
L1_frequency = L1_frequency[(L1_frequency>minimum_frequency)*(L1_frequency<maximum_frequency)]


V1_frequency, V1_data_re, V1_data_im = np.genfromtxt('../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_V1_fd_strain.txt').T
V1_data = V1_data_re + 1j*V1_data_im
V1_psd_frequency, V1_psd = np.genfromtxt('../data/GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_V1_psd.txt').T

V1_data = V1_data[(V1_frequency>minimum_frequency)*(V1_frequency<maximum_frequency)]
V1_psd = V1_psd[(V1_frequency>minimum_frequency)*(V1_frequency<maximum_frequency)]
V1_frequency = V1_frequency[(V1_frequency>minimum_frequency)*(V1_frequency<maximum_frequency)]

H1 = get_H1()
H1_response = make_detector_response(H1[0], H1[1])
L1 = get_L1()
L1_response = make_detector_response(L1[0], L1[1])
V1 = get_V1()
V1_response = make_detector_response(V1[0], V1[1])


ref_param = jnp.array([ 1.19765181e+00,  2.30871959e-01,  2.89696686e-02,
        7.57299436e-02,  3.67225424e+01,  4.97355831e-04,
        4.94017055e+00,  2.54597984e+00,  8.48429439e-01,
        3.40970408e+00, -3.42097428e-01])

data_list = [H1_data, L1_data, V1_data]
psd_list = [H1_psd, L1_psd, V1_psd]
response_list = [H1_response, L1_response, V1_response]

logL = make_heterodyne_likelihood_mutliple_detector(data_list, psd_list, response_list, gen_IMRPhenomD_polar, ref_param, H1_frequency, gmst, epoch, f_ref, 501)

n_dim = 11
n_chains = 1000
n_loop_training = 20
n_loop_production = 20
n_local_steps = 200
n_global_steps = 200
learning_rate = 0.001
max_samples = 50000
momentum = 0.9
num_epochs = 60
batch_size = 50000

guess_param = ref_param

guess_param = np.array(jnp.repeat(guess_param[None,:],int(n_chains),axis=0)*np.random.normal(loc=1,scale=0.01,size=(int(n_chains),n_dim)))
guess_param[guess_param[:,1]>0.25,1] = 0.249


print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains, seed=42)

print("Initializing MCMC model and normalizing flow model.")

prior_range = jnp.array([[1.18,1.21],[0.125,1],[-0.05,0.05],[-0.05,0.05],[1,75],[-0.01,0.02],[0,2*np.pi],[-1,1],[0,np.pi],[0,2*np.pi],[-1,1]])
optimize_prior_range = jnp.array([[1.18,1.21],[0.2,0.25],[0.0,0.3],[0.0,0.3],[1,75],[-0.01,0.02],[0,2*np.pi],[0,np.pi],[0,np.pi],[0,2*np.pi],[-np.pi/2,np.pi/2]])

initial_position = jax.random.uniform(rng_key_set[0], shape=(int(n_chains), n_dim)) * 1
for i in range(n_dim):
    initial_position = initial_position.at[:,i].set(initial_position[:,i]*(prior_range[i,1]-prior_range[i,0])+prior_range[i,0])

from ripple import Mc_eta_to_ms
m1,m2 = jax.vmap(Mc_eta_to_ms)(guess_param[:,:2])
q = m2/m1

from astropy.cosmology import Planck18 as cosmo

z = np.linspace(0.0002,0.03,10000)
dL = cosmo.luminosity_distance(z).value
dVdz = cosmo.differential_comoving_volume(z).value

def top_hat(x):
    output = 0.
    for i in range(n_dim):
        output = jax.lax.cond(x[i]>=prior_range[i,0], lambda: output, lambda: -jnp.inf)
        output = jax.lax.cond(x[i]<=prior_range[i,1], lambda: output, lambda: -jnp.inf)
    return output+jnp.log(jnp.interp(x[4],dL,dVdz))

def log_likelihood(theta):
    q = theta[1]
    theta = theta.at[1].set(q/(1+q)**2) # convert q to eta
    theta = theta.at[7].set(jnp.arccos(theta[7])) # convert cos iota to iota
    theta = theta.at[10].set(jnp.arcsin(theta[10])) # convert cos dec to dec
    return logL(theta)

def posterior(theta):
    log_prior = top_hat(theta)
    log_like = log_likelihood(theta)

    return log_like + log_prior

model = RQSpline(n_dim, 10, [128,128], 8)

print("Initializing sampler class")

posterior = posterior

mass_matrix = jnp.eye(n_dim)
mass_matrix = mass_matrix.at[0,0].set(1e-5)
mass_matrix = mass_matrix.at[1,1].set(1e-4)
mass_matrix = mass_matrix.at[2,2].set(1e-3)
mass_matrix = mass_matrix.at[3,3].set(1e-3)
mass_matrix = mass_matrix.at[5,5].set(1e-5)
mass_matrix = mass_matrix.at[9,9].set(1e-2)
mass_matrix = mass_matrix.at[10,10].set(1e-2)

local_sampler = MALA(posterior, True, {"step_size": mass_matrix*3e-2})
print("Running sampler")

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    local_sampler,
    posterior,
    model,
    n_loop_training=n_loop_training,
    n_loop_production = n_loop_production,
    n_local_steps=n_local_steps,
    n_global_steps=n_global_steps,
    n_chains=n_chains,
    n_epochs=num_epochs,
    learning_rate=learning_rate,
    momentum=momentum,
    batch_size=batch_size,
    use_global=True,
    keep_quantile=0.,
    train_thinning = 40,
)

nf_sampler.sample(initial_position)
chains, log_prob, local_accs, global_accs = nf_sampler.get_sampler_state().values()
np.savez('../data/GW170817.npz', chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)
