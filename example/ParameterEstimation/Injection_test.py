# Import packages

from xml.sax.handler import property_declaration_handler
import scipy.signal as ssig
import lalsimulation as lalsim 
import numpy as np
import jax.numpy as jnp
import jax

# from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from ripple import ms_to_Mc_eta
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from jaxgw.PE.detector_preset import * 
from jaxgw.PE.heterodyneLikelihood import make_heterodyne_likelihood


from flowMC.nfmodel.realNVP import RealNVP
from flowMC.sampler.MALA import make_mala_sampler
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *

import matplotlib.pyplot as plt

psd_func_dict = {
    'H1': lalsim.SimNoisePSDaLIGOZeroDetHighPower,
    'L1': lalsim.SimNoisePSDaLIGOZeroDetHighPower,
    'V1': lalsim.SimNoisePSDAdvVirgo,
}
ifos = list(psd_func_dict.keys())

# define center of time array
tgps_geo = 1126259462.423

# define sampling rate and duration
fsamp = 8192
duration = 4

delta_t = 1/fsamp
tlen = int(round(duration / delta_t))

freqs = np.fft.rfftfreq(tlen, delta_t)
delta_f = freqs[1] - freqs[0]



# we will want to pad low frequencies; the function below applies a
# prescription to do so smoothly, but this is not really needed: you
# could just set all values below `fmin` to a constant.
fmin = 30
def pad_low_freqs(f, psd_ref):
    return psd_ref + psd_ref*(fmin-f)*np.exp(-(fmin-f))/3

psd_dict = {}
for ifo in ifos:
    psd = np.zeros(len(freqs))
    for i,f in enumerate(freqs):
        if f >= fmin:
            psd[i] = psd_func_dict[ifo](f)
        else:
            psd[i] = pad_low_freqs(f, psd_func_dict[ifo](fmin))
    psd_dict[ifo] = psd



rng = np.random.default_rng(12345)

noise_fd_dict = {}
for ifo, psd in psd_dict.items():
    var = psd / (4.*delta_f)  # this is the variance of LIGO noise given the definition of the likelihood function
    noise_real = rng.normal(size=len(psd), loc=0, scale=np.sqrt(var))
    noise_imag = rng.normal(size=len(psd), loc=0, scale=np.sqrt(var))
    noise_fd_dict[ifo] = noise_real + 1j*noise_imag



# These are the parameters of the injected signal
m1 = 50.0
m2 = 10.0
Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
chi1 = 0.4
chi2 = -0.3
dist_mpc = 400.0
tc = 2.0
phic = 0.0
inclination = np.pi
polarization_angle = np.pi/2
ra = 0.3
dec = 0.5

detector_presets = {'H1': get_H1()}

theta_ripple = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle])
theta_ripple_vec = np.array(jnp.repeat(theta_ripple[None,:],10000,axis=0)*np.random.normal(loc=1,scale=0.0001,size=(10000,9)))
theta_ripple_vec[theta_ripple_vec[:,1]>0.25,1] = 0.25

f_list = freqs[freqs>fmin]
hp = gen_IMRPhenomD_polar(f_list, theta_ripple)
noise_psd = psd[freqs>fmin]
data = noise_psd + hp[0]


# plt.figure(figsize=(15,5))
# plt.loglog(f_list, np.abs(data), label="Signal", alpha=0.3)
# plt.loglog(f_list, np.abs(noise_fd_dict["H1"][freqs>fmin]), label="H1 noise", alpha=0.3)
# plt.loglog(f_list, np.abs(hp[0]), label="H1 waveform", alpha=0.3)
# plt.ylim(1e-25, 1e-21)
# plt.legend()
# plt.show()

@jax.jit
def LogLikelihood(theta):
    h_test = gen_IMRPhenomD_polar(f_list, theta)
    df = f_list[1] - f_list[0]
    match_filter_SNR = 4*jnp.sum((jnp.conj(h_test[0])*data)/noise_psd*df).real
    optimal_SNR = 4*jnp.sum((jnp.conj(h_test[0])*h_test[0])/noise_psd*df).real
    return (-match_filter_SNR+optimal_SNR/2)

theta_ref = jnp.array([Mc, 0.138, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle])

h_function = lambda f,theta:gen_IMRPhenomD_polar(f,theta)[0]

logL = make_heterodyne_likelihood(data, h_function, theta_ref, noise_psd, f_list, 101)

L1 = jax.vmap(LogLikelihood)(theta_ripple_vec)
L2 = jax.vmap(jax.jit(logL))(theta_ripple_vec)
# Create data

# Create likelihood object

# Samples the likelihood with flowMC

n_dim = 9
n_chains = 100
n_loop = 5
n_local_steps = 1000
n_global_steps = 0
stepsize = 0.01

print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains, seed=42)

print("Initializing MCMC model and normalizing flow model.")

initial_position = jax.random.uniform(rng_key_set[0], shape=(n_chains, n_dim)) * 1
initial_position = initial_position.at[:,0].set(initial_position[:,0]*20 + 10)
initial_position = initial_position.at[:,1].set(initial_position[:,1]*0.25)
initial_position = initial_position.at[:,2].set(initial_position[:,2]*2 - 1)
initial_position = initial_position.at[:,3].set(initial_position[:,3]*2 - 1)
initial_position = initial_position.at[:,4].set(initial_position[:,4]*2000)
initial_position = initial_position.at[:,5].set(initial_position[:,5]*10-5)
initial_position = initial_position.at[:,6].set(initial_position[:,6]*np.pi-np.pi/2)
initial_position = initial_position.at[:,7].set(initial_position[:,7]*np.pi-np.pi/2)
initial_position = initial_position.at[:,8].set(initial_position[:,8]*2*np.pi )

model = RealNVP(10, n_dim, 64, 1)
# run_mcmc = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 0, None), out_axes=0)

print("Initializing sampler class")

logL = jax.jit(logL)
dlogL = jax.jit(jax.grad(logL)) # compiling each of these function first should improve the performance by a lot

local_sampler = make_mala_sampler(logL, dlogL,1e-4)

nf_sampler = Sampler(n_dim, rng_key_set, model, local_sampler,
                    logL,
                    d_likelihood=dlogL,
                    n_loop=n_loop,
                    n_local_steps=n_local_steps,
                    n_global_steps=n_global_steps,
                    n_chains=n_chains,
                    stepsize=stepsize,
                    use_global=False,)

local_sampler = make_mala_sampler(logL, dlogL)
