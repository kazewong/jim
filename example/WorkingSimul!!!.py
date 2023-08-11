import lal
import lalsimulation as lalsim
from lal import MSUN_SI, PC_SI
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from corner import corner
import matplotlib.pyplot as plt
path = '/mnt/ceph/users/misi/lvcnr-lfs/SXS/SXS_BBH_0001_Res5.h5'
#path = '/mnt/home/averhaeghe/ceph/IMR/IMR+.txt'

mass1 = 20
mass2 = 19
M_c = (mass1*mass2)**(3/5)/(mass1+mass2)**(1/5)
distance = 400 * 1e6*PC_SI
inclination = 0.4
phiRef = 0.1
deltaT = 1./4096
fStart = 30
fRef = 30
# Compute spins in the LAL frame
s1x, s1y, s1z, s2x, s2y, s2z = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(fRef, mass1+mass2, path)
# Create a dictionary and pass /PATH/TO/H5File
params = lal.CreateDict()
lalsim.SimInspiralWaveformParamsInsertNumRelData(params, path)
'''mass1 = 20
mass2 = 19
s1z = 0.5
s2z = -0.5
tc = 0.0 # Time of coalescence in seconds
phiRef = 0.0 # Time of coalescence
distance = 440 # Distance to source in Mpc
inclination = 0.0 # Inclination Angle
polarization_angle = 0.2 # Polarization angle
deltaT = 1./4096
fStart = 20
fRef = 20
'''
def delete_zero_rows(matrix):
    non_zero_rows = jnp.any(matrix != 0, axis=1)
    non_zero_indices = jnp.where(non_zero_rows)[0]
    result = matrix[non_zero_indices]
    return result

# Generate GW polarisations
hp, hc = lalsim.SimInspiralChooseTDWaveform(mass1 * MSUN_SI, mass2 * MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z,distance, inclination, phiRef, np.pi/2, 0., 0., deltaT,fStart, fRef, params, approximant=lalsim.NR_hdf5)
'''data = np.genfromtxt(path)
freq = data[:,0]
hp = data[:,1] 
hc =  data[:,2]
'''
#t_array = np.arange(0, len(hp.data.data)*deltaT, deltaT)
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

class InjectionRecoveryParser(Tap):
    
    # Noise parameters
    seed: int = 0
    f_sampling: int  = 4096
    duration: int = 2
    fmin: float = 20
    ifos: list[str] = H1, L1

    # Injection parameters
    m1: float = mass1
    m2: float = mass2
    chi1: float = s1z
    chi2: float = s2z
    dist_mpc: float =  400
    tc: float = 0
    phic: float = 0
    inclination: float = inclination
    polarization_angle: float = 0
    ra: float = 0
    dec: float = 0

    # Sampler parameters
    n_dim: int = 11
    n_chains: int = 100
    n_loop_training: int = 10
    n_loop_production: int = 10
    n_local_steps: int = 100
    n_global_steps: int = 100
    learning_rate: float = 1e-3
    max_samples: int = 10000
    momentum: float = 0.9
    num_epochs: int = 300
    batch_size: int = 50000
    stepsize: float = 1e-3

    # Output parameters
    output_path: str = None
    downsample_factor: int = None


args = InjectionRecoveryParser().parse_args()

# opt = vars(args)
# yaml_var = yaml.load(open(opt['config'], 'r'), Loader=yaml.FullLoader)
# opt.update(yaml_var)

# Fetch noise parameters 

print("Constructing detectors")
print("Making noises")

#Fetch injection parameters and inject signal

print("Injection signals")

freqs = jnp.linspace(args.fmin, args.f_sampling/2, args.duration*args.f_sampling//2)

Mc, eta = ms_to_Mc_eta(jnp.array([args.m1, args.m2]))
f_ref = 30.0
trigger_time = 1126259462.4
post_trigger_duration = 2
epoch = args.duration - post_trigger_duration
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
true_param = jnp.array([Mc, eta, args.chi1, args.chi2, args.dist_mpc, args.tc, args.phic, args.inclination, args.polarization_angle, args.ra, args.dec])
true_param = prior.add_name(true_param, with_transform=True)
detector_param = {"ra": args.ra, "dec": args.dec, "gmst": gmst, "psi": args.polarization_angle, "epoch": epoch, "t_c": args.tc}


# h_sky = waveform(freqs, true_param)
freqs = np.fft.rfftfreq(len(hp.data.data), deltaT)
h_sky = {'p': np.fft.rfft(hp.data.data)[freqs>20]*deltaT*jnp.exp(4j*freqs[freqs>20]), 'c': np.fft.rfft(hc.data.data)[freqs>20]*deltaT*jnp.exp(4j*freqs[freqs>20])}
freqs = freqs[freqs>20]
key, subkey = jax.random.split(jax.random.PRNGKey(args.seed+1234))
H1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
L1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
V1.inject_signal(subkey, freqs, h_sky, detector_param)

likelihood = TransientLikelihoodFD([H1, L1], waveform, trigger_time, args.duration, post_trigger_duration)
mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[1,1].set(1e-3)
mass_matrix = mass_matrix.at[5,5].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix*3e-3}

jim = Jim(likelihood, 
        prior,
        n_loop_training=20,
        n_loop_production = 10,
        n_local_steps=300,
        n_global_steps=300,
        n_chains=400,
        n_epochs=300,
        learning_rate = 0.001,
        momentum = 0.9,
        batch_size = 50000,
        use_global=True,
        keep_quantile=0.,
        train_thinning = 40,
        local_sampler_arg = local_sampler_arg,
        seed = args.seed,
        )


key, subkey = jax.random.split(jax.random.PRNGKey(1234))
jim.sample(subkey)
chains = jim.get_samples(training=True)
chains = np.array(chains)
chains_resh = delete_zero_rows(chains.reshape(-1, 11))
            
figure = corner(chains.reshape(-1,11), labels = prior.naming, max_n_ticks = 4, plot_datapoints=False, quantiles=(0.16, 0.5, 0.84), show_titles = True)
figure.set_size_inches(17, 17)
figure.suptitle("Visualize PE run", fontsize = 33)
plt.show(block=False)
plt.savefig("/mnt/home/averhaeghe/ceph/PE/new_coeffs/IMR_phase_shift2.png")
