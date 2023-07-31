import time
from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1
from jimgw.likelihood import TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform
from jimgw.generate_noise import generate_fd_noise, generate_LVK_PSDdict
from ripple import ms_to_Mc_eta
import jax.numpy as jnp
import jax
from astropy.time import Time

from tap import Tap
import yaml
from tqdm import tqdm
from functools import partialmethod

class InjectionRecoveryParser(Tap):
    config: str 
    
    # Noise parameters
    seed: int = None
    f_sampling: int  = None
    duration: int = None
    fmin: float = None
    ifos: list[str]  = None

    # Injection parameters
    m1: float = None
    m2: float = None
    chi1: float = None
    chi2: float = None
    dist_mpc: float = None
    tc: float = None
    phic: float = None
    inclination: float = None
    polarization_angle: float = None
    ra: float = None
    dec: float = None

    # Sampler parameters
    n_dim: int = None
    n_chains: int = None
    n_loop_training: int = None
    n_loop_production: int = None
    n_local_steps: int = None
    n_global_steps: int = None
    learning_rate: float = None
    max_samples: int = None
    momentum: float = None
    num_epochs: int = None
    batch_size: int = None
    stepsize: float = None

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

freqs = jnp.arange(args.fmin, args.f_sampling/2+1./args.f_sampling, 1./args.f_sampling)

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
detector_param = {"ra": args.ra, "dec": args.dec, "gmst": gmst, "psi": args.polarization_angle}
h_sky = waveform(freqs, true_param)
key, subkey = jax.random.split(jax.random.PRNGKey(args.seed+1234))
H1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
L1.inject_signal(subkey, freqs, h_sky, detector_param)
key, subkey = jax.random.split(key)
V1.inject_signal(subkey, freqs, h_sky, detector_param)

likelihood = TransientLikelihoodFD([H1, L1], waveform, trigger_time, args.duration, post_trigger_duration)
mass_matrix = jnp.eye(11)
local_sampler_arg = {"step_size": mass_matrix*3e-3}

jim = Jim(likelihood, 
        prior,
        n_loop_training=10,
        n_loop_production = 10,
        n_local_steps=300,
        n_global_steps=300,
        n_chains=10,
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

jim.maximize_likleihood([prior.xmin, prior.xmax])
key, subkey = jax.random.split(key)
jim.sample(subkey)

# labels = ['Mc', 'eta', 'chi1', 'chi2', 'dist_mpc', 'tc', 'phic', 'cos_inclination', 'polarization_angle', 'ra', 'sin_dec']

# print("Saving to output")

# chains, log_prob, local_accs, global_accs, loss_vals = nf_sampler.get_sampler_state(training=True).values()
# chains, log_prob, local_accs, global_accs = nf_sampler.get_sampler_state().values()

# # Fetch output parameters

# output_path = args['output_path']
# downsample_factor = args['downsample_factor']

# np.savez(args['output_path'], chains=chains[:,::downsample_factor], log_prob=log_prob[:,::downsample_factor], local_accs=local_accs[:,::downsample_factor], global_accs=global_accs[:,::downsample_factor], loss_vals=loss_vals, labels=labels, true_param=true_param, true_log_prob=LogLikelihood(true_param))
