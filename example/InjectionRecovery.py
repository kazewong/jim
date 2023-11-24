from jimgw.jim import Jim
from jimgw.detector import H1, L1, V1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomPv2
from jimgw.prior import Uniform
from ripple import ms_to_Mc_eta
import jax.numpy as jnp
import jax
from astropy.time import Time

from tap import Tap
import yaml
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)


class InjectionRecoveryParser(Tap):
    config: str 
    
    # Noise parameters
    seed: int = 0
    f_sampling: int  = 2048
    duration: int = 4
    fmin: float = 20.0
    ifos: list[str]  = ["H1", "L1", "V1"]

    # Injection parameters
    m1: float = 30.0
    m2: float = 25.0
    s1_theta: float = 0.04
    s1_phi: float = 0.02
    s1_mag: float = 0.1
    s2_theta: float = 0.01
    s2_phi: float = 0.03
    s2_mag: float = 0.05
    dist_mpc: float = 400.
    tc: float = 0.
    phic: float = 0.1
    inclination: float = 0.5
    polarization_angle: float = 0.7
    ra: float = 1.2
    dec: float = 0.3

    # Sampler parameters
    n_dim: int = 15
    n_chains: int = 1200
    n_loop_training: int = 500
    n_loop_production: int = 10
    n_local_steps: int = 300
    n_global_steps: int = 500
    learning_rate: float = 0.001
    max_samples: int = 60000
    momentum: float = 0.9
    num_epochs: int = 200
    batch_size: int = 60000
    stepsize: float = 0.01
    use_global: bool = True
    keep_quantile: float = 0.0
    train_thinning: int = 1
    output_thinning: int = 30
    num_layers: int = 6
    hidden_size: list[int] = [32,32]
    num_bins: int = 8

    # Output parameters
    output_path: str = "./"
    downsample_factor: int = 10


args = InjectionRecoveryParser().parse_args()
opt = vars(args)
yaml_var = yaml.load(open(opt['config'], 'r'), Loader=yaml.FullLoader)
opt.update(yaml_var)
# Fetch noise parameters 
print("s1_mag:", args.s1_mag)
print("Constructing detectors")
print("Making noises")

#Fetch injection parameters and inject signal

print("Injection signals")

freqs = jnp.arange(args.fmin, args.f_sampling/2, 1/args.duration)
#freqs = jnp.linspace(args.fmin, args.f_sampling/2, args.duration*args.f_sampling/2)

Mc, eta = ms_to_Mc_eta(jnp.array([args.m1, args.m2]))
f_ref = args.fmin
trigger_time = 1126259462.4
post_trigger_duration = 2
epoch = args.duration - post_trigger_duration
gmst = Time(trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad

waveform = RippleIMRPhenomPv2(f_ref=f_ref)
prior_tc_low = args.tc - 0.2 * jnp.abs(args.tc)
prior_tc_high = args.tc + 0.2 * jnp.abs(args.tc)
prior = Uniform(
    xmin = [0.8*Mc, 0.125, 0, 0, 0, 0, 0, 0, 300., prior_tc_low, 0., -1, 0., 0.,-1.],
    #xmin = [10, 0.125, 0, 0, 0, 0, 0, 0, 300., 0., 0., -1, 0., 0.,-1.],
    xmax = [1.2*Mc, 1., jnp.pi, 2*jnp.pi, 1., jnp.pi, 2*jnp.pi, 1., 2000., prior_tc_high, 2*jnp.pi, 1., jnp.pi, 2*jnp.pi, 1.],
    #xmax = [80, 1., jnp.pi, 2*jnp.pi, 1., jnp.pi, 2*jnp.pi, 1., 2000., 1.2*args.tc, 2*jnp.pi, 1., jnp.pi, 2*jnp.pi, 1.],
    naming = ["M_c", "q", "s1_theta", "s1_phi", "s1_mag", "s2_theta", "s2_phi", "s2_mag", "d_L", "t_c", "phase_c", "cos_iota", "psi", "ra", "sin_dec"],
    transforms = {"q": ("eta", lambda params: params['q']/(1+params['q'])**2),
                 "s1_theta": ("s1_x", lambda params: jnp.sin(params['s1_theta'])*jnp.cos(params['s1_phi'])*params['s1_mag']),
                 "s1_phi": ("s1_y", lambda params: jnp.sin(params['s1_theta'])*jnp.sin(params['s1_phi'])*params['s1_mag']),
                 "s1_mag": ("s1_z", lambda params: jnp.cos(params['s1_theta'])*params['s1_mag']),
                 "s2_theta": ("s2_x", lambda params: jnp.sin(params['s2_theta'])*jnp.cos(params['s2_phi'])*params['s2_mag']),
                 "s2_phi": ("s2_y", lambda params: jnp.sin(params['s2_theta'])*jnp.sin(params['s2_phi'])*params['s2_mag']),
                 "s2_mag": ("s2_z", lambda params: jnp.cos(params['s2_theta'])*params['s2_mag']),
                 "cos_iota": ("iota",lambda params: jnp.arccos(jnp.arcsin(jnp.sin(params['cos_iota']/2*jnp.pi))*2/jnp.pi)),
                 "sin_dec": ("dec",lambda params: jnp.arcsin(jnp.arcsin(jnp.sin(params['sin_dec']/2*jnp.pi))*2/jnp.pi))} # sin and arcsin are periodize cos_iota and sin_dec
)
true_param_list = jnp.array([Mc, args.m2/args.m1, args.s1_theta, args.s1_phi, args.s1_mag, args.s2_theta, args.s2_phi, args.s2_mag, args.dist_mpc, args.tc, args.phic, args.inclination, args.polarization_angle, args.ra, args.dec])
true_param = prior.add_name(true_param_list, transform_name = True, transform_value = True)
detector_param = {"ra": args.ra, "dec": args.dec, "gmst": gmst, "psi": args.polarization_angle, "epoch": epoch, "t_c": args.tc}
h_sky = waveform(freqs, true_param)
key, subkey = jax.random.split(jax.random.PRNGKey(args.seed+1234))
H1.inject_signal(subkey, freqs, h_sky, detector_param, "../src/jimgw/detector_data/H1.txt")
key, subkey = jax.random.split(key)
L1.inject_signal(subkey, freqs, h_sky, detector_param, "../src/jimgw/detector_data/L1.txt")
key, subkey = jax.random.split(key)
V1.inject_signal(subkey, freqs, h_sky, detector_param, "../src/jimgw/detector_data/V1.txt")

likelihood = TransientLikelihoodFD([H1, L1], waveform, trigger_time, args.duration, post_trigger_duration)
# likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], prior=prior, bounds=[prior.xmin, prior.xmax],  waveform = waveform, trigger_time = trigger_time, duration = args.duration, post_trigger_duration = post_trigger_duration)

mass_matrix = jnp.eye(args.n_dim)
#mass_matrix = mass_matrix.at[1,1].set(6e-4)
#mass_matrix = mass_matrix.at[9,9].set(6e-4)
#mass_matrix = mass_matrix.at[8,8].set(1)
#local_sampler_arg = {"step_size": mass_matrix*2e-3}

mass_matrix = mass_matrix.at[1,1].set(1e-3)
mass_matrix = mass_matrix.at[4,4].set(1e-3)
mass_matrix = mass_matrix.at[7,7].set(1e-3)
mass_matrix = mass_matrix.at[9,9].set(1e-3)
mass_matrix = mass_matrix.at[8,8].set(5)
local_sampler_arg = {"step_size": mass_matrix*6e-3}


jim = Jim(likelihood, 
        prior,
        n_loop_training=args.n_loop_training,
        n_loop_production = args.n_loop_production,
        n_local_steps=args.n_local_steps,
        n_global_steps=args.n_global_steps,
        n_chains=args.n_chains,
        n_epochs=args.num_epochs,
        learning_rate = args.learning_rate,
        max_samples = args.max_samples,
        momentum = args.momentum,
        batch_size = args.batch_size,
        use_global=args.use_global,
        keep_quantile= args.keep_quantile,
        train_thinning = args.train_thinning,
        output_thinning = args.output_thinning,
        local_sampler_arg = local_sampler_arg,
        seed = args.seed,
        num_layers = args.num_layers,
        hidden_size = args.hidden_size,
        num_bins = args.num_bins
        )
jim.maximize_likelihood([prior.xmin, prior.xmax])
key, subkey = jax.random.split(key)
jim.sample(subkey)
samples = jim.get_samples()
jim.print_summary()

chains, log_prob, local_accs, global_accs, loss_vals= jim.Sampler.get_sampler_state(training=True).values()
jnp.savez( args.output_path + '.npz', 
          chains=chains, 
          log_prob=log_prob, 
          local_accs=local_accs, 
          global_accs=global_accs,
          loss_vals = loss_vals,
          true_param=true_param_list)