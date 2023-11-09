import time
from jimgw.jim import Jim
from jimgw.detector import H1, L1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD, RippleIMRPhenomPv2
from jimgw.prior import Uniform
import jax.numpy as jnp
import jax


jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
start = gps - 2
end = gps + 2
fmin = 20.0
fmax = 1024.0

ifos = ["H1", "L1"]

H1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
L1.load_data(gps, 2, 2, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

waveform = RippleIMRPhenomPv2(f_ref=20)
prior = Uniform(
    xmin = [10, 0.125, 0, 0, 0, 0, 0, 0, 0., -0.05, 0., -1, 0., 0.,-1.],
    xmax = [80., 1., jnp.pi, 2*jnp.pi, 1., jnp.pi, 2*jnp.pi, 1., 2000., 0.05, 2*jnp.pi, 1., jnp.pi, 2*jnp.pi, 1.],
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
likelihood = TransientLikelihoodFD([H1, L1], waveform=waveform, trigger_time=gps, duration=4, post_trigger_duration=2)


mass_matrix = jnp.eye(prior.n_dim)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[9, 9].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 3e-3}


jim = Jim(
    likelihood,
    prior,
    n_loop_training=10,
    n_loop_production=10,
    n_local_steps=300,
    n_global_steps=300,
    n_chains=500,
    n_epochs=300,
    learning_rate=0.001,
    max_samples = 60000,
    momentum=0.9,
    batch_size=30000,
    use_global=True,
    keep_quantile=0.,
    train_thinning=1,
    output_thinning=30,
    local_sampler_arg=local_sampler_arg,
    num_layers = 6,
    hidden_size = [32,32],
    num_bins = 8
)

jim.maximize_likelihood([prior.xmin, prior.xmax])
# initial_guess = jnp.array(jnp.load('initial.npz')['chain'])
jim.sample(jax.random.PRNGKey(42))
