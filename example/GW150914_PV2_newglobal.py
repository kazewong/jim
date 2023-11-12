import time
from jimgw.jim import Jim
from jimgw.detector import H1, L1
from jimgw.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.waveform import RippleIMRPhenomD, RippleIMRPhenomPv2
from jimgw.prior import Uniform, Unconstrained_Uniform, Composite, Sphere
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

Mc_prior = Unconstrained_Uniform(10., 80., naming=["M_c"])
q_prior = Unconstrained_Uniform(0.125, 1., naming=["q"], transforms={"q": ("eta", lambda params: params['q']/(1+params['q'])**2)})
s1_prior = Sphere("s1")
s2_prior = Sphere("s2")
dL_prior = Unconstrained_Uniform(0., 2000., naming=["d_L"])
t_c_prior = Unconstrained_Uniform(-0.05, 0.05, naming=["t_c"])
phase_c_prior = Unconstrained_Uniform(0., 2*jnp.pi, naming=["phase_c"])
cos_iota_prior = Unconstrained_Uniform(-1., 1., naming=["cos_iota"], transforms={"cos_iota": ("iota",lambda params: jnp.arccos(jnp.arcsin(jnp.sin(params['cos_iota']/2*jnp.pi))*2/jnp.pi))})
psi_prior = Unconstrained_Uniform(0., jnp.pi, naming=["psi"])
ra_prior = Unconstrained_Uniform(0., 2*jnp.pi, naming=["ra"])
sin_dec_prior = Unconstrained_Uniform(-1., 1., naming=["sin_dec"], transforms={"sin_dec": ("dec",lambda params: jnp.arcsin(jnp.arcsin(jnp.sin(params['sin_dec']/2*jnp.pi))*2/jnp.pi))})

prior = Composite([Mc_prior, q_prior, s1_prior, s2_prior, dL_prior, t_c_prior, phase_c_prior, cos_iota_prior, psi_prior, ra_prior, sin_dec_prior])

likelihood = TransientLikelihoodFD([H1, L1], waveform=waveform, trigger_time=gps, duration=4, post_trigger_duration=2)


mass_matrix = jnp.eye(prior.n_dim)
# mass_matrix = mass_matrix.at[1, 1].set(1e-3)
# mass_matrix = mass_matrix.at[9, 9].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 3e-3}


jim = Jim(
    likelihood,
    prior,
    n_loop_training=50,
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

# jim.maximize_likelihood([prior.xmin, prior.xmax])
# initial_guess = jnp.array(jnp.load('initial.npz')['chain'])
jim.sample(jax.random.PRNGKey(42))
