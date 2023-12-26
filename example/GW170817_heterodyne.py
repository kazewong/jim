from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform, Powerlaw, Composite 
import jax.numpy as jnp
import jax
import time
jax.config.update("jax_enable_x64", True)


### Fetching the data

total_time_start = time.time()
gps = 1187008882.43
trigger_time = gps
fmin = 20
fmax = 2048
minimum_frequency = fmin
maximum_frequency = fmax
T = 128
duration = T
post_trigger_duration = 2
epoch = duration - post_trigger_duration
f_ref = fmin 

### Getting ifos and overwriting with above data

tukey_alpha = 2 / (duration / 2)
H1.load_data(gps, duration, 2, fmin, fmax, psd_pad=duration+16, tukey_alpha=tukey_alpha)
L1.load_data(gps, duration, 2, fmin, fmax, psd_pad=duration+16, tukey_alpha=tukey_alpha)
V1.load_data(gps, duration, 2, fmin, fmax, psd_pad=duration+16, tukey_alpha=tukey_alpha)

### Define priors

# Internal parameters
Mc_prior = Uniform(1.18, 1.21, naming=["M_c"])
q_prior = Uniform(
    0.125,
    1.0,
    naming=["q"],
    transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
)
s1z_prior                = Uniform(-0.05, 0.05, naming=["s1_z"])
s2z_prior                = Uniform(-0.05, 0.05, naming=["s2_z"])

# External parameters
dL_prior       = Powerlaw(1.0, 75.0, 2.0, naming=["d_L"])
t_c_prior      = Uniform(-0.1, 0.1, naming=["t_c"])
phase_c_prior  = Uniform(0.0, 2 * jnp.pi, naming=["phase_c"])
cos_iota_prior = Uniform(
    -1.0,
    1.0,
    naming=["cos_iota"],
    transforms={
        "cos_iota": (
            "iota",
            lambda params: jnp.arccos(
                jnp.arcsin(jnp.sin(params["cos_iota"] / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        )
    },
)
psi_prior     = Uniform(0.0, jnp.pi, naming=["psi"])
ra_prior      = Uniform(0.0, 2 * jnp.pi, naming=["ra"])
sin_dec_prior = Uniform(
    -1.0,
    1.0,
    naming=["sin_dec"],
    transforms={
        "sin_dec": (
            "dec",
            lambda params: jnp.arcsin(
                jnp.arcsin(jnp.sin(params["sin_dec"] / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        )
    },
)

prior = Composite([
        Mc_prior,
        q_prior,
        s1z_prior,
        s2z_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        cos_iota_prior,
        psi_prior,
        ra_prior,
        sin_dec_prior,
    ]
)

# The following only works if every prior has xmin and xmax property, which is OK for Uniform and Powerlaw
bounds = jnp.array([[p.xmin, p.xmax] for p in prior.priors])

### Create likelihood object
likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], prior=prior, bounds=bounds, waveform=RippleIMRPhenomD(), trigger_time=gps, duration=T, n_bins=500)

### Create sampler and jim objects
eps = 3e-2
n_dim = 11
mass_matrix = jnp.eye(n_dim)
mass_matrix = mass_matrix.at[0,0].set(1e-5)
mass_matrix = mass_matrix.at[1,1].set(1e-4)
mass_matrix = mass_matrix.at[2,2].set(1e-3)
mass_matrix = mass_matrix.at[3,3].set(1e-3)
mass_matrix = mass_matrix.at[5,5].set(1e-5)
mass_matrix = mass_matrix.at[9,9].set(1e-2)
mass_matrix = mass_matrix.at[10,10].set(1e-2)
local_sampler_arg = {"step_size": mass_matrix * eps}

outdir_name = "./outdir/"

jim = Jim(
    likelihood,
    prior,
    n_loop_training=200,
    n_loop_production=20,
    n_local_steps=200,
    n_global_steps=200,
    n_chains=1000,
    n_epochs=300,
    learning_rate=0.001,
    max_samples=50000,
    momentum=0.9,
    batch_size=50000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=10,
    output_thinning=30,    
    n_loops_maximize_likelihood = 2000,
    local_sampler_arg=local_sampler_arg,
    outdir_name=outdir_name
)

jim.sample(jax.random.PRNGKey(42))
jim.print_summary()