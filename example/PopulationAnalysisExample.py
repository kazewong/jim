# Import some necessary modules
import jax
import jax.numpy as jnp
import numpy as np
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *
from jimgw.population_distribution import PosteriorSampleData
from jimgw.population_distribution import PowerLawModel
from jimgw.population_distribution import PopulationDistribution

# Parameters for flowMC
ndim = 4
nchains = 50

data = PosteriorSampleData('data/').get_all_posterior_samples()
log_posterior = lambda params, data: PopulationDistribution(model = PowerLawModel()).get_distribution(params, data)

rng_key_set = initialize_rng_keys(nchains, seed=42)
param_initial_guess = [2.5, 6.0, 4.5, 80.0]
initial_position = jax.random.normal(rng_key_set[0], shape=(nchains, ndim))
for i, param in enumerate(param_initial_guess):
    initial_position = initial_position.at[:, i].add(param)


model = MaskedCouplingRQSpline(ndim, 3, [64, 64], 8, jax.random.PRNGKey(21))
step_size = 1e-1
local_sampler = MALA(log_posterior, True, {"step_size": step_size})


nf_sampler = Sampler(ndim,
                    rng_key_set,
                    jnp.arange(ndim),
                    local_sampler,
                    model,
                    n_local_steps = 50,
                    n_global_steps = 50,
                    n_epochs = 30,
                    learning_rate = 1e-2,
                    batch_size = 1000,
                    n_chains = nchains)

nf_sampler.sample(initial_position, data)
chains,log_prob,local_accs, global_accs = nf_sampler.get_sampler_state().values()



