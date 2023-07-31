from flowMC.nfmodel.realNVP import RealNVP
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
import jax
import jax.numpy as jnp  # JAX NumPy

from flowMC.nfmodel.utils import *
from flowMC.nfmodel.common import Gaussian
import equinox as eqx
import optax  # Optimizers

from jimgw.population_distribution import PosteriorSampleData, PowerLawModel, PopulationDistribution

import corner
import numpy as np
import matplotlib.pyplot as plt

data_obj = PosteriorSampleData('data/')
data = data_obj.get_posterior_samples(["mass_1_source", "mass_ratio", "redshift", "chi_eff"])

# We need to reformat data from each event into array of same length, or otherwise jax will recompile for each event
new_data = []
key = jax.random.PRNGKey(0)
for event in data:
    key, subkey = jax.random.split(key)
    new_data.append(jax.random.choice(subkey, jnp.stack(event).T, (30000,))) # generates a random sample from a given array
    
new_data = jnp.stack(new_data) # converting list into jnp array

"""
Training a masked RealNVP flow to fit posterior events dataset.
"""


# batch_size = 10000
# dt = 1 / n_layers
# momentum = 0.9


n_dim = 4
n_layers = 5
n_hidden = 32
learning_rate = 0.001
num_epochs = 3000

key1, rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 3)
jnp.reshape(new_data, (-1,n_dim))
# model = RealNVP(n_layers, 2, n_hidden, rng, 1., base_cov = jnp.cov(data.T), base_mean = jnp.mean(data, axis=0))
model = MaskedCouplingRQSpline(n_dim, n_layers, [n_hidden,n_hidden], 8 , rng, data_cov = jnp.cov((new_data.reshape(-1,n_dim)).T), data_mean = jnp.mean((new_data.reshape(-1,n_dim)), axis=0))



@eqx.filter_value_and_grad
def loss_fn(model, posterior_samples):
    return -jnp.sum(jnp.mean(jax.vmap(model.log_prob)(new_data), axis=1))


@eqx.filter_jit
def make_step(model, x, opt_state):
    loss, grads = loss_fn(model, x)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

optim = optax.adam(learning_rate)
opt_state = optim.init(eqx.filter(model,eqx.is_array))
for step in range(num_epochs):
    loss, model, opt_state = make_step(model, new_data, opt_state)
    loss = loss.item()
    print(f"step={step}, loss={loss}")


nf_samples = model.sample(jax.random.PRNGKey(124098),5000)

figure = corner.corner(np.array(nf_samples))
figure.set_size_inches(7, 7)
figure.suptitle("Visualize NF samples")
plt.savefig('figure_a')
plt.show()