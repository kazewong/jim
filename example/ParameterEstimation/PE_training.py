import numpy as np
from flowMC.nfmodel.realNVP import RealNVP
import jax
import optax
import flax

from flowMC.nfmodel.utils import *
from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

from tqdm import tqdm

# data = np.load('./data/injection_posterior2.npz')
# chains = data['chains']
# log_prob = data['log_prob']
# data = jnp.array(chains).reshape(-1,9)
# data = data[::1000]

from sklearn.datasets import make_moons

data = make_moons(n_samples=10000, noise=0.05)[0]

n_dim = 2
num_epochs = 5000
batch_size = 10000
learning_rate = 0.01
momentum = 0.9
n_layers = 10
n_hidden = 100
dt = 1 / n_layers

model = RealNVP(10, n_dim, 64, 1)

key1, rng, init_rng = jax.random.split(jax.random.PRNGKey(0),3)

def create_train_state(rng, learning_rate, momentum):
    params = model.init(rng, jnp.ones((1,n_dim)))['params']
    tx = optax.adam(learning_rate, momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

state = create_train_state(init_rng, learning_rate, momentum)


variables = model.init(rng, jnp.ones((1,n_dim)))['variables']


rng, state, loss_values = train_flow(rng, model, state, data, num_epochs, batch_size, variables)
samples = sample_nf(model,state.params, rng,10000,variables)[1][0]

# from flowMC.nfmodel.utils import train_step

# @jax.jit
# def eval_step(params, batch):
#     log_det = model.apply({'params': params,'variables': variables}, batch, method=model.log_prob)
#     return -jnp.mean(log_det)

# def train_epoch(state, train_ds, batch_size, epoch, rng):
#   """Train for a single epoch."""
#   train_ds_size = len(train_ds)
#   steps_per_epoch = train_ds_size // batch_size

#   perms = jax.random.permutation(rng, train_ds_size)
#   perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
#   perms = perms.reshape((steps_per_epoch, batch_size))
#   for perm in perms:
#     batch = train_ds[perm, ...]
#     value, state = train_step(model, batch, state, variables)

#   return state

# for epoch in tqdm(range(1, num_epochs+1),desc='Training',miniters=int(num_epochs/10)):

#     # Use a separate PRNG key to permute image data during shuffling
#     rng, input_rng = jax.random.split(rng)
#     # Run an optimization step over a training batch
#     state = train_epoch(state, data, batch_size, epoch, input_rng)
#     if epoch % int(num_epochs/10) == 0:
#         print('Epoch %d' % epoch, end=' ')
#         print('Loss: %.3f' % eval_step(state.params, data))
