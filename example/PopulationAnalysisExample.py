# Import some necessary modules
import numpy as np
import jax
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.realNVP import RealNVP
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline

from jimgw.population_distribution import PosteriorSampleData, PowerLawModel, PopulationDistribution
from matplotlib import pyplot as plt

########################## Population model setup ##########################
data = PosteriorSampleData('data/')
pop_model = PowerLawModel()
pop_distribution = PopulationDistribution(model=PowerLawModel(), data=data)


########################## Hyperparameters to change ##########################
n_dim = 4
n_chains = 50
param_initial_guess = [0.61, 0.92, 6.0, 60.0]

n_layer = 10  # number of coupling layers
n_hidden = 128  # with of hidden layers in MLPs parametrizing coupling layers

step_size = 2e-1
n_local_steps = 100
n_global_steps = 100
num_epochs = 10
learning_rate = 0.001
batch_size = 5000

###############################################################################


rng_key_set = initialize_rng_keys(n_chains, seed=42)
initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
for i, param in enumerate(param_initial_guess):
    initial_position = initial_position.at[:, i].add(param)

model = MaskedCouplingRQSpline(n_dim, n_layer, [n_hidden, n_hidden], 8, jax.random.PRNGKey(21))

local_sampler = MALA(pop_distribution.evaluate, True, {"step_size": step_size})
local_sampler_caller = lambda x: MALA_Sampler.make_sampler()

nf_sampler = Sampler(n_dim,
                    rng_key_set,
                    None,
                    local_sampler,
                    model,
                    n_local_steps = n_local_steps,
                    n_global_steps = n_global_steps,
                    n_epochs = num_epochs,
                    learning_rate = learning_rate,
                    batch_size = batch_size,
                    n_chains = n_chains)

nf_sampler.sample(initial_position, data=None)


out_train = nf_sampler.get_sampler_state(training=True)
print('Logged during tuning:', out_train.keys())

import corner
import matplotlib.pyplot as plt
chains = np.array(out_train['chains'])
global_accs = np.array(out_train['global_accs'])
local_accs = np.array(out_train['local_accs'])
loss_vals = np.array(out_train['loss_vals'])


# Plot 2 chains in the plane of 2 coordinates for first visual check 
plt.figure(figsize=(6, 6))
axs = [plt.subplot(2, 2, i + 1) for i in range(4)]
plt.sca(axs[0])
plt.title("2d proj of 2 chains")

plt.plot(chains[0, :, 0], chains[0, :, 1], 'o-', alpha=0.5, ms=2)
plt.plot(chains[1, :, 0], chains[1, :, 1], 'o-', alpha=0.5, ms=2)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.sca(axs[1])
plt.title("NF loss")
plt.plot(loss_vals.reshape(-1))
plt.xlabel("iteration")

plt.sca(axs[2])
plt.title("Local Acceptance")
plt.plot(local_accs.mean(0))
plt.xlabel("iteration")

plt.sca(axs[3])
plt.title("Global Acceptance")
plt.plot(global_accs.mean(0))
plt.xlabel("iteration")
plt.tight_layout()
plt.show(block=False)
plt.savefig('figure1.png')

labels=["$alpha$", "$beta$", "$m_min$", "$m_max$"]
# Plot all chains
figure = corner.corner(
    chains.reshape(-1, n_dim), labels=labels, quantiles=(0.16, 0.5, 0.84), show_titles=True, title_fmt = '.2f', use_math_text=True, color='MediumPurple'
)
figure.set_size_inches(7, 7)
figure.suptitle("Visualize samples")
plt.savefig('figure2.png')
plt.show(block=False)



