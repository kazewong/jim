import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"

import jax
import jax.numpy as jnp

from jimgw.jim import Jim
from jimgw.prior import UniformPeriodicPrior
from jimgw.transforms import PeriodicTransform
from jimgw.base import LikelihoodBase
from jaxtyping import Float

jax.config.update("jax_enable_x64", True)

test_prior = UniformPeriodicPrior(0.0, 2.0 * jnp.pi, ["test"])

prior = test_prior

sample_transforms = [
    PeriodicTransform("test", 0.0, 2.0 * jnp.pi)
]

likelihood_transforms = []

class CosLikelihood(LikelihoodBase):

    def __init__(self):
        pass

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        return jnp.log((jnp.cos(params["test"]) + 1.0) / 4.0)

likelihood = CosLikelihood()

mass_matrix = jnp.eye(2)

n_local_steps=500

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    n_loop_training=1,
    n_loop_production=1,
    n_local_steps=n_local_steps,
    n_global_steps=0,
    n_chains=1,
    n_epochs=1,
    learning_rate=1e-4,
    n_max_examples=10000,
    n_flow_sample=1,
    momentum=0.9,
    batch_size=100,
    use_global=False,
    train_thinning=1,
    output_thinning=1,
    local_sampler_arg={"step_size": mass_matrix * 2e-1},
)

jim.sample(jax.random.PRNGKey(42))
jim.print_summary()
samples = jim.get_samples()

# import matplotlib.pyplot as plt

# plt.plot(samples["test"])
# plt.ylim(0.0, 2.0 * jnp.pi)
# plt.xlim(0, n_local_steps)
# plt.savefig("test_periodic_uniform.jpg")
