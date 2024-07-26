import jax
import jax.numpy as jnp
from jimgw.prior import PowerLawPrior

alpha = -1.0
xmin = 1.0
xmax = 10.0


q_samples = jax.random.uniform(jax.random.PRNGKey(42), (100,), minval=0.0, maxval=1.0)
if alpha == -1:
    samples = xmin * jnp.exp(q_samples * jnp.log(xmax / xmin))
else:
    samples = (
        xmin ** (1.0 + alpha)
        + q_samples * (xmax ** (1.0 + alpha) - xmin ** (1.0 + alpha))
    ) ** (1.0 / (1.0 + alpha))
samples[None]

PowerLawPrior(xmin, xmax, alpha, ["x"]).sample(jax.random.PRNGKey(42), 100)


##############
if alpha == -1:
    normalization = float(1.0 / jnp.log(xmax / xmin))
else:
    normalization = (1 + alpha) / (xmax ** (1 + alpha) - xmin ** (1 + alpha))
variable = 1.5
log_in_range = jnp.where(
    (variable >= xmax) | (variable <= xmin),
    jnp.zeros_like(variable) - jnp.inf,
    jnp.zeros_like(variable),
)
log_p = alpha * jnp.log(variable) + jnp.log(normalization)
log_p + log_in_range

PowerLawPrior(xmin, xmax, alpha, ["x"]).log_prob({"x": variable})


# transform_func = lambda x: (
#     xmin ** (1.0 + alpha) + x * (xmax ** (1.0 + alpha) - xmin ** (1.0 + alpha))
# ) ** (1.0 / (1.0 + alpha))
# input_params = variable
# assert_rank(input_params, 0)
# output_params = transform_func(input_params)
# jacobian = jax.jacfwd(transform_func)(input_params)
# x[variable] = output_params
# return x, jnp.log(jacobian)


import numpy as np

alpha = -2.0
xmin = 1.0
xmax = 20.0
p = PowerLawPrior(xmin, xmax, alpha, ["x"])
grid = np.linspace(xmin, xmax, 20)
transform = []
log_prob = []
for y in grid:
    transform.append(p.transform({"x": y})["x"].item())
    log_prob.append(np.exp(p.log_prob({"x": y}).item()))
import matplotlib.pyplot as plt

plt.plot(grid, transform)
plt.savefig("transform.png")
plt.close()
plt.plot(transform, log_prob)
plt.savefig("log_prob.png")
plt.close()
