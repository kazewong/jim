import numpy as np
import jax.numpy as jnp
from jax import random, grad, jit, vmap, value_and_grad
from jax.experimental.optimizers import adam

key, *sub_keys = random.split(random.PRNGKey(32),num=3)


@jit
def gaussian(x,mean,sigma):
    return (1./jnp.sqrt(2*jnp.pi)/sigma)*jnp.exp(-(((x-mean)/sigma)**2)/2)

@jit
def population_likelihood(params,data):
	return -jnp.sum(jnp.log(gaussian(data,params[0],params[1])))

@jit
def population_likelihood_event(point,params,obs_std,data):
	return -jnp.sum(jnp.log(gaussian(point,data,obs_std)*gaussian(data,params[0],params[1])))


N_obs = 2000
true_param = [0,1]
obs_std = 0.2
true_data = random.normal(sub_keys[0],shape=(N_obs,))*true_param[1]+true_param[0]
obs_data = true_data+random.normal(sub_keys[1],shape=(N_obs,))*obs_std
#obs_data = jnp.append(obs_data,10)


learning_rate = 1e-1
opt_init, opt_update, get_params = adam(learning_rate)
opt_state = opt_init((obs_data,[jnp.array(1.),jnp.array(2.)]))

def step(step, opt_state):
  params = get_params(opt_state)
  value, grads = value_and_grad(population_likelihood_event,argnums=(0,1))(params[0],params[1], obs_std, obs_data)
  opt_state = opt_update(step, grads, opt_state)
  return value, opt_state

for i in range(200):
  value, opt_state = step(i, opt_state)
  print(value,get_params(opt_state)[1])

#dLdlambda = grad(population_likelihood)([0.,1.],data)
#dLdtheta = grad(population_likelihood,argnums=1)([0.,1.],data)
#
#dlambdadtheta = (dLdtheta[None,:]/jnp.array(dLdlambda)[:,None]).mean(axis=0)
