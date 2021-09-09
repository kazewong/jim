import numpy as np
import jax.numpy as jnp
from jax import random, grad, jit, vmap, value_and_grad, jacfwd, jacrev, hessian
from jax.experimental.optimizers import adam
import matplotlib.pyplot as plt

key, *sub_keys = random.split(random.PRNGKey(32),num=4)

@jit
def gaussian(x,mean,sigma):
	return (1./jnp.sqrt(2*jnp.pi)/sigma)*jnp.exp(-(((x-mean)/sigma)**2)/2)

@jit
def log_gaussian(x,mean,sigma):
	return jnp.log(gaussian(x,mean,sigma))


@jit
def sum_log_gaussian(x,mean,sigma):
	return jnp.sum(jnp.log(gaussian(x,mean,sigma)))

@jit
def population_likelihood(params,data):
	return -jnp.sum(jnp.log(gaussian(data,params[0],params[1])))

@jit
def population_likelihood_event(point,params,obs_std,data):
	return -jnp.sum(jnp.log(gaussian(point[:,None],data,obs_std)*gaussian(data,params[0],params[1])))


N_obs = 10000
N_subpop = 0#1000
true_param = jnp.array([0.,5])
obs_std = 0.05
true_data = (random.normal(sub_keys[0],shape=(N_obs,))*true_param[1]+true_param[0])
true_data = jnp.append(true_data,(random.normal(sub_keys[1],shape=(N_subpop,))*0.1-5))
obs_data = true_data[:,None]+random.normal(sub_keys[2],shape=(N_obs+N_subpop,100))*obs_std


learning_rate = 1e-1
opt_init, opt_update, get_params = adam(learning_rate)
opt_state = opt_init((true_data,[jnp.array(10.),jnp.array(10.)]))

def step(step, opt_state):
  params = get_params(opt_state)
  value, grads = value_and_grad(population_likelihood_event,argnums=(0,1))(params[0],params[1], obs_std, obs_data)
  opt_state = opt_update(step, grads, opt_state)
  return value, opt_state

for i in range(200):
  value, opt_state = step(i, opt_state)
  print(value,get_params(opt_state)[1])

best_x, best_lambda = get_params(opt_state)

dlambdadtheta = jacfwd(jacrev(population_likelihood_event),argnums=1)(best_x,best_lambda,obs_std,obs_data)
#dthetadlambda = jacfwd(jacrev(population_likelihood_event,argnums=1))(best_x,best_lambda,obs_std,obs_data)

fig,ax = plt.subplots(1,2,figsize=(20,9))
ax[0].plot(dlambdadtheta[0])
ax[1].plot(dlambdadtheta[1])
fig.show()
