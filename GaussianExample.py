import numpy as np
import jax.numpy as jnp
from jax import random, grad, jit, vmap, value_and_grad, jacfwd, jacrev, hessian
from jax.experimental.optimizers import adam
import matplotlib.pyplot as plt
import matplotlib as mpl
params = {'axes.labelsize': 32,
          'font.family': 'serif',
          'font.serif': 'Computer Modern Raman',
          'font.size': 32,
          'axes.linewidth': 2,
          'legend.fontsize': 28,
          'xtick.labelsize': 28,
          'xtick.top': True,
          'xtick.direction': "in",
          'ytick.labelsize': 20,
          'ytick.right': True,
          'ytick.direction': "in",
          'axes.grid' : False,
          'text.usetex': True,
          'savefig.dpi' : 100,
          'lines.markersize' : 14,
#          'axes.formatter.useoffset': False,
          'axes.formatter.limits' : (-3,3)}

mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

mpl.rcParams.update(params)

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
	return -jnp.sum(jnp.log(gaussian(data,point[:,None],obs_std)*gaussian(point[:,None],params[0],params[1])))


N_obs = 1000
N_subpop = 100
true_param = jnp.array([0.,1])
obs_std = 0.1
true_data = (random.normal(sub_keys[0],shape=(N_obs,))*true_param[1]+true_param[0])
true_data = jnp.append(true_data,(random.normal(sub_keys[1],shape=(N_subpop,))*0.1))
obs_data = true_data[:,None]+random.normal(sub_keys[2],shape=(N_obs+N_subpop,100))*obs_std

index = np.random.choice(np.arange(N_obs+N_subpop),replace=False,size=N_obs+N_subpop)
obs_data = obs_data[index]
true_data = true_data[index]

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



fig,ax = plt.subplots(1,3,figsize=(30,9))
ax[0].hist(true_data,bins=50,density=True,histtype='step',lw=3,label='Truth')
axis = np.linspace(ax[0].get_xlim()[0],ax[0].get_xlim()[1],1000)
ax[0].plot(axis,gaussian(axis,best_lambda[0],best_lambda[1]),label='Best fitted')
ax[0].set_ylabel(r'$p(x)$')
ax[0].set_xlabel(r'$x$')
ax[0].legend(loc='upper right')
ax[1].plot(dlambdadtheta[0],label='Raw')
ax[1].plot(dlambdadtheta[0][np.argsort(dlambdadtheta[0])],label='sorted',lw=5)
ax[1].legend(loc='upper left')
ax[1].set_xlabel('Event number')
ax[1].set_ylabel(r'$\frac{\partial^2\mathcal{L}}{\partial \theta \partial \mu}$')
ax[2].plot(dlambdadtheta[1])
ax[2].plot(dlambdadtheta[1][np.argsort(dlambdadtheta[1])],lw=5)
ax[2].set_xlabel('Event number')
ax[2].set_ylabel(r'$\frac{\partial^2\mathcal{L}}{\partial \theta \partial \sigma}$')
fig.show()
