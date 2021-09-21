import jax.numpy as jnp
import copy
from jax import random, grad, jit, vmap, jacfwd, jacrev, value_and_grad
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




key = random.PRNGKey(42)

########################################
# Defining our model
########################################

def truncated_power_law(x,alpha,xmin,xmax):
	norm = (xmax**(1-alpha)-xmin**(1-alpha))/(1-alpha)
	output = (x**-alpha)/norm
	output = index_update(output,((x<xmin)+(x>xmax)),0)
	return output


# Since truncated power law is not differentiable, we choose tanh as a smoother cutoff
x_axis = jnp.linspace(1,150,100000)
@jit
def power_law_tanh(x,params):
	alpha = params['alpha']
	xmin = params['xmin']
	xmax = params['xmax']
	lower_window = (jnp.tanh((x-xmin)*10)+1)/2
	upper_window = -(jnp.tanh((x-xmax)*10)-1)/2
	power_law = x**-alpha
	output_unnorm = power_law*lower_window*upper_window
	# This normalization factor is supposed to be a good approximation but not perfect
	norm = jnp.trapz(x_axis**-alpha*(jnp.tanh(x_axis-xmin)+1)/2*(-(jnp.tanh(x_axis-xmax)-1)/2),x=x_axis)
	output = output_unnorm/norm
	return output

@jit
def gaussian(x,mean,sigma):
    return (1./jnp.sqrt(2*jnp.pi)/sigma)*jnp.exp(-(((x-mean)/sigma)**2)/2)

@jit
def power_law_plus_peak(x,params):
# !!! Add smoothing later
# Since each component is normalized, the combine pdf should be normalized
	powerlaw = power_law_tanh(x,params)
	peak = gaussian(x,params['mean'],params['sigma'])
	combine = (1-params['mixing'])*powerlaw+params['mixing']*peak
	return combine

@jit
def population_likelihood_powerlaw(point,params,obs_std,data):
	return -jnp.sum(jnp.log(gaussian(data,point[:,None],obs_std)*power_law_tanh(point[:,None],params)))

def population_likelihood_powerlaw_peak(point,params,obs_std,data):
	if params['mixing'] < 0:
		params['mixing'] = 0.
	return -jnp.sum(jnp.log(gaussian(data,point[:,None],obs_std)*power_law_plus_peak(point[:,None],params)))

########################################
# Power law Only
########################################

true_param = {}
true_param['alpha'] = 2.63
true_param['xmin'] = 4.59
true_param['xmax'] = 86.22
true_param['mean'] = 33.07
true_param['sigma'] = 5.69
true_param['mixing'] = 0.3

N_sample = 1000
obs_std = 0.1

m1_sample = jnp.empty(0)


while m1_sample.shape[0]<N_sample:
	key, *subkeys = random.split(key,num=3)
	m1_sample_ = random.uniform(subkeys[0],shape=(N_sample*100,))*100+2
	p_m1 = power_law_plus_peak(m1_sample_,true_param)
	rand_ = random.uniform(subkeys[1],shape=(N_sample*100,))*power_law_plus_peak(jnp.array([true_param['xmin']]),true_param)[0]
	m1_sample = jnp.append(m1_sample,m1_sample_[rand_<p_m1])

m1_sample = m1_sample[:N_sample]

key, *subkeys = random.split(key,num=2)
obs_m1 = m1_sample[:,None] + random.normal(subkeys[0],shape=(N_sample,100))*obs_std


guess_param = {}
guess_param['alpha'] = 2.2
guess_param['xmin'] = 1.
guess_param['xmax'] = 90.
guess_param['mean'] = 35.
guess_param['sigma'] = 5.9
guess_param['mixing'] = 0.

learning_rate = 1e-1
opt_init, opt_update, get_params = adam(learning_rate)
opt_state = opt_init((m1_sample,guess_param))

def step(step, opt_state):
	params = get_params(opt_state)
	value, grads = value_and_grad(population_likelihood_powerlaw,argnums=(0,1))(params[0],params[1], obs_std, obs_m1)
	opt_state = opt_update(step, grads, opt_state)
	return value, opt_state

for i in range(500):
	value, opt_state = step(i, opt_state)
	if jnp.isnan(value):
		break
	print(value,get_params(opt_state)[1])

best_x_pl, best_lambda_pl = get_params(opt_state)

dlambdadtheta_pl = jacfwd(jacrev(population_likelihood_powerlaw),argnums=1)(best_x_pl,best_lambda_pl,obs_std,obs_m1)

learning_rate = 1e-2
opt_init, opt_update, get_params = adam(learning_rate)
opt_state = opt_init((m1_sample,guess_param))

def step(step, opt_state):
	params = get_params(opt_state)
	value, grads = value_and_grad(population_likelihood_powerlaw_peak,argnums=(0,1))(params[0], params[1], obs_std, obs_m1)
	opt_state = opt_update(step, grads, opt_state)
	return value, opt_state

for i in range(500):
	value, opt_state = step(i, opt_state)
	if jnp.isnan(value):
		break
	print(value,get_params(opt_state)[1])

best_x_plpk, best_lambda_plpk = get_params(opt_state)

dlambdadtheta_plpk = jacfwd(jacrev(population_likelihood_powerlaw_peak),argnums=1)(best_x_plpk,best_lambda_plpk,obs_std,obs_m1)


fig,ax = plt.subplots(1,3,figsize=(30,9))
ax[0].hist(m1_sample,bins=50,density=True,histtype='step',lw=3,label='Truth',color='C2')
axis = jnp.linspace(ax[0].get_xlim()[0],ax[0].get_xlim()[1],1000)
ax[0].plot(axis,power_law_plus_peak(axis,best_lambda_pl),label='Power law',c='C0')
ax[0].plot(axis,power_law_plus_peak(axis,best_lambda_plpk),label='Power law + peak',c='C1')
ax[0].set_ylabel(r'$p(x)$')
ax[0].set_xlabel(r'$x$')
ax[0].legend(loc='upper right',fontsize=20)
ax[1].plot(dlambdadtheta_pl['alpha'][jnp.argsort(dlambdadtheta_pl['alpha'])],label='Power law sorted',lw=3)
ax[1].plot(dlambdadtheta_plpk['alpha'][jnp.argsort(dlambdadtheta_plpk['alpha'])],label='Power law + peak sorted',lw=3)
ax[1].legend(loc='lower right',fontsize=20)
ax[1].set_xlabel('Event number')
ax[1].set_ylabel(r'$\frac{\partial^2\mathcal{L}}{\partial \theta \partial \alpha}$')
ax[2].plot(dlambdadtheta_pl['xmin'][jnp.argsort(dlambdadtheta_pl['xmin'])],label='Power law sorted',lw=3)
ax[2].plot(dlambdadtheta_plpk['xmin'][jnp.argsort(dlambdadtheta_plpk['xmin'])],label='Power law + peak sorted',lw=3)
ax[2].legend(loc='lower right',fontsize=20)
ax[2].set_xlabel('Event number')
ax[2].set_ylabel(r'$\frac{\partial^2\mathcal{L}}{\partial \theta \partial x_{min}}$')

fig.show()
