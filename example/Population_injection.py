import jax.numpy as jnp
import numpy as np
import copy
import bilby
import jax
from jax import random, grad, jit, vmap, jacfwd, jacrev, value_and_grad
from jax.experimental.optimizers import adam
from jax.config import config
config.update("jax_enable_x64", True)

from jaxgw.gw.likelihood.utils import inner_product
from jaxgw.gw.waveform.IMRPhenomC import IMRPhenomC
from jaxgw.gw.likelihood.detector_preset import get_H1,get_L1
from jaxgw.gw.likelihood.detector_projection import get_detector_response


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

true_ld = 600.
true_phase = 0.
true_gt = 0.

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(['H1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=2048, duration=128,
    start_time=- 3)

psd = ifos[0].power_spectral_density_array
psd_frequency = ifos[0].frequency_array
psd_frequency = psd_frequency[jnp.isfinite(psd)]
psd = psd[jnp.isfinite(psd)]

H1, H1_vertex = get_H1()
L1, L1_vertex = get_L1()

def gen_params(m1):
	params = dict(mass_1=m1, mass_2=m1, spin_1=0., spin_2=0., luminosity_distance=true_ld, phase_c=true_phase, t_c=true_gt, theta_jn=0.4, psi=2.659, ra=1.375, dec=-1.2108)
	return params

def gen_event(params,detector, detector_vertex):	
	waveform = IMRPhenomC(psd_frequency, params)
	waveform = get_detector_response(psd_frequency, waveform, params, detector, detector_vertex)
	return waveform

@jit
def single_detector_likelihood(params, data, data_f, PSD, detector, detector_vertex):
	waveform = IMRPhenomC(data_f, params)
	waveform = get_detector_response(data_f, waveform, params, detector, detector_vertex)
	match_filter_SNR = inner_product(waveform, data, data_f, PSD)
	optimal_SNR = inner_product(waveform, waveform, data_f, PSD)
	return (-2*match_filter_SNR + optimal_SNR)/2

@jit
def log_prob(params, strain_H1, strain_L1):
	return single_detector_likelihood(params, strain_H1, psd_frequency, psd, H1, H1_vertex)+single_detector_likelihood(params, strain_L1, psd_frequency, psd, L1, L1_vertex)

@jit
def log_prob_scan(args,index):
	result = log_prob(args[0][index],args[1][index],args[2][index])
	return args, result

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

N_sample = 500
obs_std = 0.1

m1_sample = jnp.empty(0)


while m1_sample.shape[0]<N_sample:
	key, *subkeys = random.split(key,num=3)
	m1_sample_ = random.uniform(subkeys[0],shape=(N_sample*100,))*100+2
	p_m1 = power_law_plus_peak(m1_sample_,true_param)
	rand_ = random.uniform(subkeys[1],shape=(N_sample*100,))*power_law_plus_peak(jnp.array([true_param['xmin']]),true_param)[0]
	m1_sample = jnp.append(m1_sample,m1_sample_[rand_<p_m1])

m1_sample = m1_sample[:N_sample]

H1_data_array = vmap(gen_event,(0, None, None))(vmap(gen_params)(m1_sample), H1, H1_vertex)
L1_data_array = vmap(gen_event,(0, None, None))(vmap(gen_params)(m1_sample), L1, L1_vertex)
multi_event_gen_param = jit(vmap(gen_params))
multi_event_gen_event = jit(vmap(gen_event))
multi_detector_likelihood = jit(vmap(log_prob,(0,0,0),0))

def single_population_likelihood(event_params, pop_params, strain_H1, strain_L1):
	event_likelihood = log_prob(event_params, strain_H1, strain_L1)
	return log_prob(event_params, strain_H1, strain_L1) + jnp.log(power_law_plus_peak(jnp.array([event_params['mass_1']]),pop_params)[0])


single_population_likelihood_ = vmap(single_population_likelihood,(0,None,0,0),0)

def population_likelihood_powerlaw_peak(point,params):
#	_, single_event_likelihood = jax.lax.scan(log_prob_scan,[point,H1_data_array,L1_data_array],jnp.arange(N_sample))
	return -jnp.sum(single_population_likelihood_(point,params,H1_data_array,L1_data_array))



guess_param = {}
guess_param['alpha'] = 2.63
guess_param['xmin'] = 4.59
guess_param['xmax'] = 86.22
guess_param['mean'] = 33.07
guess_param['sigma'] = 5.69
guess_param['mixing'] = 0.3

event_fisher  = jacfwd(jacrev(single_population_likelihood))
event_hyper_fisher  = jacfwd(jacrev(single_population_likelihood,argnums=1),argnums=0)
hyper_fisher  = jacfwd(jacrev(population_likelihood_powerlaw_peak,argnums=1),argnums=1)
#dlambdadtheta_plpk = jacfwd(jacrev(population_likelihood_powerlaw_peak,argnums=0),argnums=1)(best_x_plpk,best_lambda_plpk)

# Loop over event_fisher and event_hyper_fisher over m1_sample

event_fisher_array = []
event_hyper_fisher_array = []

for i in range(N_sample):
	print(i)
	event_fisher_array.append(event_fisher(gen_params(m1_sample[i]),guess_param,H1_data_array[i],L1_data_array[i]))
	event_hyper_fisher_array.append(event_hyper_fisher(gen_params(m1_sample[i]),guess_param,H1_data_array[i],L1_data_array[i]))



#fig,ax = plt.subplots(1,3,figsize=(30,9))
#ax[0].hist(m1_sample,bins=50,density=True,histtype='step',lw=3,label='Truth',color='C2')
#axis = jnp.linspace(ax[0].get_xlim()[0],ax[0].get_xlim()[1],1000)
#ax[0].plot(axis,power_law_plus_peak(axis,best_lambda_pl),label='Power law',c='C0')
#ax[0].plot(axis,power_law_plus_peak(axis,best_lambda_plpk),label='Power law + peak',c='C1')
#ax[0].set_ylabel(r'$p(x)$')
#ax[0].set_xlabel(r'$x$')
#ax[0].legend(loc='upper right',fontsize=20)
#ax[1].plot(dlambdadtheta_pl['alpha'][jnp.argsort(dlambdadtheta_pl['alpha'])],label='Power law sorted',lw=3)
#ax[1].plot(dlambdadtheta_plpk['alpha'][jnp.argsort(dlambdadtheta_plpk['alpha'])],label='Power law + peak sorted',lw=3)
#ax[1].legend(loc='lower right',fontsize=20)
#ax[1].set_xlabel('Event number')
#ax[1].set_ylabel(r'$\frac{\partial^2\mathcal{L}}{\partial \theta \partial \alpha}$')
#ax[2].plot(dlambdadtheta_pl['xmin'][jnp.argsort(dlambdadtheta_pl['xmin'])],label='Power law sorted',lw=3)
#ax[2].plot(dlambdadtheta_plpk['xmin'][jnp.argsort(dlambdadtheta_plpk['xmin'])],label='Power law + peak sorted',lw=3)
#ax[2].legend(loc='lower right',fontsize=20)
#ax[2].set_xlabel('Event number')
#ax[2].set_ylabel(r'$\frac{\partial^2\mathcal{L}	}{\partial \theta \partial x_{min}}$')
#
#fig.show()
