import numpy as np
import jax.numpy as jnp
import copy
import astropy.units as u
import h5py
from jax import random, grad, jit, vmap
from jax.ops import index_update
from astropy.cosmology import Planck15
from scipy.interpolate import interp1d

key = random.PRNGKey(42)

def truncated_power_law(x,alpha,xmin,xmax):
	norm = (xmax**(1-alpha)-xmin**(1-alpha))/(1-alpha)
	output = (x**-alpha)/norm
	output = index_update(output,((x<xmin)+(x>xmax)),-jnp.inf)
	return output

@jit
def gaussian(x,mean,sigma):
    return (1./jnp.sqrt(2*jnp.pi)/sigma)*jnp.exp(-(((x-mean)/sigma)**2)/2)

def power_law_plus_peak(x,params):
# Add smoothing later
# Since each component is normalized, the 
	powerlaw = truncated_power_law(x,params['alpha'],params['xmin'],params['xmax'])
	peak = gaussian(x,params['mean'],params['sigma'])
	combine = (1-params['mixing'])*powerlaw+params['mixing']*peak
	return combine


z_range = [0.,1]
z_axis = jnp.linspace(z_range[0],z_range[1],10000)
dVdz = jnp.array(Planck15.differential_comoving_volume(z_axis).value/1e9)

def redshift_distribution(z,kappa):
	dVdz_local = jnp.interp(z,z_axis,dVdz)
	norm_z = jnp.trapz((1+z_axis)**(kappa-1)*jnp.array(dVdz))
	return (1+z)**kappa*dVdz_local/norm_z

def combine_pdf(params,data):
	m1 = data[..., 0]
	q = data[..., 1]
	z = data[..., 2]
	p_m1 = power_law_plus_peak(m1,params)
	p_q = truncated_power_law(q,params['beta'],0.01,1)
	p_z = redshift_distribution(z,params['kappa'])
	return p_m1*p_q*p_z

def population_likelihood(params, data):
	combine_pdf = combine_pdf(params,data)
	output = jnp.sum(jnp.log(combine_pdf))
	if jnp.isfinite(output):
		return output
	else:
		return -jnp.inf

true_param = {}
true_param['alpha'] = 2.
true_param['xmin'] = 2.
true_param['xmax'] = 100.
true_param['mean'] = 30.
true_param['sigma'] = 1.
true_param['mixing'] = 0.5
true_param['beta'] = 2.
true_param['kappa'] = 0.

N_sample = 1000

key, *subkeys = random.split(key,num=4)
m1_sample = random.uniform(subkeys[0],shape=(N_sample,1))*98+2
q_sample = random.uniform(subkeys[0],shape=(N_sample,1))*0.99+0.01
z_sample = random.uniform(subkeys[0],shape=(N_sample,1))

data = jnp.concatenate((m1_sample, q_sample, z_sample), axis=1)

O12 = h5py.File('./data/injections_O1O2an_spin.h5','r')
O3 = h5py.File('./data/o3a_bbhpop_inj_info.hdf','r')
O3_selection= (O3['injections/ifar_gstlal'][()]>1) | (O3['injections/ifar_pycbc_bbh'][()]>1) | (O3['injections/ifar_pycbc_full'][()]>1)
m1 = np.append(O12['mass1_source'][()],O3['injections/mass1_source'][()][O3_selection])
m2 = np.append(O12['mass2_source'][()],O3['injections/mass2_source'][()][O3_selection])
z = np.append(O12['redshift'][()],O3['injections/redshift'][()][O3_selection])
pdraw = np.append(O12['sampling_pdf'][()],O3['injections/sampling_pdf'][()][O3_selection])
# Maybe missing a jacobian due to m2->q
samples = np.array([m1,m2/m1,z]).T
Ndraw = O3.attrs['total_generated']+7.1*1e7

def evaluate_selection(params,data):
    likelihood = combine_pdf(params,data)
    return np.sum(likelihood/pdraw)/Ndraw
