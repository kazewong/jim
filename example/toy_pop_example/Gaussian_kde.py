import jax
import jax.numpy as jnp
from jax import jit,vmap

@jit
def gaussian(x,mean,sigma):
    return (1./jnp.sqrt(2*jnp.pi)/sigma)*jnp.exp(-(((x-mean)/sigma)**2)/2)

@jit
def multivariate_gaussian(x,mean,covariance,dim=1):
	numerator = jnp.exp(-1./2*(x-mean).T@jnp.linalg.inv(covariance)@(x-mean))
	denominator = jnp.sqrt((2*jnp.pi)**dim*jnp.linalg.det(covariance))
	return numerator/denominator 

batch_multivariate_gaussian = vmap(multivariate_gaussian, (None,0,None), 0)

def gaussian_kde(datapoint,training_point):
	n = datapoint.shape[0]
	d = datapoint.shape[1]
	bandwidth = n**(-1/(d+4))
	cov_matrix = jnp.eye(d)
	return jnp.mean(batch_multivariate_gaussian(datapoint,training_point,cov_matrix,dim=d))
	

