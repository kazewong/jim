import numpy as np
import jax.numpy as jnp
from jax import random, grad, jit, vmap

key = random.PRNGKey(42)


def gaussian(x,mean,sigma):
    return (1./jnp.sqrt(2*jnp.pi)/sigma)*jnp.exp(-(((x-mean)/sigma)**2)/2)

def population_likelihood(params, data):
    # Checkpoint 1, what are these lines doing 
    if (params[0]>10) or (params[0]<-10):
        return -jnp.inf
    if (params[1]>5) or (params[1]<0):
        return -jnp.inf
    # End of Checkpoint 1
    output = jnp.sum(jnp.log(gaussian(data,params[0],params[1]))) # Checkpoint 2, what is this line doing? How does it compared to the full form we have in the introduction?
    if jnp.isfinite(output):
        return output
    else:
        return -jnp.inf

true_param = [0,1]
data = random.normal(key,shape=(100,))*true_param[1]+true_param[0]
#data = jnp.append(data,10)

dLdlambda = grad(population_likelihood)([0.,1.],data)
dLdtheta = grad(population_likelihood,argnums=1)([0.,1.],data)

dlambdadtheta = (dLdtheta[None,:]/jnp.array(dLdlambda)[:,None]).mean(axis=0)
