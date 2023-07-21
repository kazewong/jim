# Import some necessary modules
import emcee
import numpy as np
from jimgw.population_distribution import PosteriorSampleData
from jimgw.population_distribution import PowerLawModel
from jimgw.population_distribution import PopulationDistribution


posterior_samples = PosteriorSampleData() 
posterior_samples.fetch("data/")# It fetch the data and save it into "data" folder
model = PowerLawModel() # Set the model to be power law
posterior_distribution = PopulationDistribution(model, posterior_samples.get_all_posterior_samples())


# parameters for the ensemble sampler
nwalkers = 100 # The number of walkers in the ensemble
ndim = 4 # The number of dimensions in the parameter space
nsteps = 1000 # The number of steps to run
param_initial_guess = [2.5, 6.0, 4.5, 80.0]
backend_file = "output.h5" # output file name

np.random.seed(0)

initial_state = np.random.randn(nwalkers, ndim)

for walker in initial_state:
    for i in range(len(param_initial_guess)):
        walker[i] += param_initial_guess[i] # initial prediction

# Set up the backend
# Don't forget to clear it in case the file already exists
backend = emcee.backends.HDFBackend(backend_file)
backend.reset(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn=posterior_distribution.get_distribution, args=[],backend=backend)
# args is a list of extra arguments for log_prob_fn, log_prob_fn will be called with the sequence log_pprob_fn(p, *args, **kwargs)

sampler.run_mcmc(initial_state, nsteps, progress=True)