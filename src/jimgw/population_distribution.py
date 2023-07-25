from abc import ABC, abstractmethod
import jax.numpy as jnp
import numpy as np
import h5py
import json # to read JSON file
import requests # to download data file via URL
import os.path

# It bookmarks functions that fetch the data, store the data, and format the data
class PosteriorSampleData:
    # Constructor
    def __init__(self, data_file):
        self.posterior_samples = None
        self.data_file = data_file
        pass
    
    # download the .h5 posterior samples data into a directory
    def fetch(self, directory = "data/"):
        self.data_file = directory
        # opening JSON file
        event_list_file = open("event_list.json", "r")
        # return "files" element in JSON object as a dictionary
        event_list = (json.load(event_list_file))['files']
        # Download the posterior samples via url for each event listed in event_list.json
        for event in event_list[:5]:
            if event['type'] == 'h5': # Check if the event links to a H5 file
                if (event['key'][-14:]) == 'mixed_cosmo.h5': # We only want cosmological reweighted data
                    url = event['links']['self'] # get the url
                    filename = event['key'] # get the file name
                    
                    if os.path.isfile(directory + filename) == False: # if the file does not exist
                        print('Downloading ' + filename)
                        r = requests.get(url, allow_redirects=True)
                        open(directory + filename, 'wb').write(r.content) # download the data file into the data folder
                    else: # if the already exist
                        print(filename + ' exists')
    
    # Get the data from the data folder
    def read_file(self, directory=None, data_type = "C01:Mixed"):
        if (directory == None):
            directory = self.data_file
            if (directory == None):
                print("No data file directory specified. ")
                return None
        posterior_samples = []
        for file in os.listdir(directory): # loop through files in the data folder
            if file[-2:] == "h5": # only if the file type is h5
                posterior_samples.append(h5py.File(directory+file)[data_type+"/posterior_samples"]) # append the address of dataframe to the list
        self.posterior_samples = posterior_samples
    
    # Read the .h5 data from a data folder and copy them into a python list with the format
    # [[event 1: [para_ 1 posterior samples], [para_2 posterior samples], ...], ...] 
    def get_all_posterior_samples(self, directory = None):
        # If there is no posterior samples stored in this object, get the data from the .h5 file first
        if self.posterior_samples == None:
            self.read_file(directory)
        return self.posterior_samples
        
    
    # Get the posterior samples of a list of specific parameters and format it into
    # [[event 1: [para_ 1 posterior samples], [para_2 posterior samples], ...], ...] 
    def get_posterior_samples(self, params):
        if (self.posterior_samples == None):
            self.read_file()
        return [[events[param] for param in params] for events in self.posterior_samples]
        
        
    
        
        
# It stores the evaluation methods for calculating population model (probability of population parameters given posterior samples)
class PopulationModelBase:
    # Constructor
    def __init__(self) -> None:
        self.population_params_list = None
        
    @abstractmethod
    def get_population_params_list(self):
        return self.population_params_list
    
    @abstractmethod
    def get_population_likelihood(self):
        pass
    
    @abstractmethod
    def get_population_prior(self):
        pass
    
    def log_uniform_prior(self, min, max, x):
        return jnp.where((x < min) | (x > max), -np.infty, 0.0)

    
class PowerLawModel(PopulationModelBase):
    # Constructor
    def __init__(self):
        self.population_params_list = ["mass_1_source", "mass_ratio"]
    
    def get_population_params_list(self):
        return self.population_params_list
    
    # Evaluate population likelihood by power law
    def get_population_likelihood(self, population_params, posterior_samples):
        alpha, beta, m_min, m_max = population_params[0], population_params[1], population_params[2], population_params[3]
        m_1, q = posterior_samples[0], posterior_samples[1]
        epsilon = 0.001 # a very small number for limit computation

        normalization_constant = 1.0
        normalization_constant *= jnp.where((alpha>(1.0-epsilon))&(alpha<(1.0+epsilon)), jnp.log(m_max/m_min), (m_max**(1.0-alpha)-m_min**(1.0-alpha))/(1.0-alpha))
        
        
        # if :
        #     return 0.0 # The normalization constant will be negative infinity, this gives 0
        # else:
        normalization_constant *= (1.0 / (beta + 1.0))
        
        return jnp.where(((m_1 > m_min) & (m_1 < m_max))|((beta > (-1.0-epsilon)) & (beta<(-1.0+epsilon))),
                        (m_1 ** (-alpha)) * (q ** beta) / normalization_constant,
                        0.0)

    # Evaluate the prior of the power law
    def get_population_prior(self, population_params):
        alpha, beta, m_min, m_max = population_params[0], population_params[1], population_params[2], population_params[3] # alpha, beta,... are double
        output = super().log_uniform_prior(-4., 12., alpha) + super().log_uniform_prior(-4., 12., beta) + super().log_uniform_prior(2.,10.,m_min) + super().log_uniform_prior(30.,100.,m_max)
        return output





# It evaluates the probability of population parameters given data
class PopulationDistribution:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.posterior_samples = self.data.get_posterior_samples(self.model.get_population_params_list())
    
    @abstractmethod
    def get_selection_effect(self):
        return NotImplementedError
    
    # Evaluate the population distribution
    def evaluate(self, population_params, data) -> float:
        # check on population parameters
        population_prior = self.model.get_population_prior(population_params)
        
        # if parameters are ok, do the computation
        log_population_distribution = 0.0 # initialize the value to zero
        for event in self.posterior_samples:
            sum = jnp.sum(self.model.get_population_likelihood(population_params, event))
            log_population_distribution += (population_prior + jnp.log(sum) - jnp.log(len(event[0]))) # sum divided by the number of samples                     
        
        return jnp.where(jnp.isfinite(log_population_distribution), log_population_distribution, -jnp.inf)

        


