from abc import ABC, abstractmethod
from typing import Any
import jax.numpy as jnp
import re
import numpy as np
import h5py
import json # to read JSON file
import requests # to download data file via URL
import os.path

# The class will download datafile according to url listed on search_file, and download the content to output_dir
class DataBase:
    # Constructor
    def __init__(self, search_file, output_dir):
        self.search_file = search_file
        self.output_dir = output_dir
        pass
    
    # To download the datafile 
    @abstractmethod
    def fetch(self):
        pass
    
    # To extract the data content from the downloaded datafile into the programme
    @abstractmethod
    def get_data(self):
        pass


# The class processes the posterior data from GWTC-2 and GWTC-3
class PosteriorSampleData(DataBase):
    # Constructor
    def __init__(self, search_file, output_dir):
        super().__init__(search_file, output_dir)
    
    def __call__(self, params):
        self.fetch()
        return self.get_data(params=params)
    
    # download the .h5 posterior samples data into output_dir
    def fetch(self):
        # opening JSON file
        event_list_file = open(self.search_file, "r")
        # Loop through each file contained in the .json list
        for event in (json.load(event_list_file))['files']:
            split_keys = re.split(r'[_\.]', event['key'])
            # check if the file type is .h5 and the data is cosmological reweighted
            if event['type'] == 'h5' and split_keys[-2] == 'cosmo': 
                    url = event['links']['self']
                    filename = event['key'] 
                    # if the file does not exist
                    if os.path.isfile(self.output_dir + filename) == False: 
                        print('Downloading ' + filename)
                        r = requests.get(url, allow_redirects=True)
                        # download the data file into the data folder
                        open(self.output_dir + filename, 'wb').write(r.content)
                    # if the already exist
                    else: 
                        print(filename + ' exists')
    
    # To extract the data content from the downloaded datafile into the programme
    # The data is returned in the form:
    # [[event 1: [para_ 1 posterior samples], [para_2 posterior samples], ...], ...] 
    def get_data(self, dataset_waveform = "C01:Mixed", params=None):
        posterior_samples = []
        # loop through files in the data folder
        for file in os.listdir(self.output_dir):
            # Check if the file is .h5
            if file.split('.')[-1] == "h5":
                try:
                    # Obtain the posterior samples from the downloaded datafile
                    posterior_samples.append(h5py.File(self.output_dir+file)[dataset_waveform+"/posterior_samples"])
                except:
                    print(file) # TODO: error message
        # If user does not specify which event parameters, return all posterior samples
        if params == None:
            return posterior_samples
        else:
            return [[events[param] for param in params] for events in self.posterior_samples]
    
    
# The class processes the sensitivity estimates data 
class SensitivityEstimatesData(DataBase):
    def __init__(self, search_file, output_dir):
        super().__init__(search_file, output_dir)
        
    def __call__(self) -> Any:
        self.fetch()
        return self.get_data()
    
    def fetch(self):
        filename = 'endo3_mixture-LIGO-T2100113-v12.hdf5'
        url = "https://zenodo.org/api/files/abb5598b-2e8d-485e-9b8c-ea8a077b6095/endo3_mixture-LIGO-T2100113-v12.hdf5"
        if os.path.isfile(self.output_dir + filename) == False:
            print('Downloading ' + filename)
            r = requests.get(url, allow_redirects=True)
            # download the data file into the data folder
            open(self.output_dir + filename, 'wb').write(r.content)
        else:
            print(filename + ' exists')
            
    
    def get_data(self):
        # Check if the file is .h5
        if (self.output_dir).split('.')[-1] == "hdf5":
            try:
                return h5py.File(self.output_dir)["injections"]
            except:
                print(self.output_dir) # TODO: error message
        
    
        
        
# It stores the evaluation methods for calculating population model (probability of population parameters given posterior samples)
class PopulationModelBase:
    # Constructor
    def __init__(self) -> None:
        self.population_params_list = None
        
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
        
        log_population_distribution = population_prior # initialize the value to zero
        for event in self.posterior_samples:
            sum = jnp.sum(self.model.get_population_likelihood(population_params, event))
            log_population_distribution += (jnp.log(sum) - jnp.log(len(event[0]))) # sum divided by the number of samples                     
        
        return jnp.where(jnp.isfinite(log_population_distribution), log_population_distribution, -jnp.inf)

        


