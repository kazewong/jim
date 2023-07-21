from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import json # to read JSON file
import requests # to download data file via URL
import os.path


class PosteriorSampleData:
    # Constructor
    def __init__(self) -> None:
        pass
    
    # download the .h5 data into a directory
    def fetch(self, directory):
        # opening JSON file
        print(os.path.join(os.path.dirname(__file__),"event_list.json"))
        event_list_file = open(os.path.join(os.path.dirname(__file__),"event_list.json"), "r")

        # return "files" element in JSON object as a dictionary
        event_list = (json.load(event_list_file))['files']

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
    
    def get_posterior_samples(self, directory, data_type = "Mixed"):
        posterior_samples = []
        for file in os.listdir(directory): # loop through files in the data folder
            posterior_samples.append(pd.read_hdf(directory+file, key="C01:"+data_type+"/posterior_samples")) # append the address of dataframe to the list
        return posterior_samples
    
    
        
        

class PopulationModelBase:
    # Constructor
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def get_population_likelihood(self):
        pass
    
    @abstractmethod
    def get_population_prior(self):
        pass
    
    def log_uniform_prior(self, min, max, x):
        if (x < min) | (x > max):
            return -np.infty
        else:
            return 0
    
    
class PowerLawModel(PopulationModelBase):
    # Evaluate population likelihood by power law
    def get_population_likelihood(self, population_params, posterior_samples):
        alpha, beta, m_min, m_max = population_params[0], population_params[1], population_params[2], population_params[3]
        m_1, q = posterior_samples['mass_1_source'], posterior_samples['mass_ratio']
        epsilon = 0.001 # a very small number for limit computation

        normalization_constant = 1.0
        if (alpha>(1.0-epsilon))&(alpha<(1.0+epsilon)):
            normalization_constant *= np.log(m_max/m_min)
        else:
            normalization_constant *= (m_max**(1.0-alpha)-m_min**(1.0-alpha))/(1.0-alpha)
        
        if (beta > (-1.0-epsilon)) & (beta<(-1.0+epsilon)):
            return 0.0 # The normalization constant will be negative infinity, this gives 0
        else:
            normalization_constant *= (1.0 / (beta + 1.0))
        
        return np.where((m_1 > m_min) & (m_1 < m_max),
                        (m_1 ** (-alpha)) * (q ** beta) / normalization_constant,
                        0.0)

    # Evaluate the prior of the power law
    def get_population_prior(self, population_params):
        alpha, beta, m_min, m_max = population_params[0], population_params[1], population_params[2], population_params[3] # alpha, beta,... are double
        output = super().log_uniform_prior(-4., 12., alpha) + super().log_uniform_prior(-4., 12., beta) + super().log_uniform_prior(2.,10.,m_min) + super().log_uniform_prior(30.,100.,m_max)
        return output


class PopulationDistribution:
    def __init__(self, model):
        self.model = model 

    def get_distribution(self, population_params, posterior_samples):
        # check on population parameters
        population_prior = self.model.get_population_prior(population_params)
        
        # if parameters are ok, do the computation
        log_population_distribution = 0.0 # initialize the value to zero
        for event in posterior_samples:
            sum = np.sum(self.model.get_population_likelihood(population_params, event))
            log_population_distribution += (population_prior + np.log(sum) - np.log(event.shape[0])) # sum divided by the number of samples                     
        
        if np.isfinite(log_population_distribution):
            return log_population_distribution
        else:
            return -np.inf


