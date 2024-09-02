import importlib
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jimgw.single_event.utils import Mc_eta_to_m1_m2

def create_model(model_name):
    try:
        module = importlib.import_module('population_model')

        # Check if model_name is a string
        if not isinstance(model_name, str):
            raise ValueError("model_name must be a string")

        model_class = getattr(module, model_name)
        return model_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import model '{model_name}': {str(e)}")


def extract_data_from_npz_files(npz_files, column_name, num_samples=5000, random_seed=42):
    """
    Extracts specified column data from the given .npz files.

    Parameters:
    - npz_files (list of str): List of paths to .npz files.
    - column_name (str): The name of the column to extract from the DataFrame.
    - num_samples (int): Number of samples to extract from each file.
    - random_seed (int): Seed for random number generation.

    Returns:
    - jnp.array: Stacked array of extracted data.
    """

    key = jax.random.PRNGKey(random_seed)
    result_dict = []

    for npz_file in npz_files:
        print("Loading file:", npz_file)
        with np.load(npz_file, allow_pickle=True) as data:
            chains = data['chains']
            reshaped_chains = chains.reshape(-1, 11)
            event_df = pd.DataFrame(reshaped_chains)
            
            # Check if the specified column exists
            if column_name not in event_df.columns:
                raise ValueError(f"Column '{column_name}' not found in the data.")

            key, subkey = jax.random.split(key)
            sample_indices = jax.random.choice(subkey, event_df.shape[0], shape=(num_samples,), replace=True)
            sampled_df = event_df.iloc[sample_indices]
            
            extracted_data = sampled_df[column_name].values
            
            data_array = jnp.array(extracted_data)
            result_dict.append(data_array)

    stacked_array = jnp.stack(result_dict)
    return stacked_array

def extract_data_from_npz_files_m1_m2(npz_files, num_samples=5000, random_seed=42):
    """
    Extracts specified column data from the given .npz files and computes masses.

    Parameters
    - npz_files (list of str): List of paths to .npz files.
    - num_samples (int): Number of samples to extract from each file.
    - random_seed (int): Seed for random number generation.

    Returns
    - m1_array (jnp.array): Stacked array of primary masses.
    - m2_array (jnp.array): Stacked array of secondary masses.
    """
    
    key = jax.random.PRNGKey(random_seed)
    m1_results = []
    m2_results = []

    for npz_file in npz_files:
        print("Loading file:", npz_file)
        with np.load(npz_file, allow_pickle=True) as data:
            chains = data['chains']
            reshaped_chains = chains.reshape(-1, 11)
            event_df = pd.DataFrame(reshaped_chains)

            # Check if the specified columns exist
            if 'M_c' not in event_df.columns:
                raise ValueError(f" M_c not found in the data.")
            if 'eta' not in event_df.columns:
                raise ValueError(f"Eta not found in the data.")

            key, subkey = jax.random.split(key)
            sample_indices = jax.random.choice(subkey, event_df.shape[0], shape=(num_samples,), replace=True)
            sampled_df = event_df.iloc[sample_indices]
            
            # Extract M_c and eta
            M_c_sampled = sampled_df[M_c_column].values
            eta_sampled = sampled_df[eta_column].values
            
            # Transform M_c and eta to m1 and m2
            m1_sampled, m2_sampled = Mc_eta_to_m1_m2(M_c_sampled, eta_sampled)

            # Convert to jax arrays and append to results
            m1_results.append(jnp.array(m1_sampled))
            m2_results.append(jnp.array(m2_sampled))

    # Stack all results into single arrays
    m1_array = jnp.stack(m1_results)
    m2_array = jnp.stack(m2_results)

    return m1_array, m2_array
