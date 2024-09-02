import argparse
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import glob
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC import Sampler
from flowMC.proposal.MALA import MALA
import corner
from jimgw.population.population_likelihood import PopulationLikelihood
from jimgw.population.utils import create_model


def parse_args():
    parser = argparse.ArgumentParser(description='Run population likelihood sampling.')
    parser.add_argument('--pop_model', type=str, required=True, help='Population model to use.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the NPZ data files.')
    return parser.parse_args()

def mtotal_from_mchirp_eta(mchirp, eta):
    """Returns the total mass from the chirp mass and symmetric mass ratio."""
    return mchirp / eta**(3./5.)

def mass1_from_mtotal_eta(mtotal, eta):
    """Returns the primary mass from the total mass and symmetric mass ratio."""
    return 0.5 * mtotal * (1.0 + (1.0 - 4.0 * eta)**0.5)

def mass1_from_mchirp_eta(mchirp, eta):
    """Returns the primary mass from the chirp mass and symmetric mass ratio."""
    mtotal = mtotal_from_mchirp_eta(mchirp, eta)
    return mass1_from_mtotal_eta(mtotal, eta)

def prior_alpha(alpha):
    return jax.lax.cond(alpha > 0, lambda: 0.0, lambda: -jnp.inf)

def prior_x_min_x_max(x_min, x_max):
    cond_1 = (x_max > x_min)
    cond_2 = (x_min >= 5) & (x_min <= 20)
    cond_3 = (x_max >= 50) & (x_max <= 100)
    
    return jax.lax.cond(cond_1 & cond_2 & cond_3, lambda: 0.0, lambda: -jnp.inf)

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # For sampling events
    directory = args.data_dir  # Use the data directory from command-line argument
    key = jax.random.PRNGKey(42)
    mass_result_dict = []
    npz_files = glob.glob(directory + '/*.npz')

    num_files_to_sample = 100
    key, subkey = jax.random.split(key)
    sample_indices = jax.random.choice(subkey, len(npz_files), shape=(num_files_to_sample,), replace=False)
    sampled_npz_files = [npz_files[i] for i in sample_indices]

    for npz_file in sampled_npz_files:
        print("Loading file:", npz_file)
        with np.load(npz_file, allow_pickle=True) as data:
            chains = data['chains']
            reshaped_chains = chains.reshape(-1, 11)
            event_df = pd.DataFrame(reshaped_chains, columns=[
                'M_c', 'eta', 's1_z', 's2_z', 'd_L', 't_c', 'phase_c', 
                'iota', 'psi', 'ra', 'dec'
            ])
            
            # Randomly sample rows within each file in a reproducible manner
            key, subkey = jax.random.split(key)
            sample_indices = jax.random.choice(subkey, event_df.shape[0], shape=(5000,), replace=False)
            sampled_df = event_df.iloc[sample_indices]
            
            # Extract M_c and eta using sampled indices
            mc_sampled = sampled_df['M_c'].values
            eta_sampled = sampled_df['eta'].values
            
            # Compute mass1
            mass1_sampled = mass1_from_mchirp_eta(mc_sampled, eta_sampled)
            
            # Append to the result dictionary
            mass_array = jnp.array(mass1_sampled)
            mass_result_dict.append(mass_array)

    # Stack all results into a single array
    mass_array = jnp.stack(mass_result_dict)
    
    def pop_likelihood(pop_params ,data):
            model = create_model(args.pop_model)
            likelihood = PopulationLikelihood(mass_array, model, pop_params)
            log_likelihood = likelihood.evaluate(mass_array, pop_params)
            return log_likelihood
    
    
    # def log_likelihood(pop_params, data):
    #         likelihood = PopulationLikelihood(mass_array,TruncatedPowerLawModel, pop_params)
    #         log_likelihood = likelihood.evaluate(mass_array, pop_params)
    #         return log_likelihood 


    n_dim = 3
    n_chains = 1000

    rng_key = jax.random.PRNGKey(42)  

    minval_0th_dim = 5
    maxval_0th_dim = 20

    minval_1st_dim = 50
    maxval_1st_dim = 100

    minval_2nd_dim = 0
    maxval_2nd_dim = 4

    initial_positions = []

    while len(initial_positions) < n_chains:
        rng_key, subkey = jax.random.split(rng_key)
        samples_0th_dim = jax.random.uniform(subkey, shape=(n_chains,), minval=minval_0th_dim, maxval=maxval_0th_dim)
        rng_key, subkey = jax.random.split(rng_key)
        samples_1st_dim = jax.random.uniform(subkey, shape=(n_chains,), minval=minval_1st_dim, maxval=maxval_1st_dim)

        valid_indices = jnp.where((samples_1st_dim >= samples_0th_dim))[0]
        valid_positions = jnp.column_stack([samples_0th_dim[valid_indices], samples_1st_dim[valid_indices]])

        remaining_chains_needed = n_chains - len(initial_positions)
        if len(valid_positions) >= remaining_chains_needed:
            valid_positions = valid_positions[:remaining_chains_needed]

        initial_positions.extend(valid_positions.tolist())

    positions = jnp.column_stack([
        jnp.array(initial_positions),
        jax.random.uniform(rng_key, shape=(n_chains,), minval=minval_2nd_dim, maxval=maxval_2nd_dim)
    ])

    model = MaskedCouplingRQSpline(n_layers=3, hidden_size=[64, 64], num_bins=8, n_features=n_dim, key=jax.random.PRNGKey(0))

    step_size = 1
    MALA_Sampler = MALA(pop_likelihood, True, {"step_size": step_size})

    rng_key_set = initialize_rng_keys(n_chains, seed=42)

    nf_sampler = Sampler(n_dim,
                         rng_key_set,
                         pop_likelihood,
                         MALA_Sampler,
                         model,
                         n_local_steps=1000,
                         n_global_steps=1000,
                         n_epochs=30,
                         learning_rate=1e-3,
                         batch_size=1000,
                         n_chains=n_chains,
                         use_global=True)

    nf_sampler.sample(positions, data=None)
    chains, log_prob, local_accs, global_accs = nf_sampler.get_sampler_state().values()

    corner.corner(np.array(chains.reshape(-1, n_dim))).savefig("corner.png")
    # np.savez("pop_chains/pop_chain.npz", chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)
    print("local:", local_accs)
    print("global:", global_accs)
    print("chains:", chains)
    print("log_prob:", log_prob)

if __name__ == "__main__":
    main()