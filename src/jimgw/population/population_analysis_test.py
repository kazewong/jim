import argparse
import numpy as np
import jax
import jax.numpy as jnp
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC import Sampler
from flowMC.proposal.MALA import MALA
import corner
from jimgw.population.population_likelihood import PopulationLikelihood
from jimgw.population.utils import create_model

def parse_args():
    parser = argparse.ArgumentParser(description='Run population likelihood sampling.')
    parser.add_argument('--pop_model', type=str, required=True, help='Population model to use.')
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
    
    # Randomly generate mass arrays for population analysis
    num_samples = 5000  # Number of samples to generate
    mass_c_samples = jax.random.uniform(jax.random.PRNGKey(0), shape=(num_samples,), minval=5, maxval=20)  # M_c samples
    eta_samples = jax.random.uniform(jax.random.PRNGKey(1), shape=(num_samples,), minval=0.1, maxval=0.25)  # eta samples (0 < eta < 0.25)

    # Compute mass1 from generated M_c and eta samples
    mass1_samples = mass1_from_mchirp_eta(mass_c_samples, eta_samples)
    
    # Convert to JAX array
    mass_array = jnp.array(mass1_samples)

    def pop_likelihood(pop_params, data):
        model = create_model(args.pop_model)
        likelihood = PopulationLikelihood(mass_array, model, pop_params)
        log_likelihood = likelihood.evaluate(mass_array, pop_params)
        return log_likelihood
     
    n_dim = create_model(args.pop_model).get_pop_params_dimension() 
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

    rng_key, subkey = jax.random.split(jax.random.PRNGKey(42))
    initial_position = jax.random.normal(subkey, shape=(n_chains, n_dim)) * 1


    nf_sampler = Sampler(n_dim,
                         subkey,
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