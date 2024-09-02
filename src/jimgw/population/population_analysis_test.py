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

def main():
    args = parse_args()
    num_samples = 500  
    mass_samples = jax.random.uniform(jax.random.PRNGKey(0), shape=(num_samples,), minval=5, maxval=20)  # M_c samples
    mass_array = jnp.array(mass_samples)

    def pop_likelihood(pop_params, data):
        model = create_model(args.pop_model)
        likelihood = PopulationLikelihood(mass_array, model, pop_params)
        log_likelihood = likelihood.evaluate(mass_array, pop_params)
        return log_likelihood
     
    n_dim = create_model(args.pop_model).get_pop_params_dimension() 
    n_chains = 10 


    model = MaskedCouplingRQSpline(n_layers=3, hidden_size=[64, 64], num_bins=8, n_features=n_dim, key=jax.random.PRNGKey(0))

    step_size = 1
    MALA_Sampler = MALA(pop_likelihood, True, {"step_size": step_size})
    

    rng_key, subkey = jax.random.split(jax.random.PRNGKey(42))
    positions = jax.random.normal(subkey, shape=(n_chains, n_dim)) * 1


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