jim_default_hyperparameters = {
        "seed": 0,
        "n_chains": 20,
        "num_layers": 10,
        "hidden_size": [128,128],
        "num_bins": 8,
        "local_sampler_arg": {},
        "n_walkers_maximize_likelihood": 100,
        "n_loops_maximize_likelihood": 200, 
}

jim_explanation_hyperparameters = {
        "seed": "(int) Value of the random seed used",
        "n_chains": "(int) Number of chains to be used",
        "num_layers": "(int) Number of hidden layers of the NF",
        "hidden_size": "List[int, int] Sizes of the hidden layers of the NF",
        "num_bins": "(int) Number of bins used in MaskedCouplingRQSpline",
        "local_sampler_arg": "(dict) Additional arguments to be used in the local sampler",
        "rng_key_set": "(jnp.array) Key set to be used in PRNG keys",
        "n_walkers_maximize_likelihood": "(int) Number of walkers used in the maximization of the likelihood with the evolutionary optimizer",
        "n_loops_maximize_likelihood": "(int) Number of loops to run the evolutionary optimizer in the maximization of the likelihood",
}
