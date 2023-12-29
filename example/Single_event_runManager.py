import time
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import (
    HeterodynedTransientLikelihoodFD,
    TransientLikelihoodFD,
)
from jimgw.single_event.waveform import RippleIMRPhenomD
from jimgw.single_event.runManager import SingleEventPERunManager, SingleEventRun
from jimgw.prior import Unconstrained_Uniform, Composite
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

run = SingleEventRun(
    seed=0,
    path="test_data/GW150914/",
    detectors=["H1", "L1"],
    priors={
        "M_c": {"name": "Unconstrained_Uniform", "xmin": 10.0, "xmax": 80.0},
        "q": {"name": "Unconstrained_Uniform", "xmin": 0.125, "xmax": 1.0},
        "s1_z": {"name": "Unconstrained_Uniform", "xmin": -1.0, "xmax": 1.0},
        "s2_z": {"name": "Unconstrained_Uniform", "xmin": -1.0, "xmax": 1.0},
        "d_L": {"name": "Unconstrained_Uniform", "xmin": 0.0, "xmax": 2000.0},
        "t_c": {"name": "Unconstrained_Uniform", "xmin": -0.05, "xmax": 0.05},
        "phase_c": {"name": "Unconstrained_Uniform", "xmin": 0.0, "xmax": 2 * jnp.pi},
        "cos_iota": {"name": "Unconstrained_Uniform", "xmin": -1.0, "xmax": 1.0},
        "psi": {"name": "Unconstrained_Uniform", "xmin": 0.0, "xmax": jnp.pi},
        "ra": {"name": "Unconstrained_Uniform", "xmin": 0.0, "xmax": 2 * jnp.pi},
        "sin_dec": {"name": "Unconstrained_Uniform", "xmin": -1.0, "xmax": 1.0},
    },
    waveform_parameters={"name": "RippleIMRPhenomD", "f_ref": 20.0},
    jim_parameters={
        "n_loop_training": 100,
        "n_loop_production": 10,
        "n_local_steps": 150,
        "n_global_steps": 150,
        "n_chains": 500,
        "n_epochs": 50,
        "learning_rate": 0.001,
        "max_samples": 45000,
        "momentum": 0.9,
        "batch_size": 50000,
        "use_global": True,
        "keep_quantile": 0.0,
        "train_thinning": 1,
        "output_thinning": 10,
    },
    likelihood_parameters={"name": "TransientLikelihoodFD"},
    injection_parameters={},
    data_parameters={
        "trigger_time": 1126259462.4,
        "duration": 4,
        "post_trigger_duration": 2,
        "f_min": 20.0,
        "f_max": 1024.0,
        "tukey_alpha": 0.2,
    },
)

run_manager = SingleEventPERunManager(run=run)
