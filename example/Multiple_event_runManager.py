import jax
import jax.numpy as jnp

from jimgw.single_event.runManager import MultipleEventPERunManager

jax.config.update("jax_enable_x64", True)

run_manager = MultipleEventPERunManager(
    run_config_path="config", # the configuration file is stored in the config folder
)

run_manager.run()
