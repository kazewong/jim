import jax
import jax.numpy as jnp

from jimgw.single_event.runManager import MultipleEventRunManager

jax.config.update("jax_enable_x64", True)

run_manager = MultipleEventRunManager(
    run_config_path="config",
)

run_manager.run()
