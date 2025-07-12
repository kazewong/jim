import bilby
from bilby.gw.utils import zenith_azimuth_to_ra_dec
from itertools import combinations
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

outdir = "test/unit/source_files/"
N_samples = 100
ifo_names = ["H1", "L1", "V1"]
geocent_time = [1126259642.413]

key = jax.random.PRNGKey(42)

for ifo_pair in combinations(ifo_names, 2):
    ifos = bilby.gw.detector.InterferometerList(ifo_pair)
    for time in geocent_time:
        key, *subkey = jax.random.split(key, 3)
        azimuth = jax.random.uniform(
            subkey[0], (N_samples,), minval=0, maxval=2 * jnp.pi
        )
        zenith = jax.random.uniform(subkey[1], (N_samples,), minval=0, maxval=jnp.pi)
        inputs = jnp.array([zenith, azimuth]).T
        output = []
        for row in inputs:
            output.append(zenith_azimuth_to_ra_dec(*row, time, ifos))
        ra, dec = jnp.array(output).T
        input_dict = {
            "zenith": zenith,
            "azimuth": azimuth,
            "ra": ra,
            "dec": dec,
            "gps_time": time,
            "ifo_pair": ifo_pair,
        }

        jnp.savez(
            f"{outdir}/sky_locations/test_{ifo_pair[0]}_{ifo_pair[1]}_{time}.npz",
            **input_dict,
        )
