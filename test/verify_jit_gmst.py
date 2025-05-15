#!/usr/bin/env python
"""
This script verifies the JAX implementation of the conversion from GPS time
to UTC date and GMST values against the LAL and Bilby implementations, all
the way up to year 2500 (and potentially beyond this).

It is done in two parts:
1. The first part without JIT, which would give an EXACT match with the precomputed values.
2. The second part with JIT enabled, it would give an error of order 1e-10 s.

There are 10,000,000 pre-computed values stored in the file lal_bilby_utc_gmst.npy, which contains
    * gps_time: The input GPS time in seconds
    * lal: LAL computed values
        * year: Year
        * month: Month
        * day: Day
        * sec: Remaing seconds
        * gmst: GMST in radians
    * bilby: Bilby computed values
        * year: Year
        * month: Month
        * day: Day
        * sec: Remaing seconds
        * gmst: GMST in radians
The exact script used to prepare these values is appended as comments.

The gps_time are generated from 1980-01-06 to 2500-12-31.
Note that the LAL implementation, limited by overflow errors, can only compute up to the year 2038.
The remaining years till 2500 (and more) are way beyond practical purposes, and are purely for the
algorithmic comparison with the Bilby implementation.
"""
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")
from jimgw.core.single_event.gps_times import gps_to_utc_date, greenwich_mean_sidereal_time as jim_gmst

# This file contains 10 million GPS timestamps of which the UTC dates
# and GMST values are computed using LAL and Bilby.
computed_times = jnp.load("lal_bilby_utc_gmst.npy")

# This verification is done in chunks as it would apparently take
# too much memory to laod and compare all JAX arrays at once.
chunks = 8

with jax.disable_jit():
    print("================================")
    print("Without JIT:")
    print("================================")
    start = 0
    for end in jnp.linspace(0, 10_000_000 + 1, chunks, dtype=jnp.int32)[1:]:
        print(f"================================")
        print(f"Start index is: {start}")
        print(f"================================")
        _computed_times = computed_times[start:end]
        print(f"Computing {_computed_times.size} samples")
        start_time = time.time()
        gps_times = jnp.asarray(_computed_times["gps_time"])
        utc_dates = jax.vmap(gps_to_utc_date)(gps_times)
        gmst_vals = jax.vmap(jim_gmst)(gps_times)
        print(f"--- Computation takes: {time.time() - start_time:.6f} seconds ---")

        print("For LAL and Bilby:")
        for item in ("year", "month", "day", "sec", "gmst"):
            is_agree = jnp.where(
                _computed_times["lal"]["year"] != 0,
                (_computed_times["lal"][item] == _computed_times["bilby"][item]),
                True,
            ).all()
            print(f" * {item} is agree:   {is_agree}")

        print("For Jim and Bilby:")
        for key, jim_val in zip(
            ("year", "month", "day", "sec", "gmst"), (*utc_dates, gmst_vals)
        ):
            is_agree = (jim_val == _computed_times["bilby"][key]).all()
            print(f" * {key} is agree:   {is_agree}")
        start = end

print("================================")
print("With JIT:")
print("================================")
RTOL = 1e-16
ATOL = 4e-10
start = 0
for end in jnp.linspace(0, 10_000_000 + 1, chunks, dtype=jnp.int32)[1:]:
    print(f"================================")
    print(f"Start index is: {start}")
    print(f"================================")
    _computed_times = computed_times[start:end]
    print(f"Computing {_computed_times.size} samples")
    start_time = time.time()
    gps_times = jnp.asarray(_computed_times["gps_time"])
    utc_dates = jax.vmap(gps_to_utc_date)(gps_times)
    gmst_vals = jax.vmap(jim_gmst)(gps_times)
    print(f"--- Computation takes: {time.time() - start_time:.6f} seconds ---")

    print("For Jim and Bilby:")
    for key, jim_val in zip(
        ("year", "month", "day", "sec", "gmst"), (*utc_dates, gmst_vals)
    ):
        is_agree = jax.numpy.allclose(
            jim_val, _computed_times["bilby"][key], rtol=RTOL, atol=ATOL
        )
        print(f" * {key} is agree:   {is_agree}")
    start = end

"""
Final remarks:
There are some additional tests that are not shown here.

The JAX implementation is tested for near GPS times at 0.0, from range -1000 to +1000, 
    * If one assumes int64, then all results are equivalent.
    * If one assumes float64, and with negative GPS times, after mod(2π), there is always a constant offset by -0.00014584 rad.
"""


"""
#######################
# prepare_UTC_GMST.py #
#######################
#/usr/bin/env python
'''
The purpose of this script is to prepare a set of
UTC dates from LAL and Bilby for comparison with Jim.

The key outputs of this script are:
1. The GPS timestamp
2. The UTC date from LAL and Bilby
3. The GMST from LAL and Bilby

Note: This script can quite some time to run.
'''
import time
from calendar import timegm
import numpy as jnp

from lal import GPSToUTC, \
    GreenwichMeanSiderealTime as LAL_gmst
from bilby_cython.time import \
    gps_time_to_utc as gps_time_to_utc, \
    greenwich_mean_sidereal_time as bilby_gmst

SIZE = 10_000_000
# SIZE = 10
# The test range is the designed time range of the
# Jim UTC date implementation.
start_time = timegm(time.strptime('1980-01-06', '%Y-%m-%d'))
end_time = timegm(time.strptime('2500-12-31', '%Y-%m-%d'))
print(f'{start_time = }, and {end_time = }.')
# Note that the end time is too far in the future and
# since no one can predict precisely when will leap seconds
# be added, this implies that by no means will the UTC time
# be accurate or valid. This is a mere algorithmic test.

def compute_seconds_from_utc_date(hour: int, min: int, sec: int) -> int:
    return sec + min * 60 + hour * 3600

rng = jnp.random.default_rng(seed=1234)
gps_times = jnp.geomspace(start_time, end_time, SIZE, dtype=jnp.int64)
print('Here are the machine precision limits:')
print(jnp.iinfo(jnp.int32), jnp.iinfo(jnp.int64))

none_tuple = tuple([0] * 9)

start_time = time.time()
results = []
for gps_time in gps_times:
    bilby_utc = gps_time_to_utc(gps_time)
    bilby_sec = compute_seconds_from_utc_date(
        bilby_utc.hour, bilby_utc.minute, bilby_utc.second)
    bilby_gmst_val = bilby_gmst(gps_time)
    try:
    # Note that LAL implementation has an "upper limit" of year 2038
    # see: https://lscsoft.docs.ligo.org/lalsuite/lal/_x_l_a_l_civil_time_8c_source.html#l00276
        lal_utc = GPSToUTC(gps_time)
        lal_sec = compute_seconds_from_utc_date(
            lal_utc[3], lal_utc[4], lal_utc[5])
        lal_gmst_val = LAL_gmst(gps_time)
    except OverflowError:
        lal_utc = none_tuple
        lal_sec = 0
        lal_gmst = 0.0
    except RuntimeError:
        print(gps_time)

    results.append(
        (gps_time, (lal_utc[0], lal_utc[1], lal_utc[2], lal_sec, lal_gmst_val),
         (bilby_utc.year, bilby_utc.month, bilby_utc.day, bilby_sec, bilby_gmst_val))
    )

print(f"--- {time.time() - start_time:.6f} seconds ---")

np_results = jnp.array(results, dtype=[
    ('gps_time', jnp.int64),
    ('lal', [('year', jnp.int32), ('month', jnp.int32), ('day', jnp.int32), ('sec', jnp.int32), ('gmst', jnp.float64)]),
    ('bilby', [('year', jnp.int64), ('month', jnp.int64), ('day', jnp.int64), ('sec', jnp.int64), ('gmst', jnp.float64)])
])
jnp.save('lal_bilby_utc_gmst.npy', np_results)
"""
