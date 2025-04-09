import numpy as np
from numpy.random import uniform
from lalsimulation import (
    SimInspiralTransformPrecessingNewInitialConditions,
    SimInspiralTransformPrecessingWvf2PE,
)
from lal import MSUN_SI

outdir = "test/unit/source_files"
N_samples = 100
np.random.seed(12345)

# The following script generates spin angles input files and the cartesian spin outputs for lal
# This is for testing the forward transform of SpinAnglesToCartesianSpinTransform in JIM
fRefs = [10, 30, 50]
for fRef in fRefs:
    m1, m2 = uniform(1, 70, (2, N_samples))
    input_dict = {
        'thetaJN': uniform(0, np.pi, N_samples),
        'phiJL': uniform(0, 2 * np.pi, N_samples),
        'theta1': uniform(0, np.pi, N_samples),
        'theta2': uniform(0, np.pi, N_samples),
        'phi12': uniform(0, 2 * np.pi, N_samples),
        'chi1': uniform(0, 1, N_samples),
        'chi2': uniform(0, 1, N_samples),
        'm1': np.where(m1 >= m2, m1, m2),
        'm2': np.where(m1 >= m2, m2, m1),
        'fRef': np.ones(N_samples) * fRef,
        'phiRef': uniform(0, 2 * np.pi, N_samples),
    }

    assert np.all(input_dict['m1'] >= input_dict['m2'])

    np.savez(
        f"{outdir}/spin_angles_input/spin_angles_input_fRef_{fRef}.npz",
        **input_dict
    )

    input_dict['m1'] *= MSUN_SI
    input_dict['m2'] *= MSUN_SI

    input_array = np.array(list(input_dict.values())).T

    bilby_outputs = []
    for row in input_array:
        bilby_outputs.append(
            SimInspiralTransformPrecessingNewInitialConditions(*row))

    bilby_outputs = np.array(bilby_outputs).T

    np.savez(
        f"{outdir}/cartesian_spins_output_for_bilby/cartesian_spins_output_for_bilby_fRef_{fRef}.npz",
        iota=bilby_outputs[0],
        S1x = bilby_outputs[1],
        S1y = bilby_outputs[2],
        S1z = bilby_outputs[3],
        S2x = bilby_outputs[4],
        S2y = bilby_outputs[5],
        S2z = bilby_outputs[6]
    )

# The following script generates cartesian angles input files and the spin angles outputs for lal
# This is for testing the inverse transform of SpinAnglesToCartesianSpinTransform in JIM
for fRef in fRefs:
    m1, m2 = uniform(1, 70, (2, N_samples))
    S1, S2 = uniform(-1, 1, (2, 3, N_samples))
    a1, a2 = uniform(1e-3, 1, (2, N_samples))
    S1 *= a1 / np.linalg.norm(S1, axis=0)
    S2 *= a2 / np.linalg.norm(S2, axis=0)

    input_dict = {
        'iota': uniform(0, np.pi, N_samples),
        'S1x': S1[0],
        'S1y': S1[1],
        'S1z': S1[2],
        'S2x': S2[0],
        'S2y': S2[1],
        'S2z': S2[2],
        'm1': np.where(m1 >= m2, m1, m2),
        'm2': np.where(m1 >= m2, m2, m1),
        'fRef': np.ones(N_samples) * fRef,
        'phiRef': uniform(0, 2*np.pi, N_samples)
    }
    assert np.all(input_dict['m1'] >= input_dict['m2'])

    np.savez(
        f"{outdir}/cartesian_spins_input/cartesian_spins_input_fRef_{fRef}.npz",
        **input_dict
    )

    bilby_outputs = []
    for row in input_array:
        bilby_outputs.append(
            SimInspiralTransformPrecessingWvf2PE(*row))

    bilby_outputs = np.array(bilby_outputs).T

    np.savez(
        f"{outdir}/spin_angles_output_for_bilby/spin_angles_output_for_bilby_fRef_{fRef}.npz",
        thetaJN = bilby_outputs[0],
        phiJL = bilby_outputs[1],
        theta1 = bilby_outputs[2],
        theta2 = bilby_outputs[3],
        phi12 = bilby_outputs[4],
        chi1 = bilby_outputs[5],
        chi2 = bilby_outputs[6],
    )
