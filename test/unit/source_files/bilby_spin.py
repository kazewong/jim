import numpy as np
from lalsimulation import (
    SimInspiralTransformPrecessingNewInitialConditions,
    SimInspiralTransformPrecessingWvf2PE,
)
from lal import MSUN_SI
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses

np.random.seed(12345)

inputs = []
for _ in range(100):
    thetaJN = np.array(np.random.uniform(0, np.pi))
    phiJL = np.array(np.random.uniform(0, 2 * np.pi))
    theta1 = np.array(np.random.uniform(0, np.pi))
    theta2 = np.array(np.random.uniform(0, np.pi))
    phi12 = np.array(np.random.uniform(0, 2 * np.pi))
    chi1 = np.array(np.random.uniform(0, 1))
    chi2 = np.array(np.random.uniform(0, 1))
    M_c = np.array(np.random.uniform(1, 100))
    q = np.array(np.random.uniform(0.125, 1))
    fRef = np.array(np.random.uniform(10, 1000))
    phiRef = np.array(np.random.uniform(0, 2 * np.pi))

    m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(M_c, q)

    inputs.append(
        (
            thetaJN,
            phiJL,
            theta1,
            theta2,
            phi12,
            chi1,
            chi2,
            m1,
            m2,
            fRef,
            phiRef,
        )
    )
inputs = np.array(inputs)
np.savez(
    "test/unit/source_files/spin_angles_input.npz",
    thetaJN=inputs[:, 0],
    phiJL=inputs[:, 1],
    theta1=inputs[:, 2],
    theta2=inputs[:, 3],
    phi12=inputs[:, 4],
    chi1=inputs[:, 5],
    chi2=inputs[:, 6],
    m1=inputs[:, 7],
    m2=inputs[:, 8],
    fRef=inputs[:, 9],
    phiRef=inputs[:, 10],
)

bilby_outputs = []
for input in inputs:
    iota, S1x, S1y, S1z, S2x, S2y, S2z = (
        SimInspiralTransformPrecessingNewInitialConditions(
            input[0],
            input[1],
            input[2],
            input[3],
            input[4],
            input[5],
            input[6],
            input[7] * MSUN_SI,
            input[8] * MSUN_SI,
            input[9],
            input[10],
        )
    )
    bilby_outputs.append((iota, S1x, S1y, S1z, S2x, S2y, S2z))
bilby_outputs = np.array(bilby_outputs)
np.savez(
    "test/unit/source_files/cartesian_spins_output_for_bilby.npz",
    iota=bilby_outputs[:, 0],
    S1x=bilby_outputs[:, 1],
    S1y=bilby_outputs[:, 2],
    S1z=bilby_outputs[:, 3],
    S2x=bilby_outputs[:, 4],
    S2y=bilby_outputs[:, 5],
    S2z=bilby_outputs[:, 6],
)


inputs = []
for _ in range(100):
    iota = np.array(np.random.uniform(0, np.pi))
    while True:
        S1x = np.array(np.random.uniform(-1, 1))
        S1y = np.array(np.random.uniform(-1, 1))
        S1z = np.array(np.random.uniform(-1, 1))
        S2x = np.array(np.random.uniform(-1, 1))
        S2y = np.array(np.random.uniform(-1, 1))
        S2z = np.array(np.random.uniform(-1, 1))
        if (
            np.linalg.norm([S1x, S1y, S1z]) <= 1
            and np.linalg.norm([S2x, S2y, S2z]) <= 1
        ):
            break
    M_c = np.array(np.random.uniform(1, 100))
    q = np.array(np.random.uniform(0.125, 1))
    fRef = np.array(np.random.uniform(10, 100))
    phiRef = np.array(np.random.uniform(0, 2 * np.pi))

    m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(M_c, q)

    inputs.append((iota, S1x, S1y, S1z, S2x, S2y, S2z, m1, m2, fRef, phiRef))
inputs = np.array(inputs)
np.savez(
    "test/unit/source_files/cartesian_spins_input.npz",
    iota=inputs[:, 0],
    S1x=inputs[:, 1],
    S1y=inputs[:, 2],
    S1z=inputs[:, 3],
    S2x=inputs[:, 4],
    S2y=inputs[:, 5],
    S2z=inputs[:, 6],
    m1=inputs[:, 7],
    m2=inputs[:, 8],
    fRef=inputs[:, 9],
    phiRef=inputs[:, 10],
)

bilby_outputs = []
for input in inputs:
    thteaJN, phiJL, theta1, theta2, phi12, chi1, chi2 = (
        SimInspiralTransformPrecessingWvf2PE(*input)
    )
    bilby_outputs.append((thteaJN, phiJL, theta1, theta2, phi12, chi1, chi2))
bilby_outputs = np.array(bilby_outputs)
np.savez(
    "test/unit/source_files/spin_angles_output_for_bilby.npz",
    thetaJN=bilby_outputs[:, 0],
    phiJL=bilby_outputs[:, 1],
    theta1=bilby_outputs[:, 2],
    theta2=bilby_outputs[:, 3],
    phi12=bilby_outputs[:, 4],
    chi1=bilby_outputs[:, 5],
    chi2=bilby_outputs[:, 6],
)
