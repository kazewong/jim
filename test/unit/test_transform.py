import jax
import jax.numpy as jnp

from jimgw.single_event.transforms import SpinAnglesToCartesianSpinTransform
from jimgw.single_event.utils import m1_m2_to_Mc_q

jax.config.update("jax_enable_x64", True)


class TestSpinTransform:
    forward_keys = (
        "theta_jn",
        "phi_jl",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "a_1",
        "a_2",
        "M_c",
        "q",
        "phase_c",
    )

    backward_keys = (
        "iota",
        "s1_x",
        "s1_y",
        "s1_z",
        "s2_x",
        "s2_y",
        "s2_z",
        "M_c",
        "q",
        "phase_c",
    )

    def test_forward_spin_transform(self):
        """
        Test transformation from spin angles to cartesian spins

        Input and output values are generated with bilby_spin.py
        """

        # read inputs from binary
        inputs = jnp.load("test/unit/source_files/spin_angles_input.npz")
        inputs = [inputs[key] for key in inputs.keys()]
        M_c, q = m1_m2_to_Mc_q(inputs[7], inputs[8])

        # read outputs from binary
        bilby_spins = jnp.load(
            "test/unit/source_files/cartesian_spins_output_for_bilby.npz"
        )
        bilby_spins = jnp.array([bilby_spins[key] for key in bilby_spins.keys()]).T

        n = 100  # max: 100

        # compute jimgw spins
        for i in range(n):
            row = [
                inputs[0][i],
                inputs[1][i],
                inputs[2][i],
                inputs[3][i],
                inputs[4][i],
                inputs[5][i],
                inputs[6][i],
                M_c[i],
                q[i],
                inputs[10][i],
            ]

            freq_ref = inputs[9][i]

            jimgw_spins, jacobian = SpinAnglesToCartesianSpinTransform(
                freq_ref=freq_ref
            ).transform(dict(zip(self.forward_keys, row)))

            for key in ("M_c", "q", "phase_c"):
                jimgw_spins.pop(key)

            assert jnp.allclose(jnp.array(list(jimgw_spins.values())), bilby_spins[i])
            # default atol: 1e-8, rtol: 1e-5
            assert not jnp.isnan(jacobian).any()

    def test_backward_spin_transform(self):
        """
        Test transformation from cartesian spins to spin angles

        Input and output values are generated with bilby_spin.py
        """

        # read inputs from binary
        inputs = jnp.load("test/unit/source_files/cartesian_spins_input.npz")
        inputs = [inputs[key] for key in inputs.keys()]
        M_c, q = m1_m2_to_Mc_q(inputs[7], inputs[8])

        # read outputs from binary
        bilby_spins = jnp.load(
            "test/unit/source_files/spin_angles_output_for_bilby.npz"
        )
        bilby_spins = jnp.array([bilby_spins[key] for key in bilby_spins.keys()]).T

        n = 100  # max: 100

        # compute jimgw spins
        for i in range(n):
            row = [
                inputs[0][i],
                inputs[1][i],
                inputs[2][i],
                inputs[3][i],
                inputs[4][i],
                inputs[5][i],
                inputs[6][i],
                M_c[i],
                q[i],
                inputs[10][i],
            ]

            freq_ref = inputs[9][i]

            jimgw_spins, jacobian = SpinAnglesToCartesianSpinTransform(
                freq_ref=freq_ref
            ).inverse(dict(zip(self.backward_keys, row)))

            for key in ("M_c", "q", "phase_c"):
                jimgw_spins.pop(key)
            jimgw_spins = list(jimgw_spins.values())

            assert jnp.allclose(jnp.array(jimgw_spins), bilby_spins[i])
            # default atol: 1e-8, rtol: 1e-5
            assert not jnp.isnan(jacobian).any()

    def test_forward_backward_consistency(self):
        """
        Test that the forward and inverse transformations are consistent
        """

        n = 10

        key = jax.random.PRNGKey(42)
        for _ in range(n):
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, 5)
            iota = jax.random.uniform(subkeys[0], (1,), minval=0, maxval=jnp.pi)
            M_c = jax.random.uniform(subkeys[1], (1,), minval=1, maxval=100)
            q = jax.random.uniform(subkeys[2], (1,), minval=0.125, maxval=1)
            fRef = jax.random.uniform(subkeys[3], (1,), minval=10, maxval=100)
            phiRef = jax.random.uniform(subkeys[4], (1,), minval=0, maxval=2 * jnp.pi)

            while True:
                key, subkey = jax.random.split(key)
                subkeys = jax.random.split(subkey, 6)
                S1x = jax.random.uniform(subkeys[0], (1,), minval=-1, maxval=1)
                S1y = jax.random.uniform(subkeys[1], (1,), minval=-1, maxval=1)
                S1z = jax.random.uniform(subkeys[2], (1,), minval=-1, maxval=1)
                S2x = jax.random.uniform(subkeys[3], (1,), minval=-1, maxval=1)
                S2y = jax.random.uniform(subkeys[4], (1,), minval=-1, maxval=1)
                S2z = jax.random.uniform(subkeys[5], (1,), minval=-1, maxval=1)
                if (
                    jnp.linalg.norm(jnp.array([S1x[0], S1y[0], S1z[0]])) <= 1
                    and jnp.linalg.norm(jnp.array([S2x[0], S2y[0], S2z[0]])) <= 1
                ):
                    break

            sample = [
                iota[0],
                S1x[0],
                S1y[0],
                S1z[0],
                S2x[0],
                S2y[0],
                S2z[0],
                M_c[0],
                q[0],
                phiRef[0],
            ]

            jimgw_spins, _ = SpinAnglesToCartesianSpinTransform(
                freq_ref=fRef[0]
            ).inverse(dict(zip(self.backward_keys, sample)))
            jimgw_spins, _ = SpinAnglesToCartesianSpinTransform(
                freq_ref=fRef[0]
            ).transform(jimgw_spins)
            jimgw_spins = list(jimgw_spins.values())
            assert jnp.allclose(
                jnp.array(jimgw_spins), jnp.array([*sample[7:], *sample[:7]])
            )
            # default atol: 1e-8, rtol: 1e-5

    def test_jitted_forward_transform(self):
        """
        Test that the forward transformation is JIT compilable
        """

        # Generate random sample
        subkeys = jax.random.split(jax.random.PRNGKey(12), 6)
        iota = jax.random.uniform(subkeys[0], (1,), minval=0, maxval=jnp.pi)
        M_c = jax.random.uniform(subkeys[1], (1,), minval=1, maxval=100)
        q = jax.random.uniform(subkeys[2], (1,), minval=0.125, maxval=1)
        fRef = jax.random.uniform(subkeys[3], (1,), minval=10, maxval=100)
        phiRef = jax.random.uniform(subkeys[4], (1,), minval=0, maxval=2 * jnp.pi)

        key = subkeys[5]
        while True:
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, 6)
            S1x = jax.random.uniform(subkeys[0], (1,), minval=-1, maxval=1)
            S1y = jax.random.uniform(subkeys[1], (1,), minval=-1, maxval=1)
            S1z = jax.random.uniform(subkeys[2], (1,), minval=-1, maxval=1)
            S2x = jax.random.uniform(subkeys[3], (1,), minval=-1, maxval=1)
            S2y = jax.random.uniform(subkeys[4], (1,), minval=-1, maxval=1)
            S2z = jax.random.uniform(subkeys[5], (1,), minval=-1, maxval=1)
            if (
                jnp.linalg.norm(jnp.array([S1x[0], S1y[0], S1z[0]])) <= 1
                and jnp.linalg.norm(jnp.array([S2x[0], S2y[0], S2z[0]])) <= 1
            ):
                break

        sample = [
            iota[0],
            S1x[0],
            S1y[0],
            S1z[0],
            S2x[0],
            S2y[0],
            S2z[0],
            M_c[0],
            q[0],
            phiRef[0],
        ]
        freq_ref_sample = fRef[0]
        sample_dict = dict(zip(self.forward_keys, sample))

        # Create a JIT compiled version of the transform.
        jit_transform = jax.jit(
            lambda data: SpinAnglesToCartesianSpinTransform(
                freq_ref=freq_ref_sample
            ).transform(data)
        )
        jitted_spins, jitted_jacobian = jit_transform(sample_dict)
        non_jitted_spins, non_jitted_jacobian = SpinAnglesToCartesianSpinTransform(
            freq_ref=freq_ref_sample
        ).transform(sample_dict)

        # Remove keys that are not used in the comparison
        for key in ("M_c", "q", "phase_c"):
            jitted_spins.pop(key)
            non_jitted_spins.pop(key)

        # Assert that the jitted and non-jitted results agree
        assert jnp.allclose(
            jnp.array(list(dict(sorted(jitted_spins.items())).values())),
            jnp.array(list(dict(sorted(non_jitted_spins.items())).values())),
        )

        # Also check that the jitted jacobian contains no NaNs
        assert not jnp.isnan(jitted_jacobian).any()

    def test_jitted_backward_transform(self):
        """
        Test that the backward transformation is JIT compilable
        """

        # Generate random sample
        subkeys = jax.random.split(jax.random.PRNGKey(123), 11)

        theta_jn = jax.random.uniform(subkeys[0], (1,), minval=0, maxval=jnp.pi)
        phi_jl = jax.random.uniform(subkeys[1], (1,), minval=0, maxval=2 * jnp.pi)
        tilt_1 = jax.random.uniform(subkeys[2], (1,), minval=0, maxval=jnp.pi)
        tilt_2 = jax.random.uniform(subkeys[3], (1,), minval=0, maxval=jnp.pi)
        phi_12 = jax.random.uniform(subkeys[4], (1,), minval=0, maxval=2 * jnp.pi)
        a_1 = jax.random.uniform(subkeys[5], (1,), minval=0, maxval=1)
        a_2 = jax.random.uniform(subkeys[6], (1,), minval=0, maxval=1)
        M_c = jax.random.uniform(subkeys[7], (1,), minval=1, maxval=100)
        q = jax.random.uniform(subkeys[8], (1,), minval=0.125, maxval=1)
        phase_c = jax.random.uniform(subkeys[9], (1,), minval=0, maxval=2 * jnp.pi)
        f_ref = jax.random.uniform(subkeys[10], (1,), minval=10, maxval=100)

        sample = [
            theta_jn[0],
            phi_jl[0],
            tilt_1[0],
            tilt_2[0],
            phi_12[0],
            a_1[0],
            a_2[0],
            M_c[0],
            q[0],
            phase_c[0],
        ]
        freq_ref_sample = f_ref[0]
        sample_dict = dict(zip(self.backward_keys, sample))

        # Create a JIT compiled version of the transform.
        jit_inverse_transform = jax.jit(
            lambda data: SpinAnglesToCartesianSpinTransform(
                freq_ref=freq_ref_sample
            ).inverse(data)
        )
        jitted_spins, jitted_jacobian = jit_inverse_transform(sample_dict)
        non_jitted_spins, non_jitted_jacobian = SpinAnglesToCartesianSpinTransform(
            freq_ref=freq_ref_sample
        ).inverse(sample_dict)

        # Remove keys that are not used in the comparison
        for key in ("M_c", "q", "phase_c"):
            jitted_spins.pop(key)
            non_jitted_spins.pop(key)

        # Assert that the jitted and non-jitted results agree.
        assert jnp.allclose(
            jnp.array(list(dict(sorted(jitted_spins.items())).values())),
            jnp.array(list(dict(sorted(non_jitted_spins.items())).values())),
        )

        # Also check that the jitted jacobian contains no NaNs
        assert not jnp.isnan(jitted_jacobian).any()
