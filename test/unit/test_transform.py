import jax
import jax.numpy as jnp

import numpy as np
from itertools import combinations
from pathlib import Path

from jimgw.transforms import (
    ScaleTransform,
    OffsetTransform,
    LogitTransform,
    SineTransform,
    CosineTransform,
    BoundToBound,
    BoundToUnbound,
    SingleSidedUnboundTransform,
    PowerLawTransform,
    reverse_bijective_transform,
)
from jimgw.single_event.transforms import (
    DistanceToSNRWeightedDistanceTransform,
    SphereSpinToCartesianSpinTransform,
    SpinAnglesToCartesianSpinTransform,
    SkyFrameToDetectorFrameSkyPositionTransform,
)

from jimgw.single_event.utils import m1_m2_to_Mc_q
from jimgw.single_event.detector import H1, L1, V1, detector_preset

jax.config.update("jax_enable_x64", True)


def common_keys_allclose(
    dict_1: dict, dict_2: dict, atol: float = 1e-8, rtol: float = 1e-5
):
    """
    Check the common values between two result dictionaries are close.

    Parameters
    ----------
    dict_1 : dict
        Result dictionary from the 1st transform.
    dict_2 : dict
        Result dictionary from the 2nd transform.
    atol : float
        Absolute tolerance between the two values, default 1e-8.
    rtol : float
        Relative tolerance between the two values, default 1e-5.

    The default values for atol and rtol here follows those
    in jax.numpy.isclose, for their definitions, see e.g.:
    https://docs.jax.dev/en/latest/_autosummary/jax.numpy.isclose.html
    """
    common_keys = set.intersection({*dict_1.keys()}, {*dict_2.keys()})
    tuples = jnp.array([[dict_1[key], dict_2[key]] for key in common_keys])
    tuple_array = jnp.swapaxes(tuples, 0, 1)
    return jnp.allclose(*tuple_array, atol=atol, rtol=rtol)


class TestBasicTransforms:
    def test_scale_transform(self):
        name_mapping = (["a", "b"], ["a_scaled", "b_scaled"])
        scale = 3.0
        transform = ScaleTransform(name_mapping, scale)
        input_data = {"a": 2.0, "b": 4.0}

        # Test forward transformation
        output, log_det = transform.transform(input_data.copy())
        assert np.allclose(list(output.values()), [2.0 * scale, 4.0 * scale])
        assert np.allclose(log_det, 2 * jnp.log(scale))

        # Test inverse transformation
        recovered, inv_log_det = transform.inverse(output.copy())
        assert common_keys_allclose(recovered, input_data)
        assert np.allclose(inv_log_det, -2 * jnp.log(scale))

        jit_transform = jax.jit(lambda x: transform.transform(x))
        jit_inverse = jax.jit(lambda x: transform.inverse(x))

        # Test jitted forward transformation
        jitted_output, jitted_log_det = jit_transform(input_data)
        assert np.allclose(list(jitted_output.values()), [2.0 * scale, 4.0 * scale])
        assert np.allclose(jitted_log_det, 2 * jnp.log(scale))

        # Test jitted inverse transformation
        jitted_recovered, jitted_inv_log_det = jit_inverse(jitted_output)
        assert common_keys_allclose(jitted_recovered, input_data)
        assert np.allclose(jitted_inv_log_det, -2 * jnp.log(scale))

    def test_offset_transform(self):
        name_mapping = (["x", "y"], ["x_offset", "y_offset"])
        offset = 5.0
        transform = OffsetTransform(name_mapping, offset)
        input_data = {"x": 10.0, "y": -3.0}

        # Test forward transformation
        output, log_det = transform.transform(input_data.copy())
        assert np.allclose(list(output.values()), [10.0 + offset, -3.0 + offset])
        assert np.allclose(log_det, 0.0)

        # Test inverse transformation
        recovered, inv_log_det = transform.inverse(output.copy())
        assert common_keys_allclose(recovered, input_data)
        assert np.allclose(inv_log_det, 0.0)

        jit_transform = jax.jit(lambda x: transform.transform(x))
        jit_inverse = jax.jit(lambda x: transform.inverse(x))

        # Test jitted forward transformation
        jitted_output, jitted_log_det = jit_transform(input_data)
        assert np.allclose(list(jitted_output.values()), [10.0 + offset, -3.0 + offset])
        assert np.allclose(jitted_log_det, 0.0)

        # Test jitted inverse transformation
        jitted_recovered, jitted_inv_log_det = jit_inverse(jitted_output)
        assert common_keys_allclose(jitted_recovered, input_data)
        assert np.allclose(jitted_inv_log_det, 0.0)

    def test_logit_transform(self):
        name_mapping = (["p"], ["p_logit"])
        transform = LogitTransform(name_mapping)
        input_data = {"p": 0.6}

        # Test forward transformation
        output, log_det = transform.transform(input_data.copy())
        assert np.allclose(output["p_logit"], 1 / (1 + jnp.exp(-0.6)))
        assert np.isfinite(log_det)

        # Test inverse transformation
        recovered, inv_log_det = transform.inverse(output.copy())
        assert common_keys_allclose(recovered, input_data)
        assert np.isfinite(inv_log_det)

        jit_transform = jax.jit(lambda x: transform.transform(x))
        jit_inverse = jax.jit(lambda x: transform.inverse(x))

        # Test jitted forward transformation
        jitted_output, jitted_log_det = jit_transform(input_data)
        assert np.allclose(jitted_output["p_logit"], 1 / (1 + jnp.exp(-0.6)))
        assert np.isfinite(jitted_log_det)

        # Test jitted inverse transformation
        jitted_recovered, jitted_inv_log_det = jit_inverse(jitted_output)
        assert common_keys_allclose(jitted_recovered, input_data)
        assert np.isfinite(jitted_inv_log_det)

    def test_sine_transform(self):
        name_mapping = (["theta"], ["sin_theta"])
        transform = SineTransform(name_mapping)
        angle = 0.3
        input_data = {"theta": angle}

        # Test forward transformation
        output, log_det = transform.transform(input_data.copy())
        assert np.allclose(output["sin_theta"], jnp.sin(angle))
        assert np.allclose(log_det, jnp.log(jnp.abs(jnp.cos(angle))))

        # Test inverse transformation
        recovered, inv_log_det = transform.inverse(output.copy())
        assert np.allclose(recovered["theta"], angle)
        assert np.allclose(inv_log_det, -jnp.log(jnp.abs(jnp.cos(angle))))

        jit_transform = jax.jit(lambda x: transform.transform(x))
        jit_inverse = jax.jit(lambda x: transform.inverse(x))

        # Test jitted forward transformation
        jitted_output, jitted_log_det = jit_transform(input_data)
        assert np.allclose(jitted_output["sin_theta"], jnp.sin(angle))
        assert np.allclose(jitted_log_det, jnp.log(jnp.abs(jnp.cos(angle))))

        # Test jitted inverse transformation
        jitted_recovered, jitted_inv_log_det = jit_inverse(jitted_output)
        assert np.allclose(jitted_recovered["theta"], angle)
        assert np.allclose(jitted_inv_log_det, -jnp.log(jnp.abs(jnp.cos(angle))))

    def test_cosine_transform(self):
        name_mapping = (["theta"], ["cos_theta"])
        transform = CosineTransform(name_mapping)
        angle = 1.2
        input_data = {"theta": angle}

        # Test forward transformation
        output, log_det = transform.transform(input_data.copy())
        assert np.allclose(output["cos_theta"], jnp.cos(angle))
        assert np.isfinite(log_det)

        # Test inverse transformation
        recovered, inv_log_det = transform.inverse(output.copy())
        assert np.allclose(recovered["theta"], angle)
        assert np.isfinite(inv_log_det)

        jit_transform = jax.jit(lambda x: transform.transform(x))
        jit_inverse = jax.jit(lambda x: transform.inverse(x))

        # Test jitted forward transformation
        jitted_output, jitted_log_det = jit_transform(input_data)
        assert np.allclose(jitted_output["cos_theta"], jnp.cos(angle))
        assert np.isfinite(jitted_log_det)

        # Test jitted inverse transformation
        jitted_recovered, jitted_inv_log_det = jit_inverse(jitted_output)
        assert np.allclose(jitted_recovered["theta"], angle)
        assert np.isfinite(jitted_inv_log_det)

    def test_bound_to_bound(self):
        # Transform a value from an original range [0, 10] to a target range [100, 200].
        name_mapping = (["x"], ["x_mapped"])
        orig_lower = jnp.array([0.0])
        orig_upper = jnp.array([10.0])
        target_lower = jnp.array([100.0])
        target_upper = jnp.array([200.0])
        transform = BoundToBound(
            name_mapping, orig_lower, orig_upper, target_lower, target_upper
        )
        input_data = {"x": 5.0}  # mid-point of original range

        # Expected: (5-0)*(200-100)/(10-0) + 100 = 150
        expected_forward = 150.0
        output, log_det = transform.transform(input_data.copy())
        assert np.allclose(output["x_mapped"], expected_forward)
        # For one dimension, derivative is constant:
        expected_log_det = jnp.log(
            (target_upper - target_lower) / (orig_upper - orig_lower)
        )
        assert np.allclose(log_det, expected_log_det)

        recovered, inv_log_det = transform.inverse(output.copy())
        assert np.allclose(recovered["x"], input_data["x"])
        assert np.allclose(inv_log_det, -expected_log_det)

        # JIT compiled version
        jit_transform = jax.jit(lambda x: transform.transform(x))
        jit_inverse = jax.jit(lambda x: transform.inverse(x))
        jitted_output, jitted_log_det = jit_transform(input_data)
        assert np.allclose(jitted_output["x_mapped"], expected_forward)
        assert np.allclose(jitted_log_det, expected_log_det)
        jitted_recovered, jitted_inv_log_det = jit_inverse(jitted_output)
        assert np.allclose(jitted_recovered["x"], input_data["x"])
        assert np.allclose(jitted_inv_log_det, -expected_log_det)

    def test_bound_to_unbound(self):
        # Transform a value from a bounded interval [0, 1] to an unbounded domain.
        name_mapping = (["p"], ["p_unbound"])
        transform = BoundToUnbound(name_mapping, 0.0, 1.0)
        input_data = {"p": 0.5}
        # Expected forward: logit((0.5-0)/(1-0)) = log(0.5/0.5) = 0.
        output, log_det = transform.transform(input_data.copy())
        assert np.allclose(output["p_unbound"], 0.0)
        assert np.isfinite(log_det)

        recovered, inv_log_det = transform.inverse(output.copy())
        assert np.allclose(recovered["p"], 0.5)
        assert np.isfinite(inv_log_det)

        # JIT compiled
        jit_transform = jax.jit(lambda x: transform.transform(x))
        jit_inverse = jax.jit(lambda x: transform.inverse(x))
        jitted_output, jitted_log_det = jit_transform(input_data)
        assert np.allclose(jitted_output["p_unbound"], 0.0)
        assert np.isfinite(jitted_log_det)
        jitted_recovered, jitted_inv_log_det = jit_inverse(jitted_output)
        assert np.allclose(jitted_recovered["p"], 0.5)
        assert np.isfinite(jitted_inv_log_det)

    def test_single_sided_unbound_transform(self):
        # Test a transform that maps from a lower-bound-limited input to an unbounded output.
        name_mapping = (["x"], ["x_unbound"])
        lower_bound = 10.0
        transform = SingleSidedUnboundTransform(name_mapping, lower_bound)
        input_data = {"x": 20.0}
        expected_forward = jnp.log(20.0 - lower_bound)  # log(10)
        output, log_det = transform.transform(input_data.copy())
        assert np.allclose(output["x_unbound"], expected_forward)
        assert np.isfinite(log_det)

        recovered, inv_log_det = transform.inverse(output.copy())
        assert np.allclose(recovered["x"], 20.0)
        assert np.isfinite(inv_log_det)

        jit_transform = jax.jit(lambda x: transform.transform(x))
        jit_inverse = jax.jit(lambda x: transform.inverse(x))
        jitted_output, jitted_log_det = jit_transform(input_data)
        assert np.allclose(jitted_output["x_unbound"], expected_forward)
        assert np.isfinite(jitted_log_det)
        jitted_recovered, jitted_inv_log_det = jit_inverse(jitted_output)
        assert np.allclose(jitted_recovered["x"], 20.0)
        assert np.isfinite(jitted_inv_log_det)

    def test_powerlaw_transform(self):
        # Test the branch with alpha == -1.0.
        name_mapping = (["x"], ["x_powerlaw"])
        xmin = 1.0
        xmax = 10.0
        alpha = -1.0
        transform = PowerLawTransform(name_mapping, xmin, xmax, alpha)
        # For input x, forward transform: xmin * exp(x * ln(xmax/xmin)).
        input_data = {"x": 0.5}
        expected_forward = xmin * jnp.exp(0.5 * jnp.log(xmax / xmin))
        output, log_det = transform.transform(input_data.copy())
        assert np.allclose(output["x_powerlaw"], expected_forward)
        assert np.isfinite(log_det)

        recovered, inv_log_det = transform.inverse(output.copy())
        assert np.allclose(recovered["x"], 0.5)
        assert np.isfinite(inv_log_det)

        jit_transform = jax.jit(lambda x: transform.transform(x))
        jit_inverse = jax.jit(lambda x: transform.inverse(x))
        jitted_output, jitted_log_det = jit_transform(input_data)
        assert np.allclose(jitted_output["x_powerlaw"], expected_forward)
        assert np.isfinite(jitted_log_det)
        jitted_recovered, jitted_inv_log_det = jit_inverse(jitted_output)
        assert np.allclose(jitted_recovered["x"], 0.5)
        assert np.isfinite(jitted_inv_log_det)

    def test_powerlaw_transformn(self):
        # Test the branch with alpha != -1.0 (e.g. alpha = 1.0).
        name_mapping = (["x"], ["x_powerlaw"])
        xmin = 1.0
        xmax = 10.0
        alpha = 1.0
        transform = PowerLawTransform(name_mapping, xmin, xmax, alpha)
        # For input x, forward transform:
        # output = (xmin^(1+alpha) + x*(xmax^(1+alpha) - xmin^(1+alpha)))^(1/(1+alpha))
        input_data = {"x": 0.5}
        inner = xmin ** (1.0 + alpha) + 0.5 * (
            xmax ** (1.0 + alpha) - xmin ** (1.0 + alpha)
        )
        expected_forward = inner ** (1.0 / (1.0 + alpha))
        output, log_det = transform.transform(input_data.copy())
        assert np.allclose(output["x_powerlaw"], expected_forward)
        assert np.isfinite(log_det)

        recovered, inv_log_det = transform.inverse(output.copy())
        assert np.allclose(recovered["x"], input_data["x"])
        assert np.isfinite(inv_log_det)

        jit_transform = jax.jit(lambda x: transform.transform(x))
        jit_inverse = jax.jit(lambda x: transform.inverse(x))
        jitted_output, jitted_log_det = jit_transform(input_data)
        assert np.allclose(jitted_output["x_powerlaw"], expected_forward)
        assert np.isfinite(jitted_log_det)
        jitted_recovered, jitted_inv_log_det = jit_inverse(jitted_output)
        assert np.allclose(jitted_recovered["x"], input_data["x"])
        assert np.isfinite(jitted_inv_log_det)


class TestDistanceTransform:
    def test_forward_distance_transform(self):
        """
        Test transformation from distance to SNR-weighted distance (boundaries excluded)
        """
        output, jacobian = DistanceToSNRWeightedDistanceTransform(
            gps_time=1126259462.4,
            ifos=[H1, L1],
            dL_min=1.0,
            dL_max=2000.0,
        ).transform(
            {
                "d_L": 200.0,
                "M_c": 30.0,
                "ra": 1.0,
                "dec": 0.0,
                "psi": 0.5,
                "iota": 0.6,
            }
        )

        assert np.isfinite(output["d_hat_unbounded"])
        assert not jnp.isnan(jacobian).any()

    def test_forward_distance_transform_at_boundaries(self):
        """
        Test transformation from distance to SNR-weighted distance at boundaries (dL_min, dL_max)
        """
        output, jacobian = DistanceToSNRWeightedDistanceTransform(
            gps_time=1126259462.4,
            ifos=[H1, L1],
            dL_min=1.0,
            dL_max=2000.0,
        ).transform(
            {
                "d_L": 1.0,
                "M_c": 30.0,
                "ra": 1.0,
                "dec": 0.0,
                "psi": 0.5,
                "iota": 0.6,
            }
        )
        assert jnp.allclose(output["d_hat_unbounded"], -jnp.inf)

    def test_backward_distance_transform(self):
        """
        Test transformation from SNR-weighted distance to distance (boundaries excluded)
        """
        output, jacobian = DistanceToSNRWeightedDistanceTransform(
            gps_time=1126259462.4,
            ifos=[H1, L1],
            dL_min=1.0,
            dL_max=2000.0,
        ).inverse(
            {
                "d_hat_unbounded": 100.0,
                "M_c": 30.0,
                "ra": 1.0,
                "dec": 0.0,
                "psi": 0.5,
                "iota": 0.6,
            }
        )
        assert np.isfinite(output["d_L"])
        assert not jnp.isnan(jacobian).any()

    def test_backward_distance_transform_at_boundaries(self):
        """
        Test transformation from SNR-weighted distance to distance at boundaries (dL_min, dL_max)
        """
        output, jacobian = DistanceToSNRWeightedDistanceTransform(
            gps_time=1126259462.4,
            ifos=[H1, L1],
            dL_min=1.0,
            dL_max=2000.0,
        ).inverse(
            {
                "d_hat_unbounded": -jnp.inf,
                "M_c": 30.0,
                "ra": 1.0,
                "dec": 0.0,
                "psi": 0.5,
                "iota": 0.6,
            }
        )
        assert jnp.allclose(output["d_L"], 1.0)

    def test_forward_backward_consistency(self):
        """
        Test that the forward and inverse transformations are consistent
        """

        key = jax.random.PRNGKey(42)
        key, *subkeys = jax.random.split(key, 7)
        dL = jax.random.uniform(subkeys[0], (10,), minval=1, maxval=2000)
        M_c = jax.random.uniform(subkeys[1], (10,), minval=1, maxval=100)
        ra = jax.random.uniform(subkeys[2], (10,), minval=0, maxval=2 * jnp.pi)
        dec = jax.random.uniform(
            subkeys[3], (10,), minval=-jnp.pi / 2, maxval=jnp.pi / 2
        )
        psi = jax.random.uniform(subkeys[4], (10,), minval=0, maxval=jnp.pi)
        iota = jax.random.uniform(subkeys[5], (10,), minval=0, maxval=jnp.pi)

        inputs = jnp.stack([dL, M_c, ra, dec, psi, iota], axis=-1).T
        param_name = ["d_L", "M_c", "ra", "dec", "psi", "iota"]
        inputs = dict(zip(param_name, inputs))
        distance_transform = DistanceToSNRWeightedDistanceTransform(
            gps_time=1126259462.4,
            ifos=[H1, L1],
            dL_min=1.0,
            dL_max=2000.0,
        )
        forward_transform_output, _ = jax.vmap(distance_transform.transform)(inputs)
        output, _ = jax.vmap(distance_transform.inverse)(forward_transform_output)
        assert jnp.allclose(output["d_L"], dL)
        # default atol: 1e-8, rtol: 1e-5

    def test_jitted_forward_transform(self):
        """
        Test that the forward transformation is JIT compilable
        """

        # Generate random sample
        subkeys = jax.random.split(jax.random.PRNGKey(12), 6)
        dL = jax.random.uniform(subkeys[0], (1,), minval=1, maxval=2000)
        M_c = jax.random.uniform(subkeys[1], (1,), minval=1, maxval=100)
        ra = jax.random.uniform(subkeys[2], (1,), minval=0, maxval=2 * jnp.pi)
        dec = jax.random.uniform(
            subkeys[3], (1,), minval=-jnp.pi / 2, maxval=jnp.pi / 2
        )
        psi = jax.random.uniform(subkeys[4], (1,), minval=0, maxval=jnp.pi)
        iota = jax.random.uniform(subkeys[5], (1,), minval=0, maxval=jnp.pi)

        sample = [
            dL[0],
            M_c[0],
            ra[0],
            dec[0],
            psi[0],
            iota[0],
        ]
        sample_dict = dict(zip(["d_L", "M_c", "ra", "dec", "psi", "iota"], sample))

        # Create a JIT compiled version of the transform.
        jit_transform = jax.jit(
            lambda data: DistanceToSNRWeightedDistanceTransform(
                gps_time=1126259462.4,
                ifos=[H1, L1],
                dL_min=1.0,
                dL_max=2000.0,
            ).transform(data)
        )
        jitted_output, jitted_jacobian = jit_transform(sample_dict)
        non_jitted_output, _ = DistanceToSNRWeightedDistanceTransform(
            gps_time=1126259462.4,
            ifos=[H1, L1],
            dL_min=1.0,
            dL_max=2000.0,
        ).transform(sample_dict)

        # Assert that the jitted and non-jitted results agree
        assert common_keys_allclose(jitted_output, non_jitted_output)

        # Also check that the jitted jacobian contains no NaNs
        assert not jnp.isnan(jitted_jacobian).any()

    def test_jitted_backward_transform(self):
        """
        Test that the backward transformation is JIT compilable
        """

        # Generate random sample
        subkeys = jax.random.split(jax.random.PRNGKey(123), 6)
        d_hat_unbounded = jax.random.uniform(subkeys[0], (1,), minval=1, maxval=100000)
        M_c = jax.random.uniform(subkeys[1], (1,), minval=1, maxval=100)
        ra = jax.random.uniform(subkeys[2], (1,), minval=0, maxval=2 * jnp.pi)
        dec = jax.random.uniform(
            subkeys[3], (1,), minval=-jnp.pi / 2, maxval=jnp.pi / 2
        )
        psi = jax.random.uniform(subkeys[4], (1,), minval=0, maxval=jnp.pi)
        iota = jax.random.uniform(subkeys[5], (1,), minval=0, maxval=jnp.pi)

        sample = [
            d_hat_unbounded[0],
            M_c[0],
            ra[0],
            dec[0],
            psi[0],
            iota[0],
        ]
        sample_dict = dict(
            zip(["d_hat_unbounded", "M_c", "ra", "dec", "psi", "iota"], sample)
        )

        # Create a JIT compiled version of the transform.
        jit_inverse_transform = jax.jit(
            lambda data: DistanceToSNRWeightedDistanceTransform(
                gps_time=1126259462.4,
                ifos=[H1, L1],
                dL_min=1.0,
                dL_max=2000.0,
            ).inverse(data)
        )
        jitted_output, jitted_jacobian = jit_inverse_transform(sample_dict)
        non_jitted_output, _ = DistanceToSNRWeightedDistanceTransform(
            gps_time=1126259462.4,
            ifos=[H1, L1],
            dL_min=1.0,
            dL_max=2000.0,
        ).inverse(sample_dict)

        assert common_keys_allclose(jitted_output, non_jitted_output)
        # Also check that the jitted jacobian contains no NaNs
        assert not jnp.isnan(jitted_jacobian).any()


class TestSphereSpinToCartesianSpinTransform:
    def test_forward_transform(self):
        """
        Test the forward transformation from spherical to Cartesian spin components
        """
        output, jacobian = SphereSpinToCartesianSpinTransform("s1").transform(
            {
                "s1_mag": 0.4,
                "s1_theta": jnp.pi,
                "s1_phi": 0.8,
            }
        )
        assert (
            np.isfinite(output["s1_x"])
            & np.isfinite(output["s1_y"])
            & np.isfinite(output["s1_z"])
        )
        assert not jnp.isnan(jacobian).any()

    def test_backward_transform(self):
        """
        Test the backward transformation from Cartesian to spherical spin components
        """
        output, jacobian = SphereSpinToCartesianSpinTransform("s1").inverse(
            {
                "s1_x": 0.4,
                "s1_y": 0.5,
                "s1_z": 0.6,
            }
        )
        assert (
            np.isfinite(output["s1_mag"])
            & np.isfinite(output["s1_theta"])
            & np.isfinite(output["s1_phi"])
        )
        assert not jnp.isnan(jacobian).any()

    def test_forward_backward_consistency(self):
        """
        Test that the forward and inverse transformations are consistent
        """

        key = jax.random.PRNGKey(42)
        key, *subkeys = jax.random.split(key, 4)
        s1_mag = jax.random.uniform(subkeys[0], (10,), minval=1e-3, maxval=1.0)
        s1_theta = jax.random.uniform(subkeys[1], (10,), minval=0, maxval=jnp.pi)
        s1_phi = jax.random.uniform(
            subkeys[2], (10,), minval=0, maxval=2 * jnp.pi
        )  # [0, 2*pi]

        inputs = {'s1_mag': s1_mag, 's1_theta': s1_theta, 's1_phi': s1_phi}
        transform = SphereSpinToCartesianSpinTransform("s1")
        forward_transform_output, _ = jax.vmap(transform.transform)(inputs)
        output, _ = jax.vmap(transform.inverse)(forward_transform_output)

        assert common_keys_allclose(output, inputs)
        # default atol: 1e-8, rtol: 1e-5

    def test_jitted_forward_transform(self):
        """
        Test that the forward transformation is JIT compilable
        """

        # Generate random sample
        subkeys = jax.random.split(jax.random.PRNGKey(12), 3)
        s1_mag = jax.random.uniform(subkeys[0], (1,), minval=1e-3, maxval=1.0)
        s1_theta = jax.random.uniform(subkeys[1], (1,), minval=0, maxval=jnp.pi)
        s1_phi = jax.random.uniform(
            subkeys[2], (1,), minval=0, maxval=2 * jnp.pi
        )  # [0, 2*pi]

        sample_dict = \
            {'s1_mag': s1_mag[0], 's1_theta': s1_theta[0], 's1_phi': s1_phi[0]}

        # Create a JIT compiled version of the transform.
        jit_transform = jax.jit(
            lambda data: SphereSpinToCartesianSpinTransform("s1").transform(data)
        )
        jitted_output, jitted_jacobian = jit_transform(sample_dict)
        non_jitted_output, _ = SphereSpinToCartesianSpinTransform("s1").transform(
            sample_dict
        )

        # Assert that the jitted and non-jitted results agree
        assert common_keys_allclose(jitted_output, non_jitted_output)
        # Also check that the jitted jacobian contains no NaNs
        assert not jnp.isnan(jitted_jacobian).any()

    def test_jitted_backward_transform(self):
        """
        Test that the backward transformation is JIT compilable
        """

        # Generate random sample
        keys = jax.random.split(jax.random.PRNGKey(123), 2)
        S1 = jax.random.uniform(keys[0], (3,), minval=-1, maxval=1)
        a1 = jax.random.uniform(keys[1], (1,), minval=1e-3, maxval=1.0)
        S1 *= a1 / jnp.linalg.norm(S1)

        sample_dict = dict(zip(["s1_x", "s1_y", "s1_z"], S1))

        # Create a JIT compiled version of the transform.
        jit_inverse_transform = jax.jit(
            lambda data: SphereSpinToCartesianSpinTransform("s1").inverse(data)
        )
        jitted_output, jitted_jacobian = jit_inverse_transform(sample_dict)
        non_jitted_output, _ = SphereSpinToCartesianSpinTransform("s1").inverse(
            sample_dict
        )

        # Assert that the jitted and non-jitted results agree
        assert common_keys_allclose(jitted_output, non_jitted_output)
        # Also check that the jitted jacobian contains no NaNs
        assert not jnp.isnan(jitted_jacobian).any()


class TestSpinAnglesToCartesianSpinTransform:
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

        """
        input_dir = Path("test/unit/source_files/spin_angles_input")
        output_dir = Path("test/unit/source_files/cartesian_spins_output_for_bilby")
        input_files = input_dir.glob("*.npz")

        for file in input_files:
            input_dict = dict(jnp.load(file).items())
            fRef = int(input_dict.pop("fRef")[0])
            print("Testing forward transform for fRef =", fRef)

            m_1 = input_dict.pop("m_1")
            m_2 = input_dict.pop("m_2")
            M_c, q = m1_m2_to_Mc_q(m_1, m_2)
            input_dict["M_c"] = M_c
            input_dict["q"] = q

            # read outputs from binary
            output_file = list(output_dir.glob(f"*fRef_{fRef}*.npz"))[0]
            bilby_spins = dict(jnp.load(output_file).items())

            transform = SpinAnglesToCartesianSpinTransform(freq_ref=fRef)
            jimgw_spins, jacobian = jax.vmap(transform.transform)(input_dict)

            assert common_keys_allclose(jimgw_spins, bilby_spins)
            assert not jnp.isnan(jacobian).any()

    def test_backward_spin_transform(self):
        """
        Test transformation from cartesian spins to spin angles

        """
        input_dir = Path("test/unit/source_files/cartesian_spins_input")
        output_dir = Path("test/unit/source_files/spin_angles_output_for_bilby")
        input_files = input_dir.glob("*.npz")

        for file in input_files:
            input_dict = dict(jnp.load(file).items())
            fRef = int(input_dict.pop("fRef")[0])
            print("Testing backward transform for fRef =", fRef)

            m_1 = input_dict.pop("m_1")
            m_2 = input_dict.pop("m_2")
            M_c, q = m1_m2_to_Mc_q(m_1, m_2)
            input_dict["M_c"] = M_c
            input_dict["q"] = q

            # read outputs from binary
            output_file = list(output_dir.glob(f"*fRef_{fRef}*.npz"))[0]
            bilby_spins = dict(jnp.load(output_file).items())

            transform = SpinAnglesToCartesianSpinTransform(freq_ref=fRef)
            jimgw_spins, jacobian = jax.vmap(transform.inverse)(input_dict)

            # default atol: 1e-8, rtol: 1e-5
            assert common_keys_allclose(jimgw_spins, bilby_spins)
            assert not jnp.isnan(jacobian).any()

    def test_forward_backward_consistency(self):
        """
        Test that the forward and inverse transformations are consistent
        """

        n = 10
        key = jax.random.PRNGKey(42)
        subkeys = jax.random.split(key, 7)

        S1, S2 = jax.random.uniform(subkeys[0], (2, 3, n), minval=-1, maxval=1)
        a1, a2 = jax.random.uniform(subkeys[1], (2, n), minval=1e-3, maxval=1)
        S1 *= a1 / jnp.linalg.norm(S1, axis=0)
        S2 *= a2 / jnp.linalg.norm(S2, axis=0)

        samples = jnp.array(
            [
                jax.random.uniform(subkeys[2], (n,), minval=0, maxval=jnp.pi),  # iota
                *S1,
                *S2,
                jax.random.uniform(subkeys[3], (n,), minval=1, maxval=100),  # M_c
                jax.random.uniform(subkeys[4], (n,), minval=0.125, maxval=1),  # q
                jax.random.uniform(
                    subkeys[5], (n,), minval=0, maxval=2 * jnp.pi
                ),  # phase_c
            ]
        ).T
        fRefs = jax.random.uniform(subkeys[6], (n,), minval=10, maxval=100)

        for fRef, sample in zip(fRefs, samples):
            jimgw_spins, _ = SpinAnglesToCartesianSpinTransform(freq_ref=fRef).inverse(
                dict(zip(self.backward_keys, sample))
            )
            jimgw_spins, _ = SpinAnglesToCartesianSpinTransform(
                freq_ref=fRef
            ).transform(jimgw_spins)
            jimgw_spins = jnp.array(list(jimgw_spins.values()))
            # default atol: 1e-8, rtol: 1e-5
            assert jnp.allclose(jimgw_spins, jnp.array([*sample[7:], *sample[:7]]))

    def test_jitted_forward_transform(self):
        """
        Test that the forward transformation is JIT compilable
        """

        # Generate random sample
        subkeys = jax.random.split(jax.random.PRNGKey(12), 8)
        iota = jax.random.uniform(subkeys[0], (1,), minval=0, maxval=jnp.pi)
        M_c = jax.random.uniform(subkeys[1], (1,), minval=1, maxval=100)
        q = jax.random.uniform(subkeys[2], (1,), minval=0.125, maxval=1)
        fRef = jax.random.uniform(subkeys[3], (1,), minval=10, maxval=100)
        phiRef = jax.random.uniform(subkeys[4], (1,), minval=0, maxval=2 * jnp.pi)

        S1, S2 = jax.random.uniform(subkeys[5], (2, 3), minval=-1, maxval=1)
        a1, a2 = jax.random.uniform(subkeys[6], (2,), minval=1e-3, maxval=1)
        S1 *= a1 / jnp.linalg.norm(S1)
        S2 *= a2 / jnp.linalg.norm(S2)

        sample = [
            iota[0],
            *S1,
            *S2,
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
        non_jitted_spins, _ = SpinAnglesToCartesianSpinTransform(
            freq_ref=freq_ref_sample
        ).transform(sample_dict)

        assert common_keys_allclose(jitted_spins, non_jitted_spins)
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
        non_jitted_spins, _ = SpinAnglesToCartesianSpinTransform(
            freq_ref=freq_ref_sample
        ).inverse(sample_dict)

        assert common_keys_allclose(jitted_spins, non_jitted_spins)
        # Also check that the jitted jacobian contains no NaNs
        assert not jnp.isnan(jitted_jacobian).any()


class TestSkyFrameToDetectorFrameSkyPositionTransform:
    # Since Bilby and Jim use different algorithm for gmst, which
    # induces errors to RA, and this is the minimum precision that passes
    RA_ATOL = 5e-5

    def test_backward_transform(self):
        """
        Test the backward transformation from spherical to detector frame sky position with different sets of detectors
        """
        input_dir = Path("test/unit/source_files/sky_locations")
        files = input_dir.glob("*.npz")
        for file in files:
            data = jnp.load(file, allow_pickle=True)

            bilby_outputs = {key: data[key] for key in ("ra", "dec")}
            zenith_azimuth = {key: data[key] for key in ("zenith", "azimuth")}
            ifos = [detector_preset[ifo_name] for ifo_name in data["ifo_pair"]]

            transform = SkyFrameToDetectorFrameSkyPositionTransform(
                gps_time=data["gps_time"],
                ifos=ifos,
            )
            # test the forward transform
            jim_outputs, jacobian = jax.vmap(transform.inverse)(zenith_azimuth)

            # Use the tailored atol for RA
            assert jnp.allclose(
                jim_outputs["ra"], bilby_outputs["ra"], atol=self.RA_ATOL
            )
            assert jnp.allclose(jim_outputs["dec"], bilby_outputs["dec"])
            assert not jnp.isnan(jacobian).any()

    def test_forward_backward_consistency(self):
        """
        Test that the forward and inverse transformations are consistent
        """
        ifos = [H1, L1, V1]
        gps_time = [1126259642.413, 1242442967.4]
        key = jax.random.PRNGKey(42)

        for ifo_pair in combinations(ifos, 2):
            for time in gps_time:
                key, *subkeys = jax.random.split(key, 3)
                inputs = {
                    "zenith": jax.random.uniform(
                        subkeys[0], (10,), minval=0, maxval=jnp.pi
                    ),
                    "azimuth": jax.random.uniform(
                        subkeys[1], (10,), minval=0, maxval=2 * jnp.pi
                    ),
                }
                transform = SkyFrameToDetectorFrameSkyPositionTransform(
                    gps_time=time,
                    ifos=list(ifo_pair),
                )
                forward_transform_output, _ = jax.vmap(transform.inverse)(inputs)
                outputs, _ = jax.vmap(transform.transform)(forward_transform_output)

                # default atol: 1e-8, rtol: 1e-5
                assert common_keys_allclose(outputs, inputs)
                # best accuracy for zenith and azimuth is 1e-16

    def test_jitted_forward_transform(self):
        """
        Test that the forward transformation is JIT compilable
        """
        # Generate random sample
        subkeys = jax.random.split(jax.random.PRNGKey(12), 2)
        sample_dict = {
            "ra": jax.random.uniform(subkeys[0], (1,), minval=0, maxval=2 * jnp.pi)[0],
            "dec": jax.random.uniform(subkeys[1], (1,), minval=0, maxval=jnp.pi)[0],
        }
        class_args = dict(gps_time=1126259642.4, ifos=[H1, L1])

        # Create a JIT compiled version of the transform.
        jit_transform = jax.jit(
            lambda data: SkyFrameToDetectorFrameSkyPositionTransform(
                **class_args
            ).transform(data)
        )
        jitted_output, jitted_jacobian = jit_transform(sample_dict)
        non_jitted_output, _ = SkyFrameToDetectorFrameSkyPositionTransform(
            **class_args
        ).transform(sample_dict)

        # Assert that the jitted and non-jitted results agree
        assert common_keys_allclose(jitted_output, non_jitted_output)
        # Also check that the jitted jacobian contains no NaNs
        assert not jnp.isnan(jitted_jacobian).any()

    def test_jitted_backward_transform(self):
        """
        Test that the backward transformation is JIT compilable
        """

        # Generate random sample
        subkeys = jax.random.split(jax.random.PRNGKey(123), 2)
        sample_dict = {
            "zenith": jax.random.uniform(subkeys[0], (1,), minval=0, maxval=jnp.pi)[0],
            "azimuth": jax.random.uniform(
                subkeys[1], (1,), minval=0, maxval=2 * jnp.pi
            )[0],
        }
        class_args = dict(gps_time=1126259642.4, ifos=[H1, L1])

        # Create a JIT compiled version of the transform.
        jit_inverse_transform = jax.jit(
            lambda data: SkyFrameToDetectorFrameSkyPositionTransform(
                **class_args
            ).inverse(data)
        )
        jitted_output, jitted_jacobian = jit_inverse_transform(sample_dict)
        non_jitted_output, _ = SkyFrameToDetectorFrameSkyPositionTransform(
            **class_args
        ).inverse(sample_dict)

        # Assert that the jitted and non-jitted results agree
        assert common_keys_allclose(jitted_output, non_jitted_output)
        # Also check that the jitted jacobian contains no NaNs
        assert not jnp.isnan(jitted_jacobian).any()


class TestHelperFunctions:
    def test_reverse_bijective_transform(self):
        # Test the reverse_bijective_transform function by applying it to a simple ScaleTransform.
        name_mapping = (["a", "b"], ["a_scaled", "b_scaled"])
        scale = 3.0
        original_transform = ScaleTransform(name_mapping, scale)
        input_data = {"a": 2.0, "b": 4.0}

        # Compute output using original transform.
        output, log_det = original_transform.transform(input_data.copy())
        # Obtain reversed transform.
        reversed_transform = reverse_bijective_transform(original_transform)
        # Now, using the reversed transform (which swaps the transform and its inverse),
        # applying its forward transformation on the original output should recover the original input.
        recovered, rev_log_det = reversed_transform.transform(output.copy())
        assert common_keys_allclose(recovered, input_data)
        assert np.allclose(rev_log_det, -log_det)
