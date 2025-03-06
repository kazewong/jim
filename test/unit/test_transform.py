import jax
import jax.numpy as jnp

import numpy as np

from jimgw.transforms import (
    ScaleTransform,
    OffsetTransform,
    LogitTransform,
    SineTransform,
    CosineTransform,
)

jax.config.update("jax_enable_x64", True)

class TestBasicTransforms:
    def test_scale_transform(self):
        # Define a simple two-dimensional scale transform.
        name_mapping = (["a", "b"], ["a_scaled", "b_scaled"])
        scale = 3.0
        transform = ScaleTransform(name_mapping, scale)
        input_data = {"a": 2.0, "b": 4.0}

        # Apply the forward transformation.
        output, log_det = transform.transform(input_data.copy())
        expected_output = {"a_scaled": 2.0 * scale, "b_scaled": 4.0 * scale}
        # Check that the forward output matches expected values.
        assert np.allclose(list(output.values()), list(expected_output.values()))

        # For a two-dimensional scaling, the Jacobian determinant is scale^2.
        expected_log_det = 2 * jnp.log(scale)
        assert np.allclose(log_det, expected_log_det)

        # Now apply the inverse transformation to recover the original input.
        recovered, inv_log_det = transform.inverse(output.copy())
        assert np.allclose(list(recovered.values()), list(input_data.values()))
        # The inverse Jacobian should be the negative of the forward.
        assert np.allclose(inv_log_det, -expected_log_det)

    def test_offset_transform(self):
        name_mapping = (["x", "y"], ["x_offset", "y_offset"])
        offset = 5.0
        transform = OffsetTransform(name_mapping, offset)
        input_data = {"x": 10.0, "y": -3.0}

        # Forward transform.
        output, log_det = transform.transform(input_data.copy())
        expected_output = {"x_offset": 10.0 + offset, "y_offset": -3.0 + offset}
        assert np.allclose(list(output.values()), list(expected_output.values()))
        # Since addition is volume-preserving, the log determinant should be zero.
        assert np.allclose(log_det, 0.0)

        # Inverse transform.
        recovered, inv_log_det = transform.inverse(output.copy())
        assert np.allclose(list(recovered.values()), list(input_data.values()))
        assert np.allclose(inv_log_det, 0.0)

    def test_logit_transform(self):
        name_mapping = (["p"], ["p_logit"])
        transform = LogitTransform(name_mapping)
        # Choose a value strictly between 0 and 1.
        input_data = {"p": 0.6}

        # Forward transform: the transform computes 1 / (1 + exp(-p)).
        output, log_det = transform.transform(input_data.copy())
        expected_forward = 1 / (1 + jnp.exp(-0.6))
        assert np.allclose(output["p_logit"], expected_forward)

        # Inverse transform should recover the original p.
        recovered, inv_log_det = transform.inverse(output.copy())
        assert np.allclose(recovered["p"], 0.6)

    def test_sine_transform(self):
        name_mapping = (["theta"], ["sin_theta"])
        transform = SineTransform(name_mapping)
        # Use an angle in [-pi/2, pi/2].
        angle = 0.3
        input_data = {"theta": angle}

        output, log_det = transform.transform(input_data.copy())
        expected = jnp.sin(angle)
        assert np.allclose(output["sin_theta"], expected)

        recovered, inv_log_det = transform.inverse(output.copy())
        assert np.allclose(recovered["theta"], angle)

    def test_cosine_transform(self):
        name_mapping = (["theta"], ["cos_theta"])
        transform = CosineTransform(name_mapping)
        # Use an angle in [0, pi].
        angle = 1.2
        input_data = {"theta": angle}

        output, log_det = transform.transform(input_data.copy())
        expected = jnp.cos(angle)
        assert np.allclose(output["cos_theta"], expected)

        recovered, inv_log_det = transform.inverse(output.copy())
        assert np.allclose(recovered["theta"], angle)
