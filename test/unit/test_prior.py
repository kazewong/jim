import jax
import jax.numpy as jnp
import scipy.stats as stats

from jimgw.prior import (
    LogisticDistribution,
    StandardNormalDistribution,
    UniformPrior,
    SinePrior,
    CosinePrior,
    UniformSpherePrior,
    PowerLawPrior,
    GaussianPrior,
    RayleighPrior,
    FullRangePrior,
    CombinePrior,
)

jax.config.update("jax_enable_x64", True)


def assert_all_finite(arr):
    """Assert all values in the array are finite."""
    assert jnp.all(jnp.isfinite(arr)), "Array contains non-finite values."


def assert_all_in_range(arr, low, high):
    assert jnp.all((arr > low) & (arr < high)), f"Values not in ({low}, {high})"


class TestUnivariatePrior:
    def test_logistic(self):
        """Test the LogisticDistribution prior."""
        p = LogisticDistribution(["x"])

        # Draw samples and check they are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert_all_finite(samples["x"])

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob matches scipy.stats.logistic
        x = jnp.linspace(-10.0, 10.0, 1000)
        assert jnp.allclose(
            jax.vmap(p.log_prob)(p.add_name(x[None])), stats.logistic.logpdf(x)
        )

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_val = jax.vmap(jitted_log_prob)(p.add_name(x[None]))
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(p.add_name(x[None])))

    def test_standard_normal(self):
        """Test the StandardNormalDistribution prior."""
        p = StandardNormalDistribution(["x"])

        # Draw samples and check they are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert_all_finite(samples["x"])

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob matches scipy.stats.norm
        x = jnp.linspace(-10.0, 10.0, 1000)
        assert jnp.allclose(
            jax.vmap(p.log_prob)(p.add_name(x[None])), stats.norm.logpdf(x)
        )

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_val = jax.vmap(jitted_log_prob)(p.add_name(x[None]))
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(p.add_name(x[None])))

    def test_uniform(self):
        """Test the UniformPrior prior."""
        xmin, xmax = -10.0, 10.0
        p = UniformPrior(xmin, xmax, ["x"])

        # Draw samples and check they are finite and in range
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert_all_finite(samples["x"])
        assert_all_in_range(samples["x"], xmin, xmax)

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob is correct in the support
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.transform)(x)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), -jnp.log(xmax - xmin))

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_val = jax.vmap(jitted_log_prob)(y)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_sine(self):
        """Test the SinePrior prior."""
        p = SinePrior(["x"])

        # Draw samples and check they are finite and in range
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert_all_finite(samples["x"])
        assert_all_in_range(samples["x"], 0.0, jnp.pi)

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob is correct in the support
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.base_prior[0].base_prior[0].transform)(x)
        y = jax.vmap(p.base_prior[0].transform)(y)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), jnp.log(jnp.sin(y["x"]) / 2.0))

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_val = jax.vmap(jitted_log_prob)(y)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_cosine(self):
        """Test the CosinePrior prior."""
        p = CosinePrior(["x"])

        # Draw samples and check they are finite and in range
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert_all_finite(samples["x"])
        assert_all_in_range(samples["x"], -jnp.pi / 2.0, jnp.pi / 2.0)

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob is correct in the support
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.base_prior[0].transform)(x)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), jnp.log(jnp.cos(y["x"]) / 2.0))

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_val = jax.vmap(jitted_log_prob)(y)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_uniform_sphere(self):
        """Test the UniformSpherePrior prior."""
        p = UniformSpherePrior(["x"])

        # Draw samples and check they are finite and in range
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert_all_finite(samples["x_mag"])
        assert_all_finite(samples["x_theta"])
        assert_all_finite(samples["x_phi"])
        assert_all_in_range(samples["x_mag"], 0.0, 1.0)
        assert_all_in_range(samples["x_theta"], 0.0, jnp.pi)
        assert_all_in_range(samples["x_phi"], 0.0, 2 * jnp.pi)

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

    def test_power_law(self):
        """Test the PowerLawPrior prior for various exponents."""
        xmin, xmax = 0.1, 100.0
        for alpha in [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]:
            alpha = float(alpha)
            p = PowerLawPrior(xmin, xmax, alpha, ["x"])

            # Draw samples and check they are finite and in range
            samples = p.sample(jax.random.PRNGKey(0), 10000)
            assert_all_finite(samples["x"])
            assert_all_in_range(samples["x"], xmin, xmax)

            # Check log_prob is finite for samples
            log_prob = jax.vmap(p.log_prob)(samples)
            assert_all_finite(log_prob)

            # Check log_prob is correct in the support
            x = p.trace_prior_parent([])[0].add_name(
                jnp.linspace(-10.0, 10.0, 1000)[None]
            )
            y = jax.vmap(p.transform)(x)
            if alpha < -1.0:
                expected = (
                    alpha * jnp.log(y["x"])
                    + jnp.log(-alpha - 1)
                    - jnp.log(xmin ** (alpha + 1) - xmax ** (alpha + 1))
                )
            elif alpha > -1.0:
                expected = (
                    alpha * jnp.log(y["x"])
                    + jnp.log(alpha + 1)
                    - jnp.log(xmax ** (alpha + 1) - xmin ** (alpha + 1))
                )
            else:
                expected = -jnp.log(y["x"]) - jnp.log(jnp.log(xmax) - jnp.log(xmin))
            assert jnp.allclose(jax.vmap(p.log_prob)(y), expected)

            # Check log_prob is jittable
            jitted_log_prob = jax.jit(p.log_prob)
            jitted_val = jax.vmap(jitted_log_prob)(y)
            assert_all_finite(jitted_val)
            assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_gaussian(self):
        """Test the GaussianPrior prior."""
        mu, sigma = 2.0, 3.0
        p = GaussianPrior(mu, sigma, ["x"])

        # Draw samples and check they are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert_all_finite(samples["x"])

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob is correct in the support
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.transform)(x)
        assert jnp.allclose(
            jax.vmap(p.log_prob)(y), stats.norm.logpdf(y["x"], loc=mu, scale=sigma)
        )

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_val = jax.vmap(jitted_log_prob)(y)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_Rayleigh(self):
        """Test the RayleighPrior prior."""
        sigma = 2.0
        p = RayleighPrior(sigma, ["x"])

        # Draw samples and check they are finite and positive
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert_all_finite(samples["x"])
        assert jnp.all(samples["x"] > 0.0)

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob is correct in the support
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(0.0, 10.0, 1000)[None])
        y = jax.vmap(p.base_prior[0].transform)(x)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(
            jax.vmap(p.log_prob)(y), stats.rayleigh.logpdf(y["x"], scale=sigma)
        )

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_val = jax.vmap(jitted_log_prob)(y)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))


class TestFullRangePrior:
    def test_full_range_prior_1d(self):
        """Test FullRangePrior for correct constraint enforcement and sampling."""
        base = UniformPrior(0.0, 1.0, ["x"])
        p = FullRangePrior(base)

        # Draw samples and check they are finite and in range
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert_all_in_range(samples["x"], 0.0, 1.0)

        # Check log_prob is finite for samples in range and -inf outside
        xs = jnp.linspace(-0.5, 1.5, 1000)
        xs_dict = p.add_name(xs[None])
        logp = jax.vmap(p.log_prob)(xs_dict)
        mask = (xs >= 0.0) & (xs <= 1.0)
        assert_all_finite(logp[mask])
        assert jnp.all(logp[~mask] == -jnp.inf)

        # Check log_prob matches base prior
        base_logp = jax.vmap(base.log_prob)(xs_dict)
        assert jnp.allclose(logp[mask], base_logp[mask])

        # Add extra constraint
        p2 = FullRangePrior(base, extra_constraints=[lambda z: z["x"] < 0.5])

        # Draw samples and check they are finite and in range
        samples2 = p2.sample(jax.random.PRNGKey(1), 10000)
        assert_all_in_range(samples2["x"], 0.0, 0.5)

        # Check log_prob is finite for samples in range and -inf outside
        xs2 = jnp.linspace(-0.5, 1.5, 1000)
        xs2_dict = p2.add_name(xs2[None])
        logp2 = jax.vmap(p2.log_prob)(xs2_dict)
        mask2 = (xs2 >= 0.0) & (xs2 < 0.5)
        assert_all_finite(logp2[mask2])
        assert jnp.all(logp2[~mask2] == -jnp.inf)

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_vals = jax.vmap(jitted_log_prob)(xs_dict)
        assert jnp.allclose(jitted_vals, logp)

    def test_full_range_prior_2d(self):
        """Test FullRangePrior for 2D priors with joint constraints and sampling."""
        # Create two independent UniformPriors
        base1 = UniformPrior(0.0, 1.0, ["x"])
        base2 = UniformPrior(-2.0, 2.0, ["y"])
        # Combine into a 2D prior
        base = CombinePrior([base1, base2])
        # No extra constraints: should behave like the product prior
        p = FullRangePrior(base)

        # Draw samples and check they are finite and in range
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert_all_in_range(samples["x"], 0.0, 1.0)
        assert_all_in_range(samples["y"], -2.0, 2.0)

        # log_prob should be finite in the box, -inf outside
        xs = jnp.linspace(-0.5, 1.5, 100)
        ys = jnp.linspace(-2.5, 2.5, 100)
        grid_x, grid_y = jnp.meshgrid(xs, ys, indexing="ij")
        flat_x = grid_x.ravel()
        flat_y = grid_y.ravel()
        xs_dict = {"x": flat_x, "y": flat_y}
        logp = jax.vmap(p.log_prob)(xs_dict)
        mask = (flat_x > 0.0) & (flat_x < 1.0) & (flat_y > -2.0) & (flat_y < 2.0)
        assert_all_finite(logp[mask])
        assert jnp.all(logp[~mask] == -jnp.inf)

        # Add a joint constraint: x + y < 1
        p2 = FullRangePrior(base, extra_constraints=[lambda z: z["x"] + z["y"] < 1.0])

        # Draw samples and check they are finite and in range
        samples2 = p2.sample(jax.random.PRNGKey(1), 10000)
        assert_all_in_range(samples2["x"], 0.0, 1.0)
        assert_all_in_range(samples2["y"], -2.0, 2.0)
        assert jnp.all(samples2["x"] + samples2["y"] < 1.0)

        # log_prob should be finite only where x + y < 1
        logp2 = jax.vmap(p2.log_prob)(xs_dict)
        mask2 = mask & (flat_x + flat_y < 1.0)
        assert_all_finite(logp2[mask2])
        assert jnp.all(logp2[~mask2] == -jnp.inf)

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(p2.log_prob)
        jitted_vals = jax.vmap(jitted_log_prob)(xs_dict)
        assert jnp.allclose(jitted_vals, logp2)
