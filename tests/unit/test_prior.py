import jax
import jax.numpy as jnp
import scipy.stats as stats

from jimgw.core.prior import (
    LogisticDistribution,
    StandardNormalDistribution,
    UniformDistribution,
    UniformPrior,
    SinePrior,
    CosinePrior,
    UniformSpherePrior,
    PowerLawPrior,
    GaussianPrior,
    NormalPrior,
    RayleighPrior,
    BoundedMixin,
    SequentialTransformPrior,
)

jax.config.update("jax_enable_x64", True)


def assert_all_finite(arr):
    """Assert all values in the array are finite."""
    assert jnp.all(jnp.isfinite(arr)), "Array contains non-finite values."


def assert_all_in_range(arr, low, high):
    assert jnp.all((arr >= low) & (arr <= high)), f"Values not in [{low}, {high}]"


class TestUnivariatePrior:
    def test_logistic_distribution(self):
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
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(p.add_name(x[None]))
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(p.add_name(x[None])))

    def test_standard_normal_distribution(self):
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
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(p.add_name(x[None]))
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(p.add_name(x[None])))

    def test_uniform_distribution(self):
        """Test the UniformDistribution prior."""
        p = UniformDistribution(["x"])
        xmin, xmax = p.xmin, p.xmax  # 0.0, 1.0

        # Draw samples and check they are finite and in range
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert_all_finite(samples["x"])
        assert_all_in_range(samples["x"], xmin, xmax)

        # Check log_prob is finite for samples
        log_prob = jax.vmap(p.log_prob)(samples)
        assert_all_finite(log_prob)

        # Check log_prob is correct in the support
        x = p.add_name(jnp.linspace(xmin, xmax, 1000)[None])
        assert jnp.allclose(
            jax.vmap(p.log_prob)(x), -jnp.log(xmax - xmin) * jnp.ones_like(x["x"])
        )

        # Check log_prob is -inf outside the support
        x_outside = p.add_name(jnp.array([xmin - 1.0, xmax + 1.0])[None])
        logp_outside = jax.vmap(p.log_prob)(x_outside)
        assert jnp.all(logp_outside == -jnp.inf)

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(x)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(x))

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

        # Check log_prob is correct in the support (use valid range [0, 1] for base)
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(0.0, 1.0, 1000)[None])
        y = jax.vmap(p.transform)(x)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), -jnp.log(xmax - xmin))

        # Check log_prob is -inf outside the support
        x_outside = p.add_name(jnp.array([xmin - 1.0, xmax + 1.0])[None])
        logp_outside = jax.vmap(p.log_prob)(x_outside)
        assert jnp.all(logp_outside == -jnp.inf)

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(y)
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

        # Check log_prob is correct in the support (use valid range (0, 1) exclusive
        # to avoid boundaries where sin(x) = 0 gives log_prob = -inf)
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(0.001, 0.999, 1000)[None])
        y = jax.vmap(p.base_prior[0].transform)(x)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), jnp.log(jnp.sin(y["x"]) / 2.0))

        # Check log_prob is -inf outside the support
        x_outside = p.add_name(jnp.array([-0.5, jnp.pi + 0.5])[None])
        logp_outside = jax.vmap(p.log_prob)(x_outside)
        assert jnp.all(logp_outside == -jnp.inf)

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(y)
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

        # Check log_prob is correct in the support (use valid range (0, 1) exclusive
        # to avoid boundaries where cos(x) = 0 gives log_prob = -inf)
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(0.001, 0.999, 1000)[None])
        y = jax.vmap(p.base_prior[0].base_prior[0].transform)(x)
        y = jax.vmap(p.base_prior[0].transform)(y)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), jnp.log(jnp.cos(y["x"]) / 2.0))

        # Check log_prob is -inf outside the support
        x_outside = p.add_name(
            jnp.array([-jnp.pi / 2.0 - 0.5, jnp.pi / 2.0 + 0.5])[None]
        )
        logp_outside = jax.vmap(p.log_prob)(x_outside)
        assert jnp.all(logp_outside == -jnp.inf)

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(y)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

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

            # Check log_prob is correct in the support (use valid range (0, 1] for base,
            # excluding 0 as it maps to the boundary where log_prob = -inf)
            x = p.trace_prior_parent([])[0].add_name(
                jnp.linspace(0.001, 1.0, 1000)[None]
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

            # Check log_prob is -inf outside the support
            x_outside = p.add_name(jnp.array([xmin - 0.01, xmax + 1.0])[None])
            logp_outside = jax.vmap(p.log_prob)(x_outside)
            assert jnp.all(logp_outside == -jnp.inf)

            # Check log_prob is jittable
            jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
            jitted_val = jitted_log_prob(y)
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
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(y)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_normal_synonym(self):
        """Test that NormalPrior is a synonym for GaussianPrior."""
        mu, sigma = 2.0, 3.0
        p_gaussian = GaussianPrior(mu, sigma, ["x"])
        p_normal = NormalPrior(mu, sigma, ["x"])

        # Draw samples from both and check they have same distribution
        rng_key = jax.random.PRNGKey(42)
        samples_gaussian = p_gaussian.sample(rng_key, 10000)
        samples_normal = p_normal.sample(rng_key, 10000)

        assert_all_finite(samples_gaussian["x"])
        assert_all_finite(samples_normal["x"])

        # Check log_prob is identical for both
        x = p_gaussian.trace_prior_parent([])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p_gaussian.transform)(x)

        logprob_gaussian = jax.vmap(p_gaussian.log_prob)(y)
        logprob_normal = jax.vmap(p_normal.log_prob)(y)

        assert jnp.allclose(logprob_gaussian, logprob_normal)
        assert jnp.allclose(
            logprob_normal, stats.norm.logpdf(y["x"], loc=mu, scale=sigma)
        )

    def test_standard_normal_is_gaussian(self):
        """Test that StandardNormalDistribution is GaussianPrior(mu=0, sigma=1)."""
        p_standard = StandardNormalDistribution(["x"])
        p_gaussian = GaussianPrior(0.0, 1.0, ["x"])

        # Draw samples from both
        rng_key = jax.random.PRNGKey(123)
        samples_standard = p_standard.sample(rng_key, 10000)
        samples_gaussian = p_gaussian.sample(rng_key, 10000)

        assert_all_finite(samples_standard["x"])
        assert_all_finite(samples_gaussian["x"])

        # Check log_prob is identical for both
        x = p_standard.trace_prior_parent([])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p_standard.transform)(x)

        logprob_standard = jax.vmap(p_standard.log_prob)(y)
        logprob_gaussian = jax.vmap(p_gaussian.log_prob)(y)

        assert jnp.allclose(logprob_standard, logprob_gaussian)
        assert jnp.allclose(logprob_standard, stats.norm.logpdf(y["x"]))

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

        # Check log_prob is correct in the support (use valid range (0, 1) exclusive
        # to avoid boundaries: 0 maps to inf and 1 maps to the boundary where log_prob = -inf)
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(0.001, 0.999, 1000)[None])
        y = jax.vmap(p.base_prior[0].transform)(x)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(
            jax.vmap(p.log_prob)(y), stats.rayleigh.logpdf(y["x"], scale=sigma)
        )

        # Check log_prob is -inf for negative values (outside support [0, inf))
        x_outside = p.add_name(jnp.array([-1.0, -10.0])[None])
        logp_outside = jax.vmap(p.log_prob)(x_outside)
        assert jnp.all(logp_outside == -jnp.inf)

        # Check log_prob is jittable
        jitted_log_prob = jax.jit(jax.vmap(p.log_prob))
        jitted_val = jitted_log_prob(y)
        assert_all_finite(jitted_val)
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))


class TestMultivariatePrior:
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


class TestOther:
    def test_bounded_mixin(self):
        """Test the BoundedMixin mixin."""

        class TestBoundedPrior(BoundedMixin, SequentialTransformPrior):
            xmin: float = -2.0
            xmax: float = 3.0

            def __init__(self, parameter_names):
                from jimgw.core.prior import StandardNormalBase
                super().__init__(
                    [StandardNormalBase([f"{parameter_names[0]}_base"])],
                    [],
                )

        p = TestBoundedPrior(["x"])

        # Check log_prob is finite inside bounds
        x_inside = p.add_name(jnp.array([-1.5, 0.0, 2.5])[None])
        logp_inside = jax.vmap(p.log_prob)(x_inside)
        assert_all_finite(logp_inside)

        # Check log_prob is -inf outside bounds
        x_outside = p.add_name(jnp.array([-2.5, 3.5])[None])
        logp_outside = jax.vmap(p.log_prob)(x_outside)
        assert jnp.all(logp_outside == -jnp.inf)

        # Check JIT works correctly
        x = p.add_name(jnp.linspace(-3.0, 4.0, 100)[None])
        logp = jax.vmap(p.log_prob)(x)
        jitted_logp = jax.jit(jax.vmap(p.log_prob))(x)
        assert jnp.allclose(logp, jitted_logp)
