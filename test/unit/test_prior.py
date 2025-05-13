import jax
import jax.numpy as jnp

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
)

import scipy.stats as stats

jax.config.update("jax_enable_x64", True)


class TestUnivariatePrior:
    def test_logistic(self):
        p = LogisticDistribution(["x"])

        # Check that the log_prob are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))

        # Cross-check log_prob with scipy.stats.logistic
        x = jnp.linspace(-10.0, 10.0, 1000)
        assert jnp.allclose(
            jax.vmap(p.log_prob)(p.add_name(x[None])), stats.logistic.logpdf(x)
        )

        # Check that log_prob is jittable
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_val = jax.vmap(jitted_log_prob)(p.add_name(x[None]))
        assert jnp.all(jnp.isfinite(jitted_val))
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(p.add_name(x[None])))

    def test_standard_normal(self):
        p = StandardNormalDistribution(["x"])

        # Check that the log_prob are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))

        # Cross-check log_prob with scipy.stats.norm
        x = jnp.linspace(-10.0, 10.0, 1000)
        assert jnp.allclose(
            jax.vmap(p.log_prob)(p.add_name(x[None])), stats.norm.logpdf(x)
        )

        # Check that log_prob is jittable
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_val = jax.vmap(jitted_log_prob)(p.add_name(x[None]))
        assert jnp.all(jnp.isfinite(jitted_val))
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(p.add_name(x[None])))

    def test_uniform(self):
        xmin, xmax = -10.0, 10.0
        p = UniformPrior(xmin, xmax, ["x"])

        # Check that all the samples are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all(jnp.isfinite(samples["x"]))
        assert jnp.all((samples["x"] > xmin) & (samples["x"] < xmax))

        # Check that the log_prob are finite
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))

        # Check that the log_prob are correct in the support
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.transform)(x)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), -jnp.log(xmax - xmin))

        # Check that log_prob is jittable
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_val = jax.vmap(jitted_log_prob)(y)
        assert jnp.all(jnp.isfinite(jitted_val))
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_sine(self):
        p = SinePrior(["x"])

        # Check that all the samples are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all(jnp.isfinite(samples["x"]))
        assert jnp.all((samples["x"] > 0.0) & (samples["x"] < jnp.pi))

        # Check that the log_prob are finite
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))

        # Check that the log_prob are correct in the support
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.base_prior[0].base_prior[0].transform)(x)
        y = jax.vmap(p.base_prior[0].transform)(y)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), jnp.log(jnp.sin(y["x"]) / 2.0))

        # Check that log_prob is jittable
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_val = jax.vmap(jitted_log_prob)(y)
        assert jnp.all(jnp.isfinite(jitted_val))
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_cosine(self):
        p = CosinePrior(["x"])

        # Check that all the samples are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all(jnp.isfinite(samples["x"]))
        assert jnp.all((samples["x"] > -jnp.pi / 2.0) & (samples["x"] < jnp.pi / 2.0))

        # Check that the log_prob are finite
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))

        # Check that the log_prob are correct in the support
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.base_prior[0].transform)(x)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), jnp.log(jnp.cos(y["x"]) / 2.0))

        # Check that log_prob is jittable
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_val = jax.vmap(jitted_log_prob)(y)
        assert jnp.all(jnp.isfinite(jitted_val))
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_uniform_sphere(self):
        p = UniformSpherePrior(["x"])

        # Check that all the samples are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all(jnp.isfinite(samples["x_mag"]))
        assert jnp.all(jnp.isfinite(samples["x_theta"]))
        assert jnp.all(jnp.isfinite(samples["x_phi"]))
        assert jnp.all((samples["x_mag"] > 0.0) & (samples["x_mag"] < 1.0))
        assert jnp.all((samples["x_theta"] > 0.0) & (samples["x_theta"] < jnp.pi))
        assert jnp.all((samples["x_phi"] > 0.0) & (samples["x_phi"] < 2 * jnp.pi))

        # Check that the log_prob are finite
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))

    def test_power_law(self):
        xmin, xmax = 0.1, 100.0
        for alpha in [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]:  # -1.0 is a special case
            alpha = float(alpha)
            p = PowerLawPrior(xmin, xmax, alpha, ["x"])

            # Check that all the samples are finite
            samples = p.sample(jax.random.PRNGKey(0), 10000)
            assert jnp.all(jnp.isfinite(samples["x"]))
            assert jnp.all((samples["x"] > xmin) & (samples["x"] < xmax))

            # Check that the log_prob are finite
            log_prob = jax.vmap(p.log_prob)(samples)
            assert jnp.all(jnp.isfinite(log_prob))

            # Check that the log_prob are correct in the support
            x = p.trace_prior_parent([])[0].add_name(
                jnp.linspace(-10.0, 10.0, 1000)[None]
            )
            y = jax.vmap(p.transform)(x)
            if alpha < -1.0:
                assert jnp.allclose(
                    jax.vmap(p.log_prob)(y),
                    alpha * jnp.log(y["x"])
                    + jnp.log(-alpha - 1)
                    - jnp.log(xmin ** (alpha + 1) - xmax ** (alpha + 1)),
                )
            elif alpha > -1.0:
                assert jnp.allclose(
                    jax.vmap(p.log_prob)(y),
                    alpha * jnp.log(y["x"])
                    + jnp.log(alpha + 1)
                    - jnp.log(xmax ** (alpha + 1) - xmin ** (alpha + 1)),
                )
            else:
                assert jnp.allclose(
                    jax.vmap(p.log_prob)(y),
                    -jnp.log(y["x"]) - jnp.log(jnp.log(xmax) - jnp.log(xmin)),
                )

            # Check that log_prob is jittable
            jitted_log_prob = jax.jit(p.log_prob)
            jitted_val = jax.vmap(jitted_log_prob)(y)
            assert jnp.all(jnp.isfinite(jitted_val))
            assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_gaussian(self):
        mu, sigma = 2.0, 3.0
        p = GaussianPrior(mu, sigma, ["x"])

        # Check that all the samples are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all(jnp.isfinite(samples["x"]))

        # Check that the log_prob are finite
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))

        # Check that the log_prob are correct in the support
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.transform)(x)
        assert jnp.allclose(
            jax.vmap(p.log_prob)(y),
            stats.norm.logpdf(y["x"], loc=mu, scale=sigma),
        )

        # Check that log_prob is jittable
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_val = jax.vmap(jitted_log_prob)(y)
        assert jnp.all(jnp.isfinite(jitted_val))
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_Rayleigh(self):
        sigma = 2.0
        p = RayleighPrior(sigma, ["x"])

        # Check that all the samples are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all(jnp.isfinite(samples["x"]))
        assert jnp.all(samples["x"] > 0.0)

        # Check that the log_prob are finite
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))

        # Check that the log_prob are correct in the support
        x = p.trace_prior_parent([])[0].add_name(jnp.linspace(0.0, 10.0, 1000)[None])
        y = jax.vmap(p.base_prior[0].transform)(x)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(
            jax.vmap(p.log_prob)(y),
            stats.rayleigh.logpdf(y["x"], scale=sigma),
        )

        # Check that log_prob is jittable
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_val = jax.vmap(jitted_log_prob)(y)
        assert jnp.all(jnp.isfinite(jitted_val))
        assert jnp.allclose(jitted_val, jax.vmap(p.log_prob)(y))

    def test_full_range_prior(self):
        # Define a uniform prior over [0, 1]
        base = UniformPrior(0.0, 1.0, ["x"])
        # FullRangePrior with no extra constraints (should behave like base)
        p = FullRangePrior(base)

        # All samples should be in [0, 1]
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all((samples["x"] > 0.0) & (samples["x"] < 1.0))
        # log_prob should be finite in [0, 1], -inf outside
        xs = jnp.linspace(-0.5, 1.5, 1000)
        xs_dict = p.add_name(xs[None])
        logp = jax.vmap(p.log_prob)(xs_dict)
        mask = (xs >= 0.0) & (xs <= 1.0)
        assert jnp.all(jnp.isfinite(logp[mask]))
        assert jnp.all(logp[~mask] == -jnp.inf)
        # Should match base prior in valid region
        base_logp = jax.vmap(base.log_prob)(xs_dict)
        assert jnp.allclose(logp[mask], base_logp[mask])

        # Add an extra constraint: x < 0.5
        p2 = FullRangePrior(base, extra_constraints=[lambda z: z["x"] < 0.5])
        xs2 = jnp.linspace(-0.5, 1.5, 1000)
        xs2_dict = p2.add_name(xs2[None])
        logp2 = jax.vmap(p2.log_prob)(xs2_dict)
        mask2 = (xs2 >= 0.0) & (xs2 < 0.5)
        assert jnp.all(jnp.isfinite(logp2[mask2]))
        assert jnp.all(logp2[~mask2] == -jnp.inf)

        # Check vmap/jit compatibility
        jitted_log_prob = jax.jit(p.log_prob)
        jitted_vals = jax.vmap(jitted_log_prob)(xs_dict)
        assert jnp.allclose(jitted_vals, logp)
