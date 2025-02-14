from jimgw.prior import *
from jimgw.utils import trace_prior_parent
import scipy.stats as stats


class TestUnivariatePrior:
    def test_logistic(self):
        p = LogisticDistribution(["x"])

        # Check that the log_prob are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))

        # Cross-check log_prob with scipy.stats.logistic
        x = jnp.linspace(-10.0, 10.0, 1000)
        assert jnp.allclose(jax.vmap(p.log_prob)(p.add_name(x[None])), stats.logistic.logpdf(x), atol=1e-16)

    def test_standard_normal(self):
        p = StandardNormalDistribution(["x"])

        # Check that the log_prob are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))

        # Cross-check log_prob with scipy.stats.norm
        x = jnp.linspace(-10.0, 10.0, 1000)
        assert jnp.allclose(jax.vmap(p.log_prob)(p.add_name(x[None])), stats.norm.logpdf(x), atol=1e-16)

    def test_uniform(self):
        xmin, xmax = -10.0, 10.0
        p = UniformPrior(xmin, xmax, ["x"])

        # Check that all the samples are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all(jnp.isfinite(samples['x']))

        # Check that the log_prob are finite
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))

        # Check that the log_prob are correct in the support
        x = trace_prior_parent(p, [])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.transform)(x)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), -jnp.log(xmax - xmin), atol=1e-16)

    def test_sine(self):
        p = SinePrior(["x"])

        # Check that all the samples are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all(jnp.isfinite(samples['x']))

        # Check that the log_prob are finite
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))

        # Check that the log_prob are correct in the support
        x = trace_prior_parent(p, [])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.base_prior.base_prior.transform)(x)
        y = jax.vmap(p.base_prior.transform)(y)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), jnp.log(jnp.sin(y['x'])/2.0), atol=1e-16)
        
    def test_cosine(self):
        p = CosinePrior(["x"])

        # Check that all the samples are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all(jnp.isfinite(samples['x']))

        # Check that the log_prob are finite
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))

        # Check that the log_prob are correct in the support
        x = trace_prior_parent(p, [])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.base_prior.transform)(x)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), jnp.log(jnp.cos(y['x'])/2.0), atol=1e-16)

    def test_uniform_sphere(self):
        p = UniformSpherePrior(["x"])

        # Check that all the samples are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all(jnp.isfinite(samples['x_mag']))
        assert jnp.all(jnp.isfinite(samples['x_theta']))
        assert jnp.all(jnp.isfinite(samples['x_phi']))

        # Check that the log_prob are finite
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))

    # def test_power_law(self):
    #     xmin, xmax = 1.0, 100.0
    #     for alpha in jnp.linspace(-2.0, 2.0, 5):
    #         alpha = float(alpha)
    #         p = PowerLawPrior(xmin, xmax, alpha, ["x"])

    #         # Check that all the samples are finite
    #         samples = p.sample(jax.random.PRNGKey(0), 10000)
    #         assert jnp.all(jnp.isfinite(samples['x']))

    #         # Check that the log_prob are finite
    #         log_prob = jax.vmap(p.log_prob)(samples)
    #         assert jnp.all(jnp.isfinite(log_prob))

    #         # Check that the log_prob are correct in the support
    #         x = trace_prior_parent(p, [])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
    #         y = jax.vmap(p.transform)(x)
    #         assert jnp.allclose(jax.vmap(p.log_prob)(y), stats.powerlaw.logpdf(y['x'], alpha, loc=xmin, scale=xmax-xmin), atol=1e-16)
