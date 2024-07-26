from jimgw.prior import *


class TestUnivariatePrior:
    def test_logistic(self):
        p = LogitTransform()

    def test_uniform(self):
        p = UniformPrior(0.0, 10.0, ["x"])
        samples = p._dist.base_prior.sample(jax.random.PRNGKey(0), 10000)
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.allclose(log_prob, -jnp.log(10.0))


class TestPriorOperations:
    def test_combine(self):
        raise NotImplementedError

    def test_sequence(self):
        raise NotImplementedError

    def test_factor(self):
        raise NotImplementedError
