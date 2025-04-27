from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from jimgw.prior import (
    PowerLawPrior,
)


@jaxtyped(typechecker=typechecker)
class UniformComponentChirpMassPrior(PowerLawPrior):
    """
    A prior in the range [xmin, xmax) for chirp mass which assumes the
    component masses to be uniformly distributed.

    p(M_c) ~ M_c
    """

    def __repr__(self):
        return f"UniformInComponentsChirpMassPrior(xmin={self.xmin}, xmax={self.xmax}, naming={self.parameter_names})"

    def __init__(self, xmin: float, xmax: float):
        super().__init__(xmin, xmax, 1.0, ["M_c"])