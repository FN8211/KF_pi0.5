import dataclasses
from openpi.models import pi0_config
import openpi.shared.array_typing as at
import flax.nnx as nnx


@dataclasses.dataclass(frozen=True)
class Pi0HamletConfig(pi0_config.Pi0Config):
    def create(self, rng: at.KeyArrayLike):
        from openpi.models.pi0_hamlet import Pi0Hamlet
        return Pi0Hamlet(self, rngs=nnx.Rngs(rng))
