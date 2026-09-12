"""Config for Pi0 Modular (KSM + Memory Modulation)."""
import dataclasses
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

HISTORY_T = 6
HISTORY_VIEWS = ("history_base", "history_wrist")


@dataclasses.dataclass(frozen=True)
class Pi0ModularConfig(pi0_config.Pi0Config):
    @override
    def create(self, rng: at.KeyArrayLike):
        from openpi.models.pi0_modular import Pi0Modular
        return Pi0Modular(self, rngs=nnx.Rngs(rng))

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Same as base Pi0Config, but excludes the new memory/history modules
        from being frozen, since they are randomly initialized and must be
        trained for the Modulator (MemoryAttention + MemoryRMSNorm) to learn
        meaningful gamma/beta modulation.
        """
        base_filter = super().get_freeze_filter()
        memory_modules = nnx_utils.PathRegex(
            ".*mem_attn.*|.*mem_rms_norm.*|.*mem_proj.*|.*pos_proj.*"
        )
        return nnx.All(base_filter, nnx.Not(memory_modules))

    @override
    def inputs_spec(self, *, batch_size: int = 1):
        image_spec      = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        hist_spec       = jax.ShapeDtypeStruct(
            [batch_size, HISTORY_T, *_model.IMAGE_RESOLUTION, 3], jnp.float32
        )
        hist_mask_spec       = jax.ShapeDtypeStruct([batch_size, HISTORY_T], jnp.bool_)
        hist_timesteps_spec  = jax.ShapeDtypeStruct([batch_size, HISTORY_T], jnp.int32)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb":        image_spec,
                    "left_wrist_0_rgb":  image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb":        image_mask_spec,
                    "left_wrist_0_rgb":  image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                history_images={k: hist_spec for k in HISTORY_VIEWS},
                history_mask=hist_mask_spec,
                history_timesteps=hist_timesteps_spec,
                phase_id=jax.ShapeDtypeStruct([batch_size], jnp.int32),
            )
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )
        return observation_spec, action_spec