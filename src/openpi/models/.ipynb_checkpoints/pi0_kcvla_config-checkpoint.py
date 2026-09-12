"""Config for KC-VLA model. Inherits Pi0Config, overrides create() and inputs_spec()."""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at

# KC-VLA history dims
HISTORY_T = 6
HISTORY_VIEWS = ("history_base", "history_wrist")


@dataclasses.dataclass(frozen=True)
class Pi0KCVLAConfig(pi0_config.Pi0Config):
    @override
    def create(self, rng: at.KeyArrayLike):
        from openpi.models.pi0_kcvla import Pi0KCVLA

        return Pi0KCVLA(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        hist_spec = jax.ShapeDtypeStruct(
            [batch_size, HISTORY_T, *_model.IMAGE_RESOLUTION, 3], jnp.float32
        )

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                history_images={
                    k: hist_spec for k in HISTORY_VIEWS
                },
                history_mask=jax.ShapeDtypeStruct([batch_size, HISTORY_T], jnp.bool_),
                phase_id=jax.ShapeDtypeStruct([batch_size], jnp.int32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec
