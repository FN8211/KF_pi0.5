"""
pi0_modular.py - Pi0 with KSM-keyframe-based perceptual memory + Modulator.

History frames: SigLIP -> max pool (4x4) -> PosEmb3D -> flatten -> LayerNorm
             -> Linear(2048->1024)  [== RoboMME's PerceptualMemory.encoder_static]
             -> mem_seq [B, T*V*16, 1024] -> Modulator (mem_attn / mem_rms_norm_ffn)

Aligned with RoboMME's memory-as-modulator design (Appendix A.3.2 / B.2.1):
mem_seq is the (linearly-projected) history token sequence itself, with no
intermediate learnable-query compression layer. Progress/phase information
is conveyed implicitly via mem_mask (number of valid keyframe slots) and
PosEmb3D's relative-temporal encoding (derived from history_timesteps),
rather than via an explicit phase_id embedding.
"""

import dataclasses
import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.integration_utils import get_config as get_gemma_config
from openpi.models.history_gemma import Module as HistoryGemmaModule
from openpi.models.representation.data_utils import pool_tokens_to_size
from openpi.models.representation.posemb_3d import PosEmb3D
from openpi.models.representation.utils import kernel_init
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

import logging
logger = logging.getLogger("openpi")


def posemb_sincos(pos, embedding_dim, min_period, max_period):
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij", pos, 1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


class Pi0Modular(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config     = get_gemma_config(config.paligemma_variant)   # width=2048
        action_expert_config = get_gemma_config(config.action_expert_variant)  # width=1024

        self.T      = 6
        self.V      = 2
        self.N_pool = 16  # 4x4

        # ── LLM (PaliGemma + action expert) ──────────────────────────────────
        llm = nnx_bridge.ToNNX(
            HistoryGemmaModule(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                integration_type="modulation",
            )
        )
        llm.lazy_init(
            rngs=rngs, method="init",
            use_adarms=[False, True] if config.pi05 else [False, False],
            mem_mods=[False, True],
        )

        # ── SigLIP image encoder ──────────────────────────────────────────────
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,  # 2048
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # ── action projection ─────────────────────────────────────────────────
        self.action_in_proj  = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        if config.pi05:
            self.time_mlp_in  = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj          = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in  = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        # ── history encoding modules ──────────────────────────────────────────
        # NOTE: RoboMME's FeatureEncoder._encode_memory has no LayerNorm
        # between the content/position concat and encoder_static --
        # mem_seq = encoder_static(concat(image_emb, silu(pos_proj(pos_emb)))).
        # We therefore do not introduce one either (no self.history_ln).

        # Position-embedding projection, mirroring RoboMME's
        # FeatureEncoder._add_pos_emb: pos_emb -> Linear -> SiLU, then
        # concatenated with the content embedding (not added). This gives
        # the model a learnable "volume knob" for how much positional
        # information to mix in, rather than a fixed-magnitude additive term.
        # pos_output_dim=768 matches perceptual-framesamp-modul.yaml's
        # memory_feature.pos.hidden_dim (RoboMME's M-RoPE pos_input_dim is
        # also 768, but here both input and output of pos_proj are sized by
        # our own PosEmb3D output dim D=2048 -> 768, the dimension that
        # matters for matching RoboMME is the *output*/hidden_dim).
        self.pos_output_dim = 768
        self.pos_proj = nnx.Linear(
            paligemma_config.width, self.pos_output_dim, rngs=rngs,
            kernel_init=kernel_init,  # normal(stddev=0.02), matches RoboMME's FeatureEncoder
        )

        # Linear projection (2048 + pos_output_dim) -> 1024, equivalent to
        # RoboMME PerceptualMemory.feature_encoder.encoder_static. The output
        # of this projection IS mem_seq -- no intermediate learnable-query
        # module.
        self.mem_proj = nnx.Linear(
            paligemma_config.width + self.pos_output_dim,
            action_expert_config.width,
            rngs=rngs,
            kernel_init=kernel_init,  # normal(stddev=0.02), matches RoboMME's encoder_static
        )

        # NOTE: pos_emb_3d is intentionally NOT stored as an attribute here.
        # PosEmb3D is a plain Python object (not an nnx.Module), and storing
        # it as self.pos_emb_3d makes it part of the static pytree structure,
        # which caused a pytree-metadata mismatch under jit (two separately
        # constructed instances compare unequal). It is constructed locally
        # inside _encode_history instead -- identical math, no stored state.

        self.deterministic = True

    # ── history encoding ──────────────────────────────────────────────────────

    def _encode_history(self, obs: _model.Observation):
        if obs.history_images is None:
            return None, None

        view_keys = sorted(obs.history_images.keys())
        sample    = obs.history_images[view_keys[0]]
        B, T      = sample.shape[0], sample.shape[1]
        V         = len(view_keys)
        N_pool    = self.N_pool

        # 1. SigLIP encoding
        per_view_tokens = []
        for vk in view_keys:
            vid = obs.history_images[vk]  # [B, T, H, W, C]
            # preprocess_observation does NOT normalize history images.
            # Training data arrives as uint8 -> normalize to [-1, 1].
            # Inference data (maniskill_kcvla_policy.to_norm) arrives already
            # normalized to [-1, 1] float32 -> no-op.
            if vid.dtype == jnp.uint8:
                vid = vid.astype(jnp.float32) / 255.0 * 2.0 - 1.0

            flat   = vid.reshape(B * T, *vid.shape[2:])          # [B*T, H, W, C]
            tokens, _ = jax.lax.stop_gradient(
                self.PaliGemma.img(flat, train=False)
            )                                                      # [B*T, 256, 2048]
            per_view_tokens.append(tokens.reshape(B, T, 256, tokens.shape[-1]))

        # [B, T, V, 256, 2048]
        img_emb = jnp.stack(per_view_tokens, axis=2)
        D = img_emb.shape[-1]  # 2048

        # 2. Max pool: 256 -> 16 (4x4)
        img_emb_flat   = img_emb.reshape(B * T, V, 256, D)        # [B*T, V, 256, D]
        img_emb_pooled = pool_tokens_to_size(
            img_emb_flat, target_size=N_pool, pool_type="mean"
        )                                                          # [B*T, V, 16, D]
        img_emb_pooled = img_emb_pooled.reshape(B, T, V, N_pool, D)  # [B, T, V, 16, D]

        # 3. Position encoding (pool 之后, flatten 之前)
        # compute relative temporal positions: current - past
        pos_emb_3d = PosEmb3D(dim=D)
        if obs.history_timesteps is not None:
            latest       = obs.history_timesteps[:, -1:]           # [B, 1]
            relative_pos = (latest - obs.history_timesteps).astype(jnp.int32)  # [B, T]
        else:
            relative_pos = jnp.broadcast_to(
                jnp.arange(T)[None, :], (B, T)
            ).astype(jnp.int32)

        # [B, T, V, 16, 2048]
        pos_emb = pos_emb_3d(relative_pos, num_views=V, spatial_size=4)

        # 4. Flatten: [B, T*V*16, 2048]
        S = T * V * N_pool
        img_flat = img_emb_pooled.reshape(B, S, D)  # content, [B,S,2048]
        pos_flat = pos_emb.reshape(B, S, D)         # position, [B,S,2048]

        # 5. Content path: raw image embedding (no LayerNorm -- RoboMME's
        #    _encode_memory has none).
        #    Position path: learnable projection 2048 -> pos_output_dim + SiLU
        #    (mirrors RoboMME FeatureEncoder._add_pos_emb: pos_proj + silu),
        #    then concat (not added) so content and position occupy separate
        #    channels and the network can learn how much weight to give each,
        #    instead of a fixed-magnitude additive term competing with content
        #    in the same 2048 channels.
        pos_proj_out = nnx.silu(self.pos_proj(pos_flat))         # [B,S,pos_output_dim]
        kv = jnp.concatenate([img_flat, pos_proj_out], axis=-1)  # [B,S,2048+pos_output_dim]

        # 6. Linear projection -> 1024. This IS mem_seq -- no intermediate
        #    learnable-query compression (aligned with RoboMME's
        #    PerceptualMemory.encoder_static).
        mem_seq = self.mem_proj(kv)  # [B, S, 1024]

        # 7. mem_mask for MemoryAttention: True = valid (per history_gemma.py's
        #    `masked_logits = where(mem_mask, logits, -inf)` convention).
        #    obs.history_mask: [B, T], True = padding.
        if obs.history_mask is not None:
            # expand [B, T] -> [B, S] (each keyframe -> V*N_pool tokens), then invert
            mem_mask = jnp.logical_not(
                jnp.repeat(obs.history_mask, V * N_pool, axis=1)
            )  # [B, S], True = valid
        else:
            mem_mask = jnp.ones((B, S), dtype=jnp.bool_)

        return mem_seq, mem_mask

    # ── prefix / suffix embedding ─────────────────────────────────────────────

    @at.typecheck
    def embed_prefix(self, obs: _model.Observation):
        input_mask, ar_mask, tokens = [], [], []

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens     = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask    = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(self, obs, noisy_actions, timestep):
        input_mask, ar_mask, tokens = [], [], []

        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )

        if self.pi05:
            time_emb             = self.time_mlp_in(time_emb)
            time_emb             = nnx.swish(time_emb)
            time_emb             = self.time_mlp_out(time_emb)
            time_emb             = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond          = time_emb
        else:
            time_tokens          = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens   = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens   = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens   = nnx.swish(action_time_tokens)
            action_time_tokens   = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond          = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens     = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask    = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    # ── training ──────────────────────────────────────────────────────────────

    @override
    def compute_loss(self, rng, observation, actions, *, train=False):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape    = actions.shape[:-2]
        noise          = jax.random.normal(noise_rng, actions.shape)
        time           = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded  = time[..., None, None]
        x_t            = time_expanded * noise + (1 - time_expanded) * actions
        u_t            = noise - actions

        mem_seq, mem_mask = self._encode_history(observation)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask    = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask  = make_attn_mask(input_mask, ar_mask)
        positions  = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            mem_seq=[None, mem_seq],
            mem_mask=[None, mem_mask],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    # ── inference ─────────────────────────────────────────────────────────────

    @override
    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        observation = _model.preprocess_observation(None, observation, train=False)
        dt         = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        mem_seq, mem_mask = self._encode_history(observation)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions        = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache      = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
            mem_seq=[None, mem_seq],
            mem_mask=[None, mem_mask],
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask   = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions        = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1) - 1
            )
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                mem_seq=[None, mem_seq],
                mem_mask=[None, mem_mask],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0