"""
gemma_modular.py - Gemma with Memory Modulation for KC-VLA.
Adds MemoryAttention + MemoryRMSNorm to action expert FFN, following RoboMME.
"""

from collections.abc import Sequence
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models.gemma import (
    PALIGEMMA_VOCAB_SIZE,
    Config,
    Embedder,
    KVCache,
    RMSNorm,
    Attention,
    FeedForward,
    _apply_rope,
    _gated_residual,
    _name,
    get_config,
)
import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding


@at.typecheck
class MemoryRMSNorm(nn.Module):
    """RMSNorm conditioned on memory vector (scale + shift)."""
    @nn.compact
    def __call__(self, x, cond=None):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        if cond is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1],))
            return (normed * (1 + scale)).astype(dtype)
        modulation = nn.Dense(
            x.shape[-1] * 2,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=dtype,
        )(cond)
        scale, shift = jnp.split(modulation, 2, axis=-1)
        return (normed * (1 + scale) + shift).astype(dtype)


@at.typecheck
class MemoryAttention(nn.Module):
    """Cross-attention: action expert tokens attend to memory tokens."""
    @nn.compact
    def __call__(self, x, mem_seq, mem_mask):
        # x: [B, T, D=1024], mem_seq: [B, S, D=1024], mem_mask: [B, S] True=padding
        B, mem_len, width = mem_seq.shape
        num_heads, num_kv_heads, head_dim = 4, 1, 256

        q_einsum = lora.Einsum(
            shape=(num_heads, width, head_dim),
            name="q_einsum_mem",
            init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
        )
        kv_einsum = lora.Einsum(
            shape=(2, num_kv_heads, width, head_dim),
            name="kv_einsum_mem",
            init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
        )
        rms = MemoryRMSNorm(name="mem_rms_norm")

        x_len = x.shape[1]
        q = q_einsum("BTD,NDH->BTNH", rms(x))
        k, v = kv_einsum("BSD,2KDH->2BSKH", rms(mem_seq))

        q_pos = einops.repeat(jnp.arange(mem_len, x_len + mem_len), "t -> b t", b=B)
        k_pos = einops.repeat(jnp.arange(mem_len), "s -> b s", b=B)
        q = _apply_rope(q, positions=q_pos)
        q *= head_dim ** -0.5
        k = _apply_rope(k, positions=k_pos)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        # mem_mask: True=padding -> mask out
        attn_mask = mem_mask[:, None, None, None, :]
        masked_logits = jnp.where(attn_mask, -2.3819763e38, logits)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(x.dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out_einsum = lora.Einsum(
            shape=(num_heads, head_dim, width),
            name="out_einsum_mem",
            init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
        )
        return out_einsum("BTNH,NHD->BTD", encoded)


@at.typecheck
class ModularBlock(nn.Module):
    """Transformer block with memory modulation on action expert."""
    configs: tuple[Config, ...]
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(
        self, xs, kv_cache, positions, attn_mask, adarms_cond,
        mem_seq, mem_mask, deterministic=True,
    ):
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x
        attn = Attention(configs=self.configs, name="attn")

        pre_attn, gates = [], []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        out, gates = [], []
        last_i = len(xs) - 1
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                # Memory modulation on action expert (last expert) before FFN
                if i == last_i and mem_seq is not None:
                    mem_mod = MemoryAttention(name="mem_attn")(x, mem_seq, mem_mask)
                    x = MemoryRMSNorm(name="mem_rms_norm_ffn")(x, mem_mod)

                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])
                x = lora.FeedForward(
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


@at.typecheck
class ModularModule(nn.Module):
    """Gemma module with memory modulation support."""
    configs: Sequence[Config]
    embed_dtype: str
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )
        block_cls = nn.remat(
            ModularBlock,
            prevent_cse=False,
            static_argnums=(7,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        mem_seq: at.Float[at.Array, "b _s _d"] | None = None,
        mem_mask: at.Bool[at.Array, "b _s"] | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, kv_cache = self.layers(
            embedded, kv_cache, positions, mask, adarms_cond,
            mem_seq, mem_mask, deterministic,
        )

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [
            f(e, a)[0] if e is not None else e
            for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def init(self, use_adarms: Sequence[bool]):
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[
                jnp.zeros((1, c.width)) if u else None
                for u, c in zip(use_adarms, self.configs, strict=True)
            ],
            mem_seq=jnp.zeros((1, 4, self.configs[-1].width)),
            mem_mask=jnp.ones((1, 4), dtype=bool),
        )
