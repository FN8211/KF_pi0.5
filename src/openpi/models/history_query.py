"""
JAX/nnx version of KC-VLA HistoryQueryModule.
Position encoding (time/view/spatial) is handled externally by PosEmb3D.
This module receives already-encoded, flattened kv tokens [B, S, D].
"""
import jax
import jax.numpy as jnp
from flax import nnx


class HistoryQueryModule(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        query_dim: int,
        *,
        num_queries: int = 32,
        num_phases: int = 100,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        rngs: nnx.Rngs,
    ):
        assert input_dim == query_dim
        self.num_queries = num_queries
        self.input_dim   = input_dim

        self.query_tokens    = nnx.Param(
            jax.random.normal(rngs.params(), (1, num_queries, query_dim))
        )
        self.phase_embedding = nnx.Embed(num_phases, query_dim, rngs=rngs)

        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=input_dim,
            qkv_features=input_dim,
            decode=False,
            rngs=rngs,
        )
        self.query_norm  = nnx.LayerNorm(query_dim, rngs=rngs)
        self.output_norm = nnx.LayerNorm(input_dim, rngs=rngs)
        self.dropout     = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, kv, phase_ids, history_mask=None, *, train: bool = False):
        """
        Args:
            kv:           [B, S, D]  already flattened + LayerNorm'd history tokens
                          S = T * V * N_pool = 6 * 2 * 16 = 192
            phase_ids:    [B] int32
            history_mask: [B, T] bool, True=padding, False=valid
                          will be expanded to [B, S] for cross-attention masking
            train:        dropout switch
        Returns:
            out: [B, num_queries, D]
        """
        B, S, D = kv.shape

        # A. query 准备：可学习 query token + phase_id embedding + LayerNorm
        queries = jnp.broadcast_to(self.query_tokens.value, (B, self.num_queries, D))
        if phase_ids is not None:
            queries = queries + self.phase_embedding(phase_ids)[:, None, :]
        queries_norm = self.query_norm(queries)

        # B. attention mask
        # history_mask: [B, T] True=padding → expand to [B, S] → logical_not → True=valid
        attn_mask = None
        if history_mask is not None:
            T = history_mask.shape[1]
            repeat_factor = S // T  # V * N_pool
            valid_expanded = jnp.logical_not(
                jnp.repeat(history_mask, repeat_factor, axis=1)
            )  # [B, S] True=valid
            attn_mask = valid_expanded[:, None, None, :]  # [B, 1, 1, S]

        # C. cross-attention：query attend kv
        attn_out = self.cross_attn(
            queries_norm,
            kv,
            kv,
            mask=attn_mask,
        )

        # D. 残差 + dropout + output norm
        out = queries + self.dropout(attn_out, deterministic=not train)
        out = self.output_norm(out)
        return out