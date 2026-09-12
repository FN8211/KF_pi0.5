"""
HAMLET 核心模块 (Flax NNX) —— 从论文 arXiv:2510.00695 重实现
=============================================================
两个部件,与 openpi 完全解耦,可独立测试:
  1. MomentTokens : 一组可学习 token,追加到 PaliGemma 的 prefix 序列末尾。
                    前向后从对应位置切出 hidden state,作为"这一帧的压缩摘要"。
  2. MemoryModule : 2 层 causal Transformer,把过去 T 步的 moment 摘要聚合成
                    history-augmented 的当前摘要,再作为额外 prefix token 注入。

论文默认: n_moment_tokens=4, history_length T=4, memory_layers=2。
第一版不含 TCL 初始化(moment token 随机初始化)。

注意: 这里只实现"模块"。如何把 moment token 注入 Pi0 的前向、如何切出它们的
输出、如何把记忆特征拼回 prefix —— 全部在 pi0_hamlet.py 里完成。
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class HamletConfig:
    width: int = 2048          # 必须等于 PaliGemma 的 embedding 宽度 (gemma_2b => 2048)
    n_moment_tokens: int = 4   # n_m,论文默认 4
    history_length: int = 4    # T,论文默认 4(含当前帧,所以历史帧=T-1=3)
    memory_layers: int = 2     # 论文默认 2 层 Transformer
    memory_heads: int = 8
    mlp_ratio: float = 4.0
    dtype: str = "bfloat16"


# --------------------------------------------------------------------------- #
# 1. Moment tokens
# --------------------------------------------------------------------------- #
class MomentTokens(nnx.Module):
    """可学习的 moment token 表 [n_m, width],运行时广播到 batch。

    这些 token 追加到 prefix 序列(图像+语言之后),前向时因为 PaliGemma 的
    注意力会去看图像和指令,于是它们的输出 = 当前帧的 context-aware 摘要。
    """

    def __init__(self, cfg: HamletConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        key = rngs.params()
        init = jax.random.normal(key, (cfg.n_moment_tokens, cfg.width)) * 0.02
        self.tokens = nnx.Param(init.astype(jnp.dtype(cfg.dtype)))

    def expand(self, batch_size: int) -> jax.Array:
        """返回 [B, n_m, width],拼到 prefix 的 token 序列里。"""
        return jnp.broadcast_to(
            self.tokens[...][None], (batch_size, self.cfg.n_moment_tokens, self.cfg.width)
        )


# --------------------------------------------------------------------------- #
# 2. Memory module (2 层 causal Transformer)
# --------------------------------------------------------------------------- #
class _Block(nnx.Module):
    """一个标准的 pre-norm Transformer block(MHA + MLP)。"""

    def __init__(self, cfg: HamletConfig, *, rngs: nnx.Rngs):
        dt = jnp.dtype(cfg.dtype)
        self.norm1 = nnx.LayerNorm(cfg.width, dtype=dt, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=cfg.memory_heads,
            in_features=cfg.width,
            dtype=dt,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(cfg.width, dtype=dt, rngs=rngs)
        hidden = int(cfg.width * cfg.mlp_ratio)
        self.fc1 = nnx.Linear(cfg.width, hidden, dtype=dt, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, cfg.width, dtype=dt, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        h = self.norm1(x)
        x = x + self.attn(h, mask=mask)
        h = self.norm2(x)
        h = self.fc2(nnx.gelu(self.fc1(h)))
        return x + h


class MemoryModule(nnx.Module):
    """把过去 T 步的 moment 摘要聚合 -> history-augmented 的当前摘要。

    输入  history: [B, T, n_m, width]   (oldest .. newest, newest = 当前帧)
    输出         : [B, n_m, width]       (当前帧、已带历史信息)

    论文 Table 5: 单纯 concat 几乎无收益,靠这个带因果注意力的 Transformer 学会
    "什么时候该回忆、回忆哪一刻"才是关键。
    """

    def __init__(self, cfg: HamletConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        dt = jnp.dtype(cfg.dtype)
        # 按"时间槽"的可学习位置编码(顺序信息),broadcast 到每个 moment token。
        key = rngs.params()
        pos = jax.random.normal(key, (cfg.history_length, cfg.width)) * 0.02
        self.pos = nnx.Param(pos.astype(dt))
        # 固定层数,存成命名属性,绕开 nnx 容器 API 的版本差异。
        self.n_layers = cfg.memory_layers
        for li in range(cfg.memory_layers):
            setattr(self, f"block_{li}", _Block(cfg, rngs=rngs))

    def _causal_mask(self, T: int, n_m: int) -> jax.Array:
        # token i 不能看 i 之后的;同一时间槽内的 n_m 个 token 互相可见。
        # 先在"时间槽"粒度上构造下三角,再 repeat 到 token 粒度。
        slot = jnp.tril(jnp.ones((T, T), dtype=bool))               # [T, T]
        mask = jnp.repeat(jnp.repeat(slot, n_m, axis=0), n_m, axis=1)  # [L, L]
        return mask[None, None]  # [1, 1, L, L] for MultiHeadAttention

    def __call__(self, history: jax.Array) -> jax.Array:
        B, T, n_m, d = history.shape
        # 加时间槽位置编码
        history = history + self.pos[...][None, :T, None, :]
        x = history.reshape(B, T * n_m, d)
        mask = self._causal_mask(T, n_m)
        for li in range(self.n_layers):
            x = getattr(self, f"block_{li}")(x, mask)
        # 取最后 n_m 个 token = 当前帧(现在带历史信息了)
        return x[:, -n_m:, :]


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = HamletConfig(width=256, n_moment_tokens=4, history_length=4, memory_layers=2, dtype="float32")
    rngs = nnx.Rngs(0)

    mt = MomentTokens(cfg, rngs=rngs)
    mm = MemoryModule(cfg, rngs=rngs)

    B = 2
    moment_in = mt.expand(B)
    assert moment_in.shape == (B, 4, 256), moment_in.shape

    # 假装我们已经为 T=4 帧各自切出了 moment 摘要
    history = jax.random.normal(jax.random.key(1), (B, 4, 4, 256))
    aug = mm(history)
    assert aug.shape == (B, 4, 256), aug.shape

    # 检查因果性:改最后一帧(当前帧)不应影响……其实当前帧是 query 末尾,
    # 改更早的帧应当影响当前输出;改"未来"无意义(没有未来)。这里只验证形状+可微。
    def loss_fn(graphdef, state):
        m = nnx.merge(graphdef, state)
        return jnp.mean(m(history) ** 2)

    graphdef, state = nnx.split(mm)
    g = jax.grad(loss_fn, argnums=1)(graphdef, state)
    n_grad = sum(int(jnp.size(x)) for x in jax.tree.leaves(g))
    print("moment_in:", tuple(moment_in.shape))
    print("aug      :", tuple(aug.shape))
    print("memory module trainable params have grads, total elems:", n_grad)
    print("OK")
