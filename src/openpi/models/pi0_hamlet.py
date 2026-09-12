"""
pi0_hamlet.py  —— Pi0 (pi0.5) + HAMLET 记忆模块
================================================
本文件是 openpi 原 `pi0.py` 的【整份复制】,仅在三处插入 HAMLET 逻辑,
每处都用  # ===== HAMLET: 新增开始/结束 =====  框出。其余代码与原 pi0.py 逐字一致。

原版 pi0.py 不受影响 —— 用本类需要在 TrainConfig 里把模型换成 Pi0HamletConfig
(见 train config 说明),原版 config 继续用原 Pi0,互不干扰。

HAMLET 接法(针对 pi0.5 的双塔 + attention 结构):
  - moment token 作为额外 token 追加到 PaliGemma 的 prefix 末尾(与图像/语言同侧);
  - 对历史 T-1 帧各跑一次 PaliGemma(stop_gradient,只为提取 moment 摘要);
  - 记忆模块聚合 T 步摘要 -> history-augmented 当前摘要;
  - 把它作为【额外 prefix token】注入当前前向,action expert 通过既有 attention 读到。

历史帧从哪来:数据侧用 delta_timestamps 取负偏移历史,经 data_loader_hamlet 透传到
batch 里的 "history" 字段;compute_loss 读取它。见文末 build_history_observation。

第一版不含 TCL(moment token 随机初始化)。history_length T=4。
"""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

# ===== HAMLET: 新增开始 =====
# 纯模块,不依赖 openpi。放在同目录或可 import 到的位置。
from openpi.models.hamlet_modules import HamletConfig, MomentTokens, MemoryModule
# ===== HAMLET: 新增结束 =====

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0Hamlet(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # ===== HAMLET: 新增开始 =====
        # moment token 的宽度必须 = PaliGemma 宽度(prefix 侧的 embedding 维度)。
        hcfg = HamletConfig(
            width=paligemma_config.width,
            n_moment_tokens=4,
            history_length=4,
            memory_layers=2,
            dtype=config.dtype,
        )
        self.hamlet_cfg = hcfg
        self.moment_tokens = MomentTokens(hcfg, rngs=rngs)
        self.memory_module = MemoryModule(hcfg, rngs=rngs)
        # ===== HAMLET: 新增结束 =====

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        # ===== HAMLET: 新增开始 =====
        # 只编码"当前帧"的图像 key,跳过塞进来的历史 key("__hist" 后缀)。
        # 否则历史帧会被当成额外的当前输入视图编码进 prefix —— 那就退化成 naive 多帧,
        # 正是 HAMLET 要避免的(论文 Table 2:naive 多帧反而降性能)。
        # ===== HAMLET: 新增结束 =====
        for name in obs.images:
            if "__hist" in name:  # HAMLET: 跳过历史帧 key
                continue
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    # ===== HAMLET: 新增开始 =====
    def _append_moment_tokens(self, tokens, input_mask, ar_mask):
        """把 n_m 个 moment token 追加到 prefix 序列末尾。
        返回新的 (tokens, input_mask, ar_mask) 以及 moment token 所在的位置数(便于切回)。
        moment token 设为可被全部 prefix（图像+语言）注意到（ar_mask=False），
        这样它们的输出 = 当前帧的 context-aware 摘要。
        """
        b = tokens.shape[0]
        n_m = self.hamlet_cfg.n_moment_tokens
        moment = self.moment_tokens.expand(b)  # [b, n_m, emb]
        tokens = jnp.concatenate([tokens, moment], axis=1)
        input_mask = jnp.concatenate(
            [input_mask, jnp.ones((b, n_m), dtype=jnp.bool_)], axis=1
        )
        ar_mask = jnp.concatenate([ar_mask, jnp.array([False] * n_m)], axis=0)
        return tokens, input_mask, ar_mask, n_m

    def _extract_moment_summary(self, observation, *, stop_grad: bool = True):
        """单帧 prefix-only 前向, 切出 moment 摘要 [b, n_m, emb]。"""
        pt, pm, pa = self.embed_prefix(observation)
        pt, pm, pa, n_m = self._append_moment_tokens(pt, pm, pa)
        attn_mask = make_attn_mask(pm, pa)
        positions = jnp.cumsum(pm, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm([pt, None], mask=attn_mask, positions=positions)
        out = prefix_out[:, -n_m:, :]
        return jax.lax.stop_gradient(out) if stop_grad else out


    def _split_history(self, observation):
        """从 observation.images 里把历史帧拆出来,构造 T-1 个"当前帧形态"的历史 observation。

        约定(data_loader_hamlet 负责写入):
          - 当前帧图像 key: "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"
          - 历史帧图像 key: "<view>__hist{i}",  i=0..T-2  (i 越小越久远)
          - 历史 state    : "state__hist{i}"  同理(可选,见下)
        返回:
          - cur_obs   : 只含当前帧图像的 observation(历史 key 已剔除)
          - history   : list 长度 T-1,每个是一个只含当前帧形态的 Observation(第 i 个历史时刻)
        若没有任何 __hist key(例如跑原版数据 / 不加记忆的 ablation),history 返回 []。
        """
        import dataclasses as _dc

        cur_images, cur_masks = {}, {}
        # 收集每个历史时刻 i 的各视图图像
        hist_imgs: dict[int, dict] = {}
        for name, img in observation.images.items():
            if "__hist" in name:
                view, idx = name.split("__hist")
                i = int(idx)
                hist_imgs.setdefault(i, {})[view] = img
            else:
                cur_images[name] = img
                if name in observation.image_masks:
                    cur_masks[name] = observation.image_masks[name]

        cur_obs = _dc.replace(observation, images=cur_images, image_masks=cur_masks)

        history = []
        for i in sorted(hist_imgs.keys()):  # oldest first
            imgs = dict(hist_imgs[i])
            masks = {v: jnp.ones(observation.state.shape[:-1], dtype=jnp.bool_) for v in imgs}
            # 方案A:补齐当前帧拥有、但历史帧缺失的视图(如 right_wrist_0_rgb)。
            # 用当前帧对应视图填充(当前帧里这些缺失视图本就是 LiberoInputs 填零的),
            # 并把它们的 mask 设为 False(无效,不参与有效注意)。
            for v in cur_images:
                if v not in imgs:
                    imgs[v] = cur_images[v]
                    masks[v] = jnp.zeros(observation.state.shape[:-1], dtype=jnp.bool_)
            h_obs = _dc.replace(
                observation,
                images=imgs,
                image_masks=masks,
                state=observation.state,
            )
            history.append(h_obs)
        history = []  # === DIAG: 临时关闭历史,诊断显存 ===
        return cur_obs, history
    # ===== HAMLET: 新增结束 =====

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        # 注:历史帧不走额外参数(openpi 的 train_step 是 jit 死签名,只传 obs/actions),
        #     而是由 data_loader_hamlet 塞进 observation.images 的 "__hist{i}" key,
        #     在下面用 _split_history 拆出来。见文件头说明。
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # ===== HAMLET: 新增开始 =====
        # 路线甲:当前帧只做 1 次 prefix-only 轻前向提摘要(带梯度),不做多余的完整前向。
        # 0) 先拆历史(此时 observation 还带 __hist key),再分别 preprocess。
        #    注意:必须先拆再 preprocess —— preprocess_observation 只认 IMAGE_KEYS,
        #    若把历史 key 一起传进去会校验/处理出错。
        raw_cur_obs, raw_history = self._split_history(observation)
        cur_obs = _model.preprocess_observation(preprocess_rng, raw_cur_obs, train=train)
        history = [
            _model.preprocess_observation(None, h, train=False) for h in raw_history
        ]
        n_m = self.hamlet_cfg.n_moment_tokens
        # === DIAG: 跳过提摘要/记忆/拼接, 退化成纯当前帧前向 ===
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(cur_obs)
        # ===== HAMLET: 新增结束 =====

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(cur_obs, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        # ===== HAMLET: 新增结束(完整前向用上 aug_moment) =====

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        # ===== HAMLET: 新增开始 =====
        history: list[_model.Observation] | None = None,
        # ===== HAMLET: 新增结束 =====
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # ===== HAMLET: 新增开始 =====
        # 与 compute_loss 对齐:当前帧 moment 摘要 + 历史摘要 -> 记忆模块 -> 注入 prefix。
        pt, pm, pa, n_m = self._append_moment_tokens(prefix_tokens, prefix_mask, prefix_ar_mask)
        attn_mask0 = make_attn_mask(pm, pa)
        positions0 = jnp.cumsum(pm, axis=1) - 1
        (prefix_out0, _), _ = self.PaliGemma.llm([pt, None], mask=attn_mask0, positions=positions0)
        cur_summary = prefix_out0[:, -n_m:, :]
        if history is not None and len(history) > 0:
            hist_summaries = [
                self._extract_moment_summary(_model.preprocess_observation(None, h, train=False)) for h in history
            ]
            stack = jnp.stack([*hist_summaries, cur_summary], axis=1)
        else:
            T = self.hamlet_cfg.history_length
            stack = jnp.broadcast_to(cur_summary[:, None], (cur_summary.shape[0], T, n_m, cur_summary.shape[-1]))
        aug_moment = self.memory_module(stack)
        prefix_tokens = jnp.concatenate([prefix_tokens, aug_moment], axis=1)
        prefix_mask = jnp.concatenate(
            [prefix_mask, jnp.ones((prefix_mask.shape[0], n_m), dtype=jnp.bool_)], axis=1
        )
        prefix_ar_mask = jnp.concatenate([prefix_ar_mask, jnp.array([False] * n_m)], axis=0)
        # ===== HAMLET: 新增结束 =====

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
