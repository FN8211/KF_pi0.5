"""
serve_policy_zero_mem.py

在加载 serve_policy.py 之前，对 Pi0Modular._encode_history 做 class-level
monkey-patch，强制 mem_seq 全部置零（mem_mask 保持不变）。

用途：诊断"mem_attn/Modulator 这条路径是否是抖动的元凶"——
用现有 v2 checkpoint，不重新训练，对比 zero-mem 前后 eval 行为。

用法（与原 serve_policy.py 完全一致的命令行参数）：
    cd ~/autodl-tmp/openpi
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python scripts/serve_policy_zero_mem.py \
        policy:checkpoint \
        --policy.config=pi05_maniskill_modular \
        --policy.dir=checkpoints/pi05_maniskill_modular/modular_experiment/2999
"""

import sys
sys.path.insert(0, "/root/autodl-tmp/openpi/src")

import jax
from openpi.models.pi0_modular import Pi0Modular

_original_encode_history = Pi0Modular._encode_history


def _zero_encode_history(self, observation):
    mem_seq, mem_mask = _original_encode_history(self, observation)
    mem_seq = jax.tree.map(
        lambda x: jax.numpy.zeros_like(x) if x is not None else None, mem_seq
    )
    return mem_seq, mem_mask


Pi0Modular._encode_history = _zero_encode_history
print(">>> [PATCHED] Pi0Modular._encode_history -> mem_seq forced to zero <<<")

# 运行原始 serve_policy.py 的 main 逻辑（参数透传）
import runpy

runpy.run_path("/root/autodl-tmp/openpi/scripts/serve_policy.py", run_name="__main__")