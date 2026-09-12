"""
预计算 ManiSkill 数据集所有帧的 SigLIP token。
只加载 SigLIP 参数，不加载完整模型。
输出：/root/autodl-tmp/local/maniskill_siglip/{image,wrist_image}/episode_XXXXXX.npy
每个文件 shape: [T, 16, 2048] float16
"""
import numpy as np
import pathlib
import json
import cv2
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import flax.traverse_util as traverse_util
from tqdm import tqdm

import openpi.models.siglip as _siglip
from openpi.models.model import restore_params
from openpi.shared.image_tools import resize_with_pad
from openpi.models.representation.data_utils import pool_tokens_to_size

DATASET_ROOT = pathlib.Path("/root/autodl-tmp/local/maniskill_dataset")
OUTPUT_DIR   = pathlib.Path("/root/autodl-tmp/local/maniskill_siglip")
CHECKPOINT   = "/root/autodl-tmp/openpi-cache/openpi-assets/checkpoints/pi05_libero/params/params"
CHUNK_SIZE   = 1000
N_POOL       = 16
BATCH_SIZE   = 64


def load_siglip():
    print("Loading SigLIP params...")
    params = restore_params(CHECKPOINT, restore_type=np.ndarray)
    siglip_params = params["PaliGemma"]["img"]

    img = nnx_bridge.ToNNX(
        _siglip.Module(
            num_classes=2048,
            variant="So400m/14",
            pool_type="none",
            scan=True,
            dtype_mm="bfloat16",
        )
    )
    img.lazy_init(jnp.ones((1, 224, 224, 3)), train=False, rngs=nnx.Rngs(0))
    graph_def, state = nnx.split(img)
    state.replace_by_pure_dict(siglip_params)
    img = nnx.merge(graph_def, state)
    print("SigLIP loaded")
    return img


@jax.jit
def encode_batch(img_module, frames):
    """frames: [B, 224, 224, 3] float32 [-1,1] -> [B, 16, 2048]"""
    tokens, _ = img_module(frames, train=False)
    pooled = pool_tokens_to_size(tokens, target_size=N_POOL)
    return pooled


def read_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0 * 2.0 - 1.0
        frames.append(frame)
    cap.release()
    return np.stack(frames) if frames else None


def process_episode(img_module, episode_id, video_key):
    chunk = episode_id // CHUNK_SIZE
    video_path = DATASET_ROOT / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_id:06d}.mp4"
    if not video_path.exists():
        return None

    frames = read_video_frames(video_path)
    if frames is None:
        return None

    T = frames.shape[0]
    frames_jnp = resize_with_pad(jnp.array(frames), 224, 224)

    all_tokens = []
    for i in range(0, T, BATCH_SIZE):
        batch = frames_jnp[i:i+BATCH_SIZE]
        tokens = encode_batch(img_module, batch)
        all_tokens.append(np.array(tokens, dtype=np.float16))

    return np.concatenate(all_tokens, axis=0)  # [T, 16, 2048]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    episodes_file = DATASET_ROOT / "meta/episodes.jsonl"
    episodes = []
    with open(episodes_file) as f:
        for line in f:
            ep = json.loads(line)
            episodes.append(ep["episode_index"])
    print(f"Total episodes: {len(episodes)}")

    img_module = load_siglip()

    for video_key in ["image", "wrist_image"]:
        print(f"\nProcessing {video_key}...")
        for ep_id in tqdm(episodes):
            out_path = OUTPUT_DIR / f"{video_key}/episode_{ep_id:06d}.npy"
            if out_path.exists():
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tokens = process_episode(img_module, ep_id, video_key)
            if tokens is not None:
                np.save(str(out_path), tokens)

    print("\nDone!")


if __name__ == "__main__":
    main()
