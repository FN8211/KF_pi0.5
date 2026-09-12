"""
KC-VLA data config for ManiSkill + openpi.
Wraps the standard LeRobot dataset, adding history frames per step.
Follows KC-VLA compute_obs_frames logic exactly.
"""

import dataclasses
import json
import pathlib
from typing_extensions import override

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi import transforms as _transforms
from openpi.policies import libero_policy

MAX_OBS_FRAMES = 6
CHUNK_SIZE = 1000


def _load_keyframes(path: str) -> dict[int, list[int]]:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def compute_obs_frames(
    keyframes: dict[int, list[int]],
    trajectory_id: int,
    base_index: int,
) -> tuple[list[int], list[int]]:
    """Identical to KC-VLA LeRobotSingleDataset.compute_obs_frames."""
    kfs = keyframes.get(trajectory_id, [])
    kfs = sorted(set(kfs))
    if 0 not in kfs:
        kfs = [0] + kfs

    used = [k for k in kfs if k <= base_index]
    if not used or used[-1] != base_index:
        used.append(base_index)

    if len(used) > MAX_OBS_FRAMES:
        frames = used[-MAX_OBS_FRAMES:]
    else:
        frames = used

    mask = [1] * len(frames)

    if len(frames) < MAX_OBS_FRAMES:
        pad_len = MAX_OBS_FRAMES - len(frames)
        frames = [0] * pad_len + frames
        mask = [0] * pad_len + mask

    return frames, mask


def read_video_frames(
    video_path: pathlib.Path,
    frame_indices: list[int],
) -> np.ndarray:
    """Read specific frames from a video file. Returns [T, H, W, C] uint8 RGB."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            # pad with zeros if frame read fails
            if frames:
                frames.append(np.zeros_like(frames[-1]))
            else:
                frames.append(np.zeros((256, 256, 3), dtype=np.uint8))
    cap.release()
    return np.stack(frames, axis=0)  # [T, H, W, C]


class KCVLADatasetWrapper(Dataset):
    """
    Wraps a LeRobotDataset, adding KC-VLA history frames to each sample.
    Reads video frames directly using cv2 for arbitrary frame access.
    """

    def __init__(
        self,
        base_dataset,
        keyframes: dict[int, list[int]],
        dataset_root: pathlib.Path,
        model_type: _model.ModelType,
    ):
        self.base = base_dataset
        self.keyframes = keyframes
        self.root = pathlib.Path(dataset_root)
        self.model_type = model_type

        # Load parquet timestamps for accurate frame lookup
        self._timestamps: dict[int, np.ndarray] = {}

    def __len__(self):
        return len(self.base)

    def _get_video_path(self, episode_id: int, video_key: str) -> pathlib.Path:
        chunk = episode_id // CHUNK_SIZE
        return self.root / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_id:06d}.mp4"

    def __getitem__(self, idx: int) -> dict:
        sample = self.base[idx]

        episode_id = int(sample["episode_index"].item())
        frame_id   = int(sample["frame_index"].item())

        # --- compute obs frames (KC-VLA logic) ---
        frames, mask = compute_obs_frames(self.keyframes, episode_id, frame_id)
        # mask: 1=valid, 0=padding -> True=padding (consistent with training convention)
        history_mask = np.array([not bool(m) for m in mask], dtype=bool)

        # --- history_timesteps: real frame indices, padding positions are 0 ---
        history_timesteps = np.array(frames, dtype=np.int32)  # [T]

        # --- phase_id ---
        kfs = self.keyframes.get(episode_id, [])
        all_kfs = sorted(set(kfs))
        phase_id = int(sum(1 for k in all_kfs if k > 0 and k <= frame_id))

        # --- read history frames from video ---
        base_path  = self._get_video_path(episode_id, "image")
        wrist_path = self._get_video_path(episode_id, "wrist_image")
        hist_base  = read_video_frames(base_path,  frames)  # [T, H, W, C] uint8
        hist_wrist = read_video_frames(wrist_path, frames)  # [T, H, W, C] uint8

        # --- current frame images (from base dataset, CHW float -> HWC uint8) ---
        def to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
            arr = t.numpy()
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = arr.transpose(1, 2, 0)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            return arr

        base_image  = to_hwc_uint8(sample["image"])
        wrist_image = to_hwc_uint8(sample["wrist_image"])
        state       = sample["state"].numpy().astype(np.float32)
        actions     = sample["actions"].numpy().astype(np.float32)
        task        = sample["task"] if isinstance(sample["task"], str) else sample["task"][0]

        return {
            "observation/image":       base_image,
            "observation/wrist_image": wrist_image,
            "observation/state":       state,
            "history_image": {
                "history_base":  hist_base,
                "history_wrist": hist_wrist,
            },
            "history_mask":       history_mask,
            "history_timesteps":  history_timesteps,
            "phase_id":           np.array(phase_id, dtype=np.int32),
            "actions":            actions,
            "prompt":             task,
        }