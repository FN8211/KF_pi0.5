"""
KC-VLA policy transforms for ManiSkill + openpi pi05_maniskill_kcvla.
Input dict comes from KCVLADatasetWrapper.__getitem__.
"""

import dataclasses
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    import einops
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image.astype(np.uint8)


@dataclasses.dataclass(frozen=True)
class ManiSkillKCVLAInputs(transforms.DataTransformFn):
    """
    Converts KCVLADatasetWrapper output into the format expected by Pi0KCVLA.

    Input keys:
        observation/image        [H,W,3] uint8
        observation/wrist_image  [H,W,3] uint8
        observation/state        [8] float32
        history_image/history_base   [T,H,W,3] uint8
        history_image/history_wrist  [T,H,W,3] uint8
        history_mask             [T] bool  (True=padding)
        phase_id                 scalar int32
        actions                  [7] float32
        prompt                   str

    Output keys (model input format):
        state, image, image_mask, history_image, history_mask, phase_id,
        actions (if present), prompt (if present)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image  = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # history images: already [T,H,W,3] uint8 from wrapper
        hist_base  = np.asarray(data["history_image"]["history_base"]).astype(np.uint8)
        hist_wrist = np.asarray(data["history_image"]["history_wrist"]).astype(np.uint8)

        # convert history to [-1,1] float32 (matches what Pi0KCVLA.embed_prefix expects)
        def to_norm(x: np.ndarray) -> np.ndarray:
            return x.astype(np.float32) / 255.0 * 2.0 - 1.0

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb":        base_image,
                "left_wrist_0_rgb":  wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb":        np.array(True),
                "left_wrist_0_rgb":  np.array(True),
                "right_wrist_0_rgb": (
                    np.array(True)
                    if self.model_type == _model.ModelType.PI0_FAST
                    else np.array(False)
                ),
            },
            "history_image": {
                "history_base":  to_norm(hist_base),   # [T,H,W,3] float32 [-1,1]
                "history_wrist": to_norm(hist_wrist),
            },
            "history_mask": np.asarray(data["history_mask"]).astype(bool),
            "phase_id":     np.asarray(data["phase_id"]).astype(np.int32),
            "history_timesteps": np.asarray(data["history_timesteps"]).astype(np.int32),
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ManiSkillKCVLAOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
