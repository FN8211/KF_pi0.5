"""
data_loader_hamlet.py  —— openpi data_loader 的 HAMLET 版【整份复制 + 历史采样】
================================================================================
本文件复制自 openpi 原 `data_loader.py`,只为 HAMLET 增加"历史帧采样"能力,
其余与原文件一致。原 data_loader.py 不受影响(方案乙:原文件零改动)。

它做的额外两件事:
  1. 在底层 LeRobotDataset 的 delta_timestamps 里,为 image/state 增加历史负偏移
     [-k*(T-1), ..., -2k, -k]  (k = action_horizon, T = history_length)。
     LeRobot 会自动从同一条 episode 里按时间戳取这些历史帧,episode 头部越界时
     用边界帧 padding(LeRobot 默认行为)。
  2. 把取回的历史帧 image 重命名成 "<view>__hist{i}" 塞进每个样本的 image dict。
     pi0_hamlet.py 的 _split_history 会按这个约定把历史拆出来。

只覆盖训练所需路径(create_torch_data_loader -> TorchDataLoader -> DataLoaderImpl)。
RLDS 路径(DROID)原样保留但 HAMLET 不用。

用法:训练脚本里把 `import openpi.training.data_loader as _data_loader` 换成本模块,
     或在你的 HAMLET 训练入口里 `from data_loader_hamlet import create_data_loader`。
"""

from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
from PIL import Image as _PILImage
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms
from openpi.shared import image_tools as _imgt

# 复用原 data_loader 里不变的部件,避免重复粘贴(它们与 HAMLET 无关)。
from openpi.training.data_loader import (
    Dataset,
    IterableDataset,
    DataLoader,
    TransformedDataset,
    FakeDataset,
    transform_dataset,
    _collate_fn,
    _worker_init_fn,
    RLDSDataLoader,
    create_rlds_data_loader,
)

T_co = TypeVar("T_co", covariant=True)

logger = logging.getLogger("openpi")


# ===== HAMLET: 历史采样的核心 =====
HAMLET_HISTORY_LENGTH = 4   # T,论文默认 4(含当前帧 => 历史帧 = T-1 = 3)
# k(历史间隔)默认 = action_horizon,见论文 Eq.(5)。在 create 里用 action_horizon 传入。


class HamletHistoryDataset(Dataset):
    """包一层:底层 LeRobotDataset 用 delta_timestamps 取了历史帧(带时间维),
    本 wrapper 负责:
      - 对"当前帧"走正常 openpi transform(repack/LiberoInputs/normalize/tokenize);
      - 把历史帧 image 抽出来,重命名 __hist{i} 塞进 transform 后的结果。

    注意:openpi 的 transform 设计为单帧。LeRobot 在 delta_timestamps 下会让
    被索引的 image/state 键多出一个时间维(shape 前面多个 T_total)。我们在
    transform 之前先把"当前帧"切出来喂给 transform,历史帧另存,transform 后再合并。
    """

    def __init__(
        self,
        base_lerobot_ds,
        transform: _transforms.DataTransformFn,
        *,
        history_length: int,
        action_horizon: int,
        image_keys_in_dataset: Sequence[str],
    ):
        self._ds = base_lerobot_ds
        self._transform = transform
        self._T = history_length
        self._k = action_horizon
        # 数据集里的原始 image 键名(LeRobot 里通常是 "image", "wrist_image")。
        self._img_keys = list(image_keys_in_dataset)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, index: SupportsIndex):
        sample = self._ds[index]

        # delta_timestamps 让 image 键带上时间维:约定历史偏移在前、当前帧在最后一位。
        # 见 create_torch_dataset_hamlet 里 delta_timestamps 的构造顺序。
        hist_imgs: dict[str, list] = {}
        cur_sample = dict(sample)
        for key in self._img_keys:
            if key not in sample:
                continue
            arr = np.asarray(sample[key])
            # arr shape: [T_total, H, W, C](T_total = (T-1) 历史 + 1 当前)
            # 最后一帧 = 当前帧,前面 T-1 帧 = 历史(oldest..newest-1)。
            cur_sample[key] = arr[-1]  # 当前帧喂给正常 transform
            hist_imgs[key] = [arr[i] for i in range(arr.shape[0] - 1)]  # 历史帧另存

        # 当前帧走正常 openpi transform(得到 image dict + tokenized prompt 等)。
        transformed = self._transform(cur_sample)

        # 把历史帧重命名塞回:键名规则与 transform 后当前帧 image 的视图名对应。
        # transform 后图像在 transformed["image"][<view>],<view> 形如 base_0_rgb 等。
        # 历史帧没经过 resize/normalize 之外的处理(它们只用于提 moment 摘要,
        # pi0_hamlet 里会对历史帧单独 preprocess_observation 做 resize)。
        # 因此这里只需把历史原图按 i 塞进 image dict,视图名沿用 LiberoInputs 的映射。
        # LiberoInputs 把 "image"->base_0_rgb, "wrist_image"->left_wrist_0_rgb(见 libero_policy)。
        view_map = {"image": "base_0_rgb", "wrist_image": "left_wrist_0_rgb"}
        img_dict = transformed["image"]
        for key, frames in hist_imgs.items():
            view = view_map.get(key, key)
            for i, f in enumerate(frames):
                # LeRobot 视频解码输出确定为 [C,H,W] float32 值域[0,1](已实测)。
                # 对齐当前帧: CHW->HWC, resize 到 224, [0,1]->[-1,1]。
                # 注意: 必须用纯 CPU(PIL)resize, 不能用 jax 的 resize_with_pad,
                #       否则会在 DataLoader worker 子进程里初始化 CUDA 导致 OOM。
                f = np.asarray(f, dtype=np.float32)
                f = np.transpose(f, (1, 2, 0))                      # [C,H,W] -> [H,W,C]
                # [0,1] float -> uint8 给 PIL, resize 后再回 float
                arr8 = np.clip(f * 255.0, 0, 255).astype(np.uint8)
                im = _PILImage.fromarray(arr8).resize((224, 224), _PILImage.BILINEAR)
                f = np.asarray(im, dtype=np.float32) / 255.0        # [224,224,3] in [0,1]
                f = f * 2.0 - 1.0                                   # -> [-1,1]
                img_dict[f"{view}__hist{i}"] = f.astype(np.float32)

        return transformed
# ===== HAMLET: 历史采样的核心 结束 =====


def create_torch_dataset_hamlet(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    history_length: int = HAMLET_HISTORY_LENGTH,
) -> Dataset:
    """对应原 create_torch_dataset,但 delta_timestamps 额外加入历史负偏移,
    并用 HamletHistoryDataset 包一层。"""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    fps = dataset_meta.fps

    # 1) action:沿用原逻辑,取未来 action_horizon 个偏移。
    delta_timestamps = {
        key: [t / fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
    }

    # 2) HAMLET:为 image(和 state)加入历史负偏移 + 当前帧(0)。
    #    顺序:历史最久远 -> ... -> 当前(0),当前帧在最后,与 HamletHistoryDataset 切片约定一致。
    k = action_horizon
    hist_offsets = [-k * (history_length - 1 - i) for i in range(history_length)]  # e.g. [-30,-20,-10,0]
    hist_timestamps = [off / fps for off in hist_offsets]
    # LeRobot 数据集里图像键名(转换脚本里定义的)。LIBERO 转换通常是 "image","wrist_image"。
    image_keys_in_dataset = ["image", "wrist_image"]
    for key in image_keys_in_dataset:
        delta_timestamps[key] = hist_timestamps
    # state 也取历史(可选;pi0_hamlet 目前历史帧沿用当前 state,这里取了也不影响)。

    dataset = lerobot_dataset.LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    # 组装 transform(与原 transform_dataset 相同的顺序),但作用在"当前帧"上。
    norm_stats = {}
    if data_config.repo_id != "fake":
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. Run scripts/compute_norm_stats.py for your config."
            )
        norm_stats = data_config.norm_stats
    transform = _transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )

    return HamletHistoryDataset(
        dataset,
        transform,
        history_length=history_length,
        action_horizon=action_horizon,
        image_keys_in_dataset=image_keys_in_dataset,
    )


# ---- 以下三个函数复制自原 data_loader,仅把 create_torch_dataset(+transform) ----
# ---- 的部分换成上面的 create_torch_dataset_hamlet。其余逐字一致。            ----


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    history_length: int = HAMLET_HISTORY_LENGTH,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"[HAMLET] data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        # HAMLET 不支持 RLDS 历史采样;回退到原始 RLDS loader(无历史)。
        logging.warning("[HAMLET] RLDS path has no history sampling; falling back to vanilla RLDS loader.")
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )

    return create_torch_data_loader_hamlet(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        history_length=history_length,
    )


def create_torch_data_loader_hamlet(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    history_length: int = HAMLET_HISTORY_LENGTH,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    dataset = create_torch_dataset_hamlet(
        data_config, action_horizon, model_config, history_length=history_length
    )
    # 注:HamletHistoryDataset 内部已经做了 transform(含 norm/tokenize),
    #     所以这里不再调 transform_dataset(避免二次 transform)。

    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"[HAMLET] local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoaderHamlet(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImplHamlet(data_config, data_loader)


class TorchDataLoaderHamlet:
    """与原 TorchDataLoader 逐字一致(仅类名不同,避免与原模块混淆)。"""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        self._sharding = sharding
        if sharding is None and framework == "jax":
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


class DataLoaderImplHamlet(DataLoader):
    """与原 DataLoaderImpl 一致:把 batch dict 转成 (Observation, Actions)。
    历史帧已经在 batch["image"] 里以 __hist{i} key 存在,Observation.from_dict
    会把它们一并放进 observation.images(因为 from_dict 直接用 data["image"])。
    pi0_hamlet 的 _split_history 再把它们拆出来。"""

    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoaderHamlet):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
