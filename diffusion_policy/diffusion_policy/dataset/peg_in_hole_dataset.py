from typing import Dict, List
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
# 导入基类
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset

# 🟢 SafeLinearNormalizer 保持不变，非常好用
class SafeLinearNormalizer(LinearNormalizer):
    def normalize(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = x.copy()
        for key, value in x.items():
            if key in self.params_dict:
                out[key] = super().normalize({key: value})[key]
        return out

    def unnormalize(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = x.copy()
        for key, value in x.items():
            if key in self.params_dict:
                out[key] = super().unnormalize({key: value})[key]
        return out

class PegInHoleDataset(BaseImageDataset):
    def __init__(self,
            dataset_path: str,
            shape_meta: dict,
            n_obs_steps: int=2,
            horizon: int=16,
            pad_before: int=0,
            pad_after: int=0,
            seed: int=42,
            val_ratio: float=0.0,
            use_cache: bool=False,
            ):
        
        print(f"Loading ReplayBuffer from: {dataset_path}")
        self.replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=dataset_path, 
            mode='r'
        )

        # 1. 自动探测数据前缀 (适应新的 image 键名)
        prefix = ""
        # 先检查新的键名 'image'
        if 'image' in self.replay_buffer:
            prefix = ""
        elif 'data/image' in self.replay_buffer:
            prefix = "data/"
        # 兼容旧代码 (防止以后复用旧数据报错)
        elif 'img' in self.replay_buffer:
            prefix = ""
        elif 'data/img' in self.replay_buffer:
            prefix = "data/"
        else:
            # 最后的 fallback
            if hasattr(self.replay_buffer, 'root') and 'data/image' in self.replay_buffer.root:
                prefix = "data/"
        
        self.key_prefix = prefix
        print(f"Detected Data Prefix: '{prefix}'")

        # 🟢 [修改] 定义需要采样的 Key (包含双相机)
        # 注意: 这里的名字必须和 Zarr 文件里的键完全一致
        # 我们新的 Zarr 结构是: image, image_wrist, state, action
        self.target_keys = [
            f'{prefix}image', 
            f'{prefix}image_wrist',  # 新增
            f'{prefix}state', 
            f'{prefix}action'
        ]
        
        # 处理旧数据的兼容性 (如果 Zarr 里只有 img)
        if f'{prefix}img' in self.replay_buffer and f'{prefix}image' not in self.replay_buffer:
             self.target_keys[0] = f'{prefix}img'
             print("⚠️ Warning: Detected legacy key 'img'.")

        self.shape_meta = shape_meta
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        # 3. 初始化采样器
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=None,
            keys=self.target_keys 
        )
        
        # 4. 划分数据集
        n_episodes = self.replay_buffer.n_episodes
        indices = np.arange(n_episodes)
        val_size = int(n_episodes * val_ratio)
        if val_size > 0:
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(indices)
            self.val_mask = _create_mask(indices[:val_size], n_episodes)
            self.train_mask = _create_mask(indices[val_size:], n_episodes)
        else:
            self.train_mask = _create_mask(indices, n_episodes)
            self.val_mask = None

        # 5. 初始化归一化器
        self.normalizer = SafeLinearNormalizer()
        
        # 读取全量数据用于统计
        action_data = self.replay_buffer[f'{prefix}action'][:]
        state_data = self.replay_buffer[f'{prefix}state'][:]

        all_data = {
            'action': action_data,
            'state': state_data
        }
        self.normalizer.fit(data=all_data)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
            keys=self.sampler.keys
        )
        val_set.train_mask = None
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        return self.normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        p = self.key_prefix
        
        # 1. 获取数据 (兼容旧键名)
        key_img = f'{p}image' if f'{p}image' in sample else f'{p}img'
        
        img_global = sample[key_img]
        # 🟢 获取手眼相机数据
        img_wrist = sample[f'{p}image_wrist']
        
        state = sample[f'{p}state'] 
        action = sample[f'{p}action'] 

        # 2. 图像处理函数 (DRY: Don't Repeat Yourself)
        def process_img(img_numpy):
            img = torch.from_numpy(img_numpy).float()
            img = img / 255.0
            # img = img.permute(0, 3, 1, 2) # (T, H, W, C) -> (T, C, H, W)
            return img

        # 分别处理两张图
        img_global = process_img(img_global)
        img_wrist = process_img(img_wrist)
        
        # 3. 状态处理
        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).float()
        
        # 🟢 [关键] 返回字典的 key 必须和 peg_in_hole.yaml 里的 shape_meta 对应
        return {
            'obs': {
                'image': img_global,         # 对应 yaml: obs.image
                'image_wrist': img_wrist,    # 对应 yaml: obs.image_wrist
                'state': state
            },
            'action': action
        }

def _create_mask(indices, length):
    mask = np.zeros(length, dtype=bool)
    mask[indices] = True
    return mask