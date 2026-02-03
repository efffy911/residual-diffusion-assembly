import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from typing import Dict, Any
import numpy as np
import torch
import collections
import cv2
import gymnasium as gym 

# 继承 BaseImageRunner
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

# 路径补全
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
if project_root not in sys.path:
    sys.path.append(project_root)

import custom_envs 

class FrankaPegInHoleRunner(BaseImageRunner):
    def __init__(self, 
                 output_dir, 
                 n_train=0, 
                 n_test=10, 
                 max_steps=400, 
                 steps_per_render=10, 
                 fps=10, 
                 crf=22, 
                 tqdm_interval_sec=5.0, 
                 render_size=96, 
                 past_action=False, 
                 abs_action=True, 
                 n_obs_steps=2
                 ):
        super().__init__(output_dir)
        self.n_test = n_test
        self.max_steps = max_steps
        self.render_size = render_size
        self.n_obs_steps = n_obs_steps

        # 🟢 1. 创建环境 (确保 disable_env_checker 以避免 key 检查报错)
        try:
            self.env = gym.make('FrankaPegInHole-v0', render_mode='rgb_array', control_mode='ee', disable_env_checker=True)
            print("✅ Runner: Created env with control_mode='ee'")
        except TypeError:
            self.env = gym.make('FrankaPegInHole-v0', render_mode='rgb_array', disable_env_checker=True)
            print("⚠️ Runner: Created env with default args (fallback)")

    def _get_obs(self, obs_dict):
        """
        🟢 [核心修改] 从环境返回的 obs 中提取 双相机图像 和 状态
        """
        # A. 获取图像
        # 情况 1: 环境直接返回了图像 (推荐，因为我们在 custom_envs 里写好了)
        if isinstance(obs_dict, dict) and 'image' in obs_dict and 'image_wrist' in obs_dict:
            img_global = obs_dict['image']       # (3, 96, 96) float32 [0,1]
            img_wrist  = obs_dict['image_wrist'] # (3, 96, 96) float32 [0,1]
            
            # 转回 (H, W, C) uint8 [0,255] 以便 resize 和存入 deque
            # 因为我们的 custom_envs 输出的是 (C, H, W)，所以需要 transpose
            img_global = np.moveaxis(img_global, 0, -1) * 255.0
            img_wrist  = np.moveaxis(img_wrist, 0, -1) * 255.0
            
            img_global = img_global.astype(np.uint8)
            img_wrist  = img_wrist.astype(np.uint8)

        # 情况 2: 环境没返回图像，需要手动 Render (Fallback)
        else:
            print("⚠️ Runner: Obs dict missing images, fallback to env.render()")
            try:
                # 注意: 这里只能拿到一张默认图，双摄模式下这其实会崩
                img_global = self.env.render()
                img_wrist = np.zeros_like(img_global) # 假数据
            except Exception:
                img_global = np.zeros((self.render_size, self.render_size, 3), dtype=np.uint8)
                img_wrist = np.zeros((self.render_size, self.render_size, 3), dtype=np.uint8)

        # B. 统一 Resize (保底)
        def resize_if_needed(img):
            if img.shape[0] != self.render_size or img.shape[1] != self.render_size:
                return cv2.resize(img, (self.render_size, self.render_size))
            return img
            
        img_global = resize_if_needed(img_global)
        img_wrist = resize_if_needed(img_wrist)

        # C. 获取状态
        if isinstance(obs_dict, dict) and 'observation' in obs_dict:
            state = obs_dict['observation']
        elif isinstance(obs_dict, dict) and 'state' in obs_dict:
            state = obs_dict['state']
        else:
            # 如果都没有，尝试用 agent_pos
            state = obs_dict.get('agent_pos', np.zeros(19, dtype=np.float32))

        return img_global, img_wrist, state

    def run(self, policy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        n_envs = self.n_test
        log_data = collections.defaultdict(list)

        print(f"Starting evaluation on {n_envs} episodes (Dual-Cam Mode)...")

        for i in range(n_envs):
            # Reset 环境
            raw_obs, _ = env.reset()
            
            # 获取初始帧
            img_g, img_w, state = self._get_obs(raw_obs)

            # 🟢 2. 初始化双相机的历史 Buffer
            # deque 里面存的是 numpy (H, W, C) uint8
            img_g_history = collections.deque([img_g] * self.n_obs_steps, maxlen=self.n_obs_steps)
            img_w_history = collections.deque([img_w] * self.n_obs_steps, maxlen=self.n_obs_steps)
            state_history = collections.deque([state] * self.n_obs_steps, maxlen=self.n_obs_steps)

            done = False
            policy.reset()
            episode_reward = 0
            
            steps = 0
            while not done and steps < self.max_steps:
                # --- A. 构造 Policy 输入 ---
                # 辅助函数: 处理图像序列
                def prepare_img_tensor(history_deque):
                    # Stack -> (T, H, W, C)
                    imgs_np = np.stack(history_deque)
                    # To Torch & Norm -> (T, C, H, W)
                    imgs_torch = torch.from_numpy(imgs_np).float() / 255.0
                    imgs_torch = imgs_torch.permute(0, 3, 1, 2) 
                    # Add Batch -> (1, T, C, H, W)
                    return imgs_torch.unsqueeze(0).to(device, dtype=dtype)

                # 1. 准备两个相机的 Tensor
                tensor_global = prepare_img_tensor(img_g_history)
                tensor_wrist  = prepare_img_tensor(img_w_history)

                # 2. 堆叠历史状态: (1, T, D)
                states_np = np.stack(state_history)
                states_torch = torch.from_numpy(states_np).float().unsqueeze(0).to(device, dtype=dtype)

                # 🟢 [关键] 构造符合 Policy 要求的字典
                # 必须包含: image, image_wrist, state
                obs_dict_input = {
                    'image': tensor_global,       # 对应 yaml: obs.image
                    'image_wrist': tensor_wrist,  # 对应 yaml: obs.image_wrist
                    'state': states_torch
                }
                
                # --- B. 推理 ---
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict_input)
                
                # 获取动作: (B, T, D) -> 取 batch 0, time 0 -> (D,)
                action = action_dict['action'][0, 0].cpu().numpy()

                # --- C. 执行 ---
                action_to_env = action.flatten()[:4] # 只取前4维 (XYZ+G)

                try:
                    raw_obs, reward, terminated, truncated, info = env.step(action_to_env)
                except ValueError:
                    print(f"Warning: Env rejected action, padding...")
                    padded = np.zeros(7)
                    padded[:4] = action_to_env
                    raw_obs, reward, terminated, truncated, info = env.step(padded)

                done = terminated or truncated
                episode_reward += reward
                steps += 1

                # --- D. 更新历史 ---
                new_img_g, new_img_w, new_state = self._get_obs(raw_obs)
                
                img_g_history.append(new_img_g)
                img_w_history.append(new_img_w)
                state_history.append(new_state)

            print(f"Episode {i} finished. Steps: {steps}, Reward: {episode_reward:.4f}")
            log_data['test/mean_episode_reward'].append(episode_reward)
            log_data['test_mean_score'].append(episode_reward)
            
        # 计算平均值
        final_log = dict()
        for key, values in log_data.items():
            final_log[key] = np.mean(values)
            
        return final_log