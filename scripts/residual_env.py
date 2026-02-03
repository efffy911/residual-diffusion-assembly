import os
import sys
import gymnasium as gym
import numpy as np
import torch
import collections
from gymnasium import spaces
from omegaconf import OmegaConf
import hydra

# =========================
# Path Hack
# =========================
current_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(script_dir)
source_root = os.path.join(project_root, 'diffusion_policy')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if source_root not in sys.path:
    sys.path.insert(0, source_root)

import custom_envs


class ResidualPegEnv(gym.Env):
    """
    Residual RL Environment for Peg-in-Hole (Minimalist Pass-through Version)
    
    Logic:
    - Assume 'FrankaPegInHole-v0' ALREADY returns pre-processed images:
      (3, 96, 96), float32, range [0, 1].
    - We just pass them to the Base Policy directly, exactly like eval_policy_batch.py.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        base_ckpt_path,
        residual_scale=0.05,
        residual_clip=0.2,
        max_steps=400,  # 保持 400，与 Runner 一致
        device="cuda:0",
        action_chunk_size=1
    ):
        super().__init__()

        self.device = torch.device(device)
        self.residual_scale = residual_scale
        self.residual_clip = residual_clip
        self.max_steps = max_steps
        self.current_step = 0
        self.action_chunk_size = action_chunk_size

        # =========================
        # Underlying Environment
        # =========================
        self.env = gym.make(
            "FrankaPegInHole-v0",
            render_mode="rgb_array",
            control_mode="ee",
            disable_env_checker=True,
            # 注意：如果你的 custom_envs 里默认 max_episode_steps 就是 400，这里改不改都行
            max_episode_steps=max_steps, 
        )

        # 🟢 强制 Residual Policy 只输出 3 维动作
        self.residual_action_dim = 3
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.residual_action_dim,),
            dtype=np.float32,
        )

        # =========================
        # Residual Observation
        # =========================
        self.residual_obs_dim = 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.residual_obs_dim,),
            dtype=np.float32,
        )

        print("🧊 Loading Base Policy from:", base_ckpt_path)
        print("🔧 Config: Scale={}, Chunk={}, Pass-through Mode".format(
            residual_scale, action_chunk_size))

        # =========================
        # Load Base Policy
        # =========================
        self.base_policy = self._load_policy(base_ckpt_path)
        self.base_policy.eval()
        self.base_policy.to(self.device)

        for p in self.base_policy.parameters():
            p.requires_grad = False

        # =========================
        # Buffers
        # =========================
        self.n_obs_steps = 2
        self.obs_deque = collections.deque(maxlen=self.n_obs_steps)
        self.base_action_queue = collections.deque(maxlen=self.action_chunk_size)

    # -------------------------------------------------
    # Base Policy Loader
    # -------------------------------------------------
    def _load_policy(self, ckpt_path):
        run_dir = os.path.dirname(os.path.dirname(ckpt_path))
        cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config not found at {cfg_path}")

        cfg = OmegaConf.load(cfg_path)
        try:
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
        except Exception:
            from diffusion_policy.workspace.base_workspace import BaseWorkspace
            workspace = BaseWorkspace(cfg)

        workspace.load_checkpoint(ckpt_path)
        return workspace.model

    # -------------------------------------------------
    # Helper
    # -------------------------------------------------
    def _get_residual_obs(self, obs):
        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]
        pos_err = achieved - desired
        return pos_err.astype(np.float32)

    # -------------------------------------------------
    # Reset
    # -------------------------------------------------
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        self.base_policy.reset()
        self.obs_deque.clear()
        self.base_action_queue.clear()

        for _ in range(self.n_obs_steps):
            self.obs_deque.append(obs)

        self.current_step = 0
        return self._get_residual_obs(obs), info

    # -------------------------------------------------
    # Step
    # -------------------------------------------------
    def step(self, residual_action):
        # 1. 获取 Base 动作
        base_action = self._get_next_base_action()

        # =========================================================
        # 🛡️ Z-Axis Gating (防撞墙)
        # =========================================================
        current_obs = self.obs_deque[-1]
        current_pos_err = self._get_residual_obs(current_obs) 
        xy_err = np.linalg.norm(current_pos_err[:2])
        
        # 1cm 保护阈值
        if xy_err > 0.01:
            if residual_action[2] < 0:
                residual_action[2] = 0.0
        # =========================================================

        # 2. 动作合成
        residual_action = np.clip(
            residual_action, -self.residual_clip, self.residual_clip
        )
        scaled_residual = residual_action * self.residual_scale

        final_action = base_action.copy()
        final_action[:3] += scaled_residual
        final_action = np.clip(final_action, -1.0, 1.0)

        # 3. 环境执行
        obs, _, terminated, truncated, info = self.env.step(final_action)

        # =========================================================
        # Reward Calculation
        # =========================================================
        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]
        dist = np.linalg.norm(achieved - desired)

        # 1. 距离奖励 (保持不变)
        r_dist = 1.0 - np.tanh(10.0 * dist) - 1.0
        
        # 2. 成功奖励 (保持不变)
        r_success = 0.0
        if info.get("is_success", False):
            progress = 1.0 - self.current_step / self.max_steps
            r_success = 100.0 * progress

        # 🟢 3. [新增] 动作幅度惩罚 (Action Regularization)
        # 目的：让 SAC 学会"非必要不乱动"。
        # 使用 raw residual_action (范围通常是 -1 到 1)，而不是 scale 后的。
        # 系数 0.05 是经验值，配合 scale=0.01 使用效果很好。
        action_norm = np.linalg.norm(residual_action)
        r_penalty = -0.05 * (action_norm ** 2)

        # 4. 总奖励
        reward = r_dist + r_success + r_penalty

        # 🟢 [建议] 把分项奖励放进 info 里，方便在 TensorBoard 观察 SAC 是否在"偷懒"
        info["r_dist"] = r_dist
        info["r_success"] = r_success
        info["r_penalty"] = r_penalty
        # =========================================================

        self.obs_deque.append(obs)
        self.current_step += 1

        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_residual_obs(obs), reward, terminated, truncated, info

    # -------------------------------------------------
    # Base Policy Inference (极简透传版)
    # 🟢 完全复刻 eval_policy_batch.py 的数据流
    # -------------------------------------------------
    def _get_next_base_action(self):
        if len(self.base_action_queue) > 0:
            return self.base_action_queue.popleft()

        batch = {"image": [], "image_wrist": [], "state": []}
        
        for o in self.obs_deque:
            batch["image"].append(o["image"])
            batch["image_wrist"].append(o["image_wrist"])
            
            if "state" in o: s = o["state"]
            elif "observation" in o: s = o["observation"]
            else: s = np.zeros(19, dtype=np.float32)
            batch["state"].append(s)

        # 🟢 [直接转换] 不做任何 Resize, Permute 或 /255
        # 因为环境出来的已经是 (T, 3, 96, 96) 且是 [0, 1] 的 float 了
        t_img = torch.from_numpy(np.stack(batch["image"])).float().unsqueeze(0).to(self.device)
        t_wri = torch.from_numpy(np.stack(batch["image_wrist"])).float().unsqueeze(0).to(self.device)
        t_state = torch.from_numpy(np.stack(batch["state"])).float().unsqueeze(0).to(self.device)

        inp = {"image": t_img, "image_wrist": t_wri, "state": t_state}

        try:
            with torch.no_grad():
                result = self.base_policy.predict_action(inp)
        except Exception as e:
            print("\n❌ Model Inference Failed!")
            # 打印一下形状以便最后确认
            print(f"Input Img Shape: {t_img.shape} (Expected: 1, T, 3, 96, 96)")
            raise e

        all_actions = result["action"][0].cpu().numpy()
        chunk = all_actions[: self.action_chunk_size]
        for act in chunk:
            self.base_action_queue.append(act)

        return self.base_action_queue.popleft()

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()