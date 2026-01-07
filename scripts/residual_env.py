from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from torch import amp

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import os, sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

dp_repo = os.path.join(repo_root, "diffusion_policy")

if dp_repo not in sys.path:
    sys.path.insert(0, dp_repo)


import dill
import torch
import hydra

# 让 gym 能注册 FrankaPickAndPlaceSparse-v0
import panda_mujoco_gym  # noqa: F401


def force_time_limit(env, max_episode_steps: int):
    base = env
    while base.__class__.__name__ == "TimeLimit":
        base = base.env
    return TimeLimit(base, max_episode_steps=max_episode_steps)


def flat_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = []
        # 1. 原始部分
        for k in ["observation", "achieved_goal", "desired_goal"]:
            if k in obs:
                parts.append(np.asarray(obs[k], dtype=np.float32).reshape(-1))
        
        # -----------------------------------------------------------
        # 【核心修改】加入两组相对向量，覆盖 Pick 和 Place 全流程
        # -----------------------------------------------------------
        if "achieved_goal" in obs and "desired_goal" in obs and "observation" in obs:
            ag = np.asarray(obs["achieved_goal"], dtype=np.float32).reshape(-1)
            dg = np.asarray(obs["desired_goal"], dtype=np.float32).reshape(-1)
            
            # 假设 observation 的前3维是手爪坐标 (x,y,z)
            # 这是一个非常标准的设定，如果你的环境特殊，请确认一下
            eef_pos = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)[:3]

            # 向量1: 物体 -> 目标 (管 Place，解决运送偏差)
            vec_place = dg - ag
            
            # 向量2: 手 -> 物体 (管 Pick，解决抓取偏差)
            # 这就是你缺的那块拼图！有了它，Agent 才知道怎么把手挪过去抓
            vec_pick = ag - eef_pos 
            
            parts.append(vec_place)
            parts.append(vec_pick)
            # -----------------------------------------------------------

        if not parts:
            raise ValueError(f"Empty obs dict keys={list(obs.keys())}")
        return np.concatenate(parts, axis=0)
    
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def build_obs_hist(obs_vec: np.ndarray, n_obs_steps: int) -> np.ndarray:
    return np.repeat(obs_vec[None, :], n_obs_steps, axis=0)


@dataclass
class ResidualConfig:
    checkpoint: str
    device: str = "cuda:0"
    env_id: str = "FrankaPickAndPlaceSparse-v0"
    max_steps: int = 150
    n_obs_steps: int = 2

    # residual 相关
    delta_scale: float = 0.1          # 你需要调（看 action range）
    freeze_gripper: bool = True       # 强烈建议先 True
    gripper_dim: Optional[int] = None # 如果 freeze_gripper=True，需要告诉我夹爪是第几维（常见是最后一维）

    # ✅ NEW: DP rollout / speed-stability knobs
    dp_inference_steps: int = 4      # 覆盖 dp_cfg.policy.num_inference_steps
    dp_replan_every: int = 4         # 每 K 步重规划一次（缓存长度）
    warmup_steps: int = 50_000       # 前 warmup_steps 不施加 residual（delta=0）
    ramp_steps: int = 10_000         # ramp up residual over this many steps
    train_seed_pool: Optional[list[int]] = None
    seed: int = 0

class ResidualPickPlaceEnv(gym.Env):
    """
    SB3 看到的 action 是 delta（残差），obs 是 flat(obs_hist)。
    env 内部执行：a = clip(a_dp + delta_scale * delta)
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg: ResidualConfig, render: bool = False):
        super().__init__()
        self.cfg = cfg
        self.render = render

        # --- create env ---
        self.env = gym.make(
            cfg.env_id,
            render_mode="human" if render else None
        )
        self.env = force_time_limit(self.env, cfg.max_steps)

        # --- load diffusion policy (frozen) ---
        payload = torch.load(open(cfg.checkpoint, "rb"), pickle_module=dill)
        self.dp_cfg = payload["cfg"]
        if hasattr(self.dp_cfg, "policy") and hasattr(self.dp_cfg.policy, "num_inference_steps"):
            self.dp_cfg.policy.num_inference_steps = int(cfg.dp_inference_steps)

        print("[DP CONFIG] num_inference_steps =", self.dp_cfg.policy.num_inference_steps,
            "horizon =", getattr(self.dp_cfg.policy, "horizon", None))
        cls = hydra.utils.get_class(self.dp_cfg._target_)
        workspace = cls(self.dp_cfg, output_dir="/tmp/_residual_rl_dummy")
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if getattr(self.dp_cfg.training, "use_ema", False):
            policy = workspace.ema_model
        self.dp_policy = policy

        self.torch_device = torch.device(cfg.device)
        self.dp_policy.to(self.torch_device)
        self.dp_policy.eval()

        # --- spaces ---
        # SB3 学 delta，空间用 env.action_space 形状（范围我们设为 [-1,1]，再乘 delta_scale）
        act_shape = self.env.action_space.shape
        self.action_space = gym.spaces.Box(
            low=-0.5, high=0.5, shape=act_shape, dtype=np.float32
        )
        # obs = (n_obs_steps, obs_dim) -> flatten
        obs0, _ = self.env.reset(seed=self.cfg.seed)
        obs_vec0 = flat_obs(obs0)
        self.obs_dim = obs_vec0.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(cfg.n_obs_steps * self.obs_dim,),
            dtype=np.float32
        )

        # internal state
        self.obs_hist = build_obs_hist(obs_vec0, cfg.n_obs_steps)
        self.t = 0
        self.total_steps = 0  # ✅ add this
        self.last_dp_action = None

        # infer gripper dim if not set
        if cfg.freeze_gripper and cfg.gripper_dim is None:
            # 常见：最后一维是 gripper
            cfg.gripper_dim = act_shape[0] - 1
        
        self._dp_action_cache = None
        self._dp_action_idx = 0
        self.dp_replan_every = 4

        self._train_seed0 = 0          # 你可以从 cfg.seed 传进来
        self._train_ep_idx = 0
        self._rng = np.random.default_rng(self.cfg.seed)


    def _dp_act(self, obs_hist: np.ndarray) -> np.ndarray:
        """
        Diffusion policy action with sequence caching.
        【修正注】修复了 shape 报错。DP 需要 (Batch, T, D) 输入，
        所以切片后不能 flatten，要保持 (T, D)。
        """
        # 如果缓存还有动作，直接用
        if self._dp_action_cache is not None and self._dp_action_idx < len(self._dp_action_cache):
            a = self._dp_action_cache[self._dp_action_idx]
            self._dp_action_idx += 1
            return a

        # ---------------------------------------------------------
        # 1. Reshape 还原成 (n_steps, obs_dim_new)
        # obs_hist 是平铺的，先还原
        obs_seq = obs_hist.reshape(self.cfg.n_obs_steps, -1)

        # 2. 切片：去掉最后 6 维 (我们新增的 relative goal)
        # DP 原始模型没见过这 6 维，必须切掉
        obs_seq_dp = obs_seq[:, :-6]

        # 3. 【关键修正】这里不要 flatten()！
        # DP Policy 期待的输入是 (Batch, Time, Dim)
        # 这里我们要保留 (Time, Dim)，加上后面的 [None, ...] 就会变成 (1, Time, Dim)
        obs_in = obs_seq_dp.astype(np.float32)
        # ---------------------------------------------------------

        with torch.no_grad():
            # 可选：fp16 自动混合精度
            with amp.autocast(device_type="cuda", enabled=self.torch_device.type == "cuda"):
                # obs_in[None, ...] 会把 (T, D) 变成 (1, T, D)，满足 DP 要求
                if hasattr(self.dp_policy, "predict_action"):
                    out = self.dp_policy.predict_action({"obs": obs_in[None, ...]})
                    a = out["action"]
                else:
                    out = self.dp_policy({"obs": obs_in[None, ...]})
                    a = out["action"]

        a = a.detach().to("cpu").numpy()

        # 处理形状
        if a.ndim == 3:      # (B, H, A)
            a_seq = a[0]     # (H, A)
        elif a.ndim == 2:    # (B, A)
            a_seq = a        # (1, A)
        else:
            raise ValueError(f"Unexpected action shape from DP: {a.shape}")

        # 缓存整段动作序列
        self._dp_action_cache = a_seq.astype(np.float32)

        K = int(self.cfg.dp_replan_every)
        self._dp_action_cache = self._dp_action_cache[:K]

        self._dp_action_idx = 1
        return self._dp_action_cache[0]

    def _compose_action(self, a_dp: np.ndarray, delta: np.ndarray) -> np.ndarray:
        # delta ∈ [-1,1] -> scale
        a = a_dp + self.cfg.delta_scale * delta

        if self.cfg.freeze_gripper:
            gd = int(self.cfg.gripper_dim)
            a[gd] = a_dp[gd]

        # clip to env action bounds
        low = self.env.action_space.low
        high = self.env.action_space.high
        return np.clip(a, low, high).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return self.obs_hist.reshape(-1).astype(np.float32)

    def _compute_reward(self, obs: Any, info: Dict[str, Any], delta: np.ndarray) -> float:
        """
        Residual RL reward (Generalization Tuned)
        
        修改逻辑：
        1. 降低 distance 权重：防止 Residual Agent 破坏 Diffusion Policy 的自然轨迹。
        2. 增加 delta 正则惩罚：鼓励 "Lazy Agent"，非必要不修改动作，防止过拟合。
        """

        # -------- 1. Step Penalty (时间成本) --------
        # 保持小一点，鼓励尽快完成，但不要让它急得乱撞
        step_penalty = -0.01

        # -------- 2. 距离项 (Guidance) --------
        ag = obs["achieved_goal"]
        dg = obs["desired_goal"]
        dist = float(np.linalg.norm(ag - dg))

        # 【关键修改】降低权重 (原 -1.0 -> -0.5 或 -0.25)
        # 只要能提供梯度方向即可，不要让它主导 Reward，
        # 否则 Agent 会试图为了缩短 1mm 距离而破坏抓取姿态。
        goal_term = -0.5 * dist

        # -------- 3. Success Bonus (稀疏奖励 - 核心目标) --------
        # 这是最重要的信号，只有完成了才给大奖
        success_bonus = 10.0 if info.get("is_success", False) else 0.0

        # -------- 4. Residual Regularization (正则项) --------
        delta_reg = -0.001 * float(np.sum(np.square(delta)))

        # -------- Total --------
        reward = step_penalty + goal_term + success_bonus + delta_reg
        return reward


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._dp_action_cache = None
        self._dp_action_idx = 0

        if seed is None:
            pool = getattr(self.cfg, "train_seed_pool", None)
            if pool is not None and len(pool) > 0:
                seed = int(pool[self._rng.integers(0, len(pool))])
            else:
                seed = int(getattr(self.cfg, "seed", 0)) + self._train_ep_idx
            self._train_ep_idx += 1

        obs, info = self.env.reset(seed=seed)

        obs_vec = flat_obs(obs)
        self.obs_hist = build_obs_hist(obs_vec, self.cfg.n_obs_steps)
        self.t = 0
        self.last_dp_action = None
        return self._get_obs(), info


    def step(self, action: np.ndarray):
        # action is delta from SB3
        delta = np.asarray(action, dtype=np.float32)
        
        # warmup：前 N step 不施加 residual（强烈建议）
        warmup = int(self.cfg.warmup_steps)
        ramp = int(self.cfg.ramp_steps)
        
        # residual gate: 0 -> 1 逐渐放开
        if self.total_steps < warmup:
            scale = 0.0
        elif self.total_steps < warmup + ramp:
            scale = (self.total_steps - warmup) / float(ramp)
        else:
            scale = 1.0

        delta *= scale

        # 1) clip 每维（硬约束）
        delta = np.clip(delta, -0.5, 0.5)   # 先用 0.5，想更保守就 0.3

        # 2) clip 范数（更关键）
        max_norm = 0.5
        norm = float(np.linalg.norm(delta))
        if norm > max_norm:
            delta *= (max_norm / (norm + 1e-8))

        # 获取 DP 动作
        a_dp = self._dp_act(self.obs_hist)
        self.last_dp_action = a_dp.copy()
        
        # 合成最终动作 (Final Action = DP + Residual)
        a = self._compose_action(a_dp, delta)

        # =========================================================================
        # ⚠️⚠️⚠️ 鲁棒性测试专用区域 (Robustness Test Zone) ⚠️⚠️⚠️
        # =========================================================================
        # 【说明】：以下 3 行代码仅在运行 eval_robustness (加噪测试) 时取消注释！
        # 【警告】：正常训练 (train) 或普通测试 (eval) 时，必须将其注释掉，否则无法收敛！
        # -------------------------------------------------------------------------
        
        # noise_scale = 0.1  # 噪声强度：0.05(轻微), 0.1(显著), 0.2(剧烈)
        # noise = np.random.normal(loc=0.0, scale=noise_scale, size=a.shape)
        # a = a + noise
        
        # =========================================================================
        # ⚠️⚠️⚠️ 区域结束 ⚠️⚠️⚠️
        # =========================================================================

        # 执行动作
        obs, env_reward, terminated, truncated, info = self.env.step(a)
        self.t += 1
        self.total_steps += 1

        # update obs_hist
        obs_vec = flat_obs(obs)
        self.obs_hist = np.concatenate([self.obs_hist[1:], obs_vec[None, :]], axis=0)

        # residual RL reward
        reward = self._compute_reward(obs, info, delta)

        # 建议：成功就提前结束（压步数）
        if info.get("is_success", False):
            terminated = True

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.env.close()
