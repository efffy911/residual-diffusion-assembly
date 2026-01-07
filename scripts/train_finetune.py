# scripts/train_finetune.py

import os
import time
from dataclasses import dataclass
import torch
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# 复用之前的配置和环境
from scripts.residual_env import ResidualPickPlaceEnv, ResidualConfig
from scripts.train_residual import make_env, StepSpeedCallback, ResidualEvalCallback

# ===================== CONFIG =====================

# 1. 这里的配置必须跟训练时完全一致！
ENV_CFG = ResidualConfig(
    checkpoint="./diffusion_policy/data/outputs/2025.12.18/21.05.00_train_diffusion_unet_lowdim_pickplace_lowdim/checkpoints/latest.ckpt",
    device="cuda:0",
    env_id="FrankaPickAndPlaceSparse-v0",
    max_steps=150,
    n_obs_steps=2,
    delta_scale=0.005,
    freeze_gripper=True,
    gripper_dim=None,
    
    # 保持原来的 Replan 逻辑
    dp_inference_steps=4,
    dp_replan_every=4, # 保持 4
    
    warmup_steps=3_000,
    ramp_steps=2_000,
    
    # 泛化训练：Seed Pool 为 None
    seed=0,
    train_seed_pool=None, 
)

@dataclass
class FinetuneCfg:
    # ===== 关键路径 =====
    # 这里填你那个 75% 胜率的 best 模型路径
    load_path: str = "./data/residual_sac_runs/run_0/best_sac_residual.zip"
    save_dir: str = "./data/residual_sac_runs/run_0_finetune"
    
    # ===== 关键参数 =====
    # 学习率降低 10 倍 (3e-4 -> 3e-5)
    new_learning_rate: float = 3e-5
    
    # 再跑 100k 步 (足够收敛了)
    timesteps: int = 100_000
    
    # 保持并行的配置
    n_envs: int = 4
    train_freq: int = 1
    gradient_steps: int = 4  # 保持 1:1 的数据消化率
    
    # Eval 配置
    eval_every_steps: int = 10_000
    n_eval_episodes: int = 20
    eval_seed: int = 99999 # 用个没见过的 seed 测泛化

CFG = FinetuneCfg()

# =================================================

def main():
    os.makedirs(CFG.save_dir, exist_ok=True)
    
    print(f"[INFO] Fine-tuning starts. Loading from: {CFG.load_path}")
    print(f"[INFO] New Learning Rate: {CFG.new_learning_rate}")

    # 1. 创建并行环境 (跟原来保持一致)
    env = make_vec_env(
        make_env(ENV_CFG, render=False),
        n_envs=CFG.n_envs,
        seed=1000, # 稍微换个随机种子偏移量
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"}
    )

    # 2. 加载模型
    # custom_objects 会覆盖保存时的参数，这是修改 LR 最优雅的方法
    model = SAC.load(
        CFG.load_path, 
        env=env, 
        device="cuda:0",
        custom_objects={
            "learning_rate": CFG.new_learning_rate,
            "lr_schedule": lambda _: CFG.new_learning_rate # 强制覆盖 Schedule
        }
    )

    # ⚠️ 双重保险：手动更新优化器里的 LR
    # 因为有时候 load 完优化器状态会恢复旧的
    for optimizer in [model.actor.optimizer, model.critic.optimizer, model.ent_coef_optimizer]:
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = CFG.new_learning_rate

    # 3. 设置 Callbacks
    speed_cb = StepSpeedCallback(print_every=2000)
    
    # Eval 环境 (Unseen Seed)
    def make_eval_env():
        res_cfg_eval = ResidualConfig(
            checkpoint=ENV_CFG.checkpoint,
            device=ENV_CFG.device,
            env_id=ENV_CFG.env_id,
            max_steps=ENV_CFG.max_steps,
            n_obs_steps=ENV_CFG.n_obs_steps,
            delta_scale=ENV_CFG.delta_scale,
            freeze_gripper=ENV_CFG.freeze_gripper,
            gripper_dim=ENV_CFG.gripper_dim,
            dp_inference_steps=ENV_CFG.dp_inference_steps,
            dp_replan_every=ENV_CFG.dp_replan_every,
            warmup_steps=0, # 残差全开
            ramp_steps=0,
            seed=0,
            train_seed_pool=None,
        )
        return ResidualPickPlaceEnv(res_cfg_eval, render=False)

    eval_cb = ResidualEvalCallback(
        eval_env_fn=make_eval_env,
        eval_every_steps=CFG.eval_every_steps,
        n_eval_episodes=CFG.n_eval_episodes,
        save_dir=CFG.save_dir,
        best_name="best_finetuned", # 保存为新名字
        eval_seed=CFG.eval_seed,
        strict_success_end=False,
        save_with_step=True,
    )

    cb = CallbackList([speed_cb, eval_cb])

    # 4. 开始微调
    # reset_num_timesteps=False 会让日志里的 step 接着之前的计数 (比如从 75k 开始画图)
    model.learn(total_timesteps=CFG.timesteps, callback=cb, reset_num_timesteps=False)
    
    # 5. 保存最终结果
    model.save(os.path.join(CFG.save_dir, "finetuned_final.zip"))
    print("[DONE] Fine-tuning finished.")
    env.close()

if __name__ == "__main__":
    main()