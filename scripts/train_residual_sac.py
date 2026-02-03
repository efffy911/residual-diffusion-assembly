import os
import sys
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# ==============================================================================
# 🟢 路径设置 & 导入环境
# ==============================================================================
current_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 确保 residual_env.py 就在 scripts 文件夹下
try:
    from scripts.residual_env import ResidualPegEnv
except ImportError:
    # 如果你在根目录运行，尝试直接导入
    from residual_env import ResidualPegEnv

def main():
    # --------------------------------------------------------------------------
    # 🔧 [核心超参数 - Round 6 Rebirth]
    # --------------------------------------------------------------------------
    RUN_NAME = "SAC_Residual_v1"   # 给他起个响亮的名字
    TOTAL_TIMESTEPS = 500_000                  # 建议跑久一点，反正很快
    RESIDUAL_SCALE = 0.01                      # 保持 0.01 不变
    SEED = 42
    
    # 👇 [请务必修改] 指向你的 Diffusion Policy 权重文件
    # 这是你之前训练好的那个 63% 成功率的模型
    CKPT_RELATIVE_PATH = "diffusion_policy/data/outputs/2026.01.26/16.00.38_train_diffusion_unet_hybrid_peg_in_hole/checkpoints/base_policy.ckpt"
    BASE_CKPT_PATH = os.path.join(project_root, CKPT_RELATIVE_PATH)
    
    # --------------------------------------------------------------------------
    # 1. 创建环境 (Factory)
    # --------------------------------------------------------------------------
    if not os.path.exists(BASE_CKPT_PATH):
        print(f"❌ 错误: 找不到 Base Policy 文件，请检查路径:\n{BASE_CKPT_PATH}")
        return

    def make_env():
        env = ResidualPegEnv(
            base_ckpt_path=BASE_CKPT_PATH,
            residual_scale=RESIDUAL_SCALE,
            residual_clip=0.2,       # 这里的 clip 对应 env 里的设置
            action_chunk_size=4,     
            max_steps=200,
            device='cuda:0'          # 确保 base policy 在 GPU 上
        )
        # Monitor 用于记录 Reward 曲线到 Tensorboard
        log_dir = os.path.join(project_root, "data", "logs", RUN_NAME)
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        return env

    # 向量化环境
    env = DummyVecEnv([make_env])

    # --------------------------------------------------------------------------
    # 2. 定义 SAC 模型
    # --------------------------------------------------------------------------
    # 🔴 关键修改：从 "MultiInputPolicy" 改为 "MlpPolicy"
    # 因为现在的 Observation 只有 3 个数字 (x,y,z error)，不需要 CNN 处理图像
    
    policy_kwargs = dict(
        net_arch=[256, 256],  # 网络不用太大，256足够了
    )

    model = SAC(
        "MlpPolicy",          # 👈 [重点] 纯向量输入用 MLP
        env,
        verbose=1,
        seed=SEED,
        
        # --- 优化后的参数 ---
        learning_rate=3e-4,   # 标准学习率即可，因为 Dense Reward 很好学
        buffer_size=100_000,
        batch_size=256,
        
        # 自动调整熵。因为维度低 (3维)，SAC 自动调参会非常准，不用我们操心。
        ent_coef='auto',      
        
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(project_root, "data", "tensorboard"),
        device='cuda:0'
    )

    # --------------------------------------------------------------------------
    # 3. 设置回调
    # --------------------------------------------------------------------------
    save_dir = os.path.join(project_root, "data", "models", RUN_NAME)
    
    # 每 5000 步保存一次 (因为跑得快，可以存勤快点)
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=save_dir,
        name_prefix="rl_model"
    )

    print(f"\n🚀 [Round 6] 凤凰涅槃 - 开始训练!")
    print(f"👀 观测空间: 3维 (Pos Error)")
    print(f"🎯 奖励机制: 距离引导 (Dense) + 时间缩放成功奖励")
    print(f"📂 模型保存: {save_dir}")
    print(f"📈 监控命令: tensorboard --logdir data/tensorboard\n")

    # --------------------------------------------------------------------------
    # 4. 开始炼丹
    # --------------------------------------------------------------------------
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n🛑 用户手动停止训练，正在保存最后一步...")
    
    # 保存最终模型
    model.save(os.path.join(save_dir, "rl_model_final"))
    print("✅ 训练结束。去 Tensorboard 看看这一轮的曲线有多漂亮吧！")
    env.close()

if __name__ == "__main__":
    main()