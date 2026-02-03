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
# 🟢 路径设置 (保持和你训练脚本一致)
# ==============================================================================
current_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入你的环境
try:
    from scripts.residual_env import ResidualPegEnv
except ImportError:
    from residual_env import ResidualPegEnv

def main():
    # ==============================================================================
    # ⚙️ [配置区域] - 请根据你的需求修改这里
    # ==============================================================================
    
    # 1. 场景选择 (是"断点续训"还是"新一轮Round"?)
    # True  = 开启 Round 4 (重置步数，不加载旧 Buffer，应用新 Scale)
    # False = 纯断点续训 (接上之前的步数和 Buffer，参数不变)
    START_NEW_ROUND = True 

    # 2. 路径配置
    # 上一轮训练好的模型 (.zip)
    LOAD_MODEL_PATH = os.path.join(project_root, "data/models/SAC_Residual_v1/rl_model_105000_steps.zip") # 👈 修改这里指向你的 checkpoint
    
    # 上一轮的 Replay Buffer (如果 START_NEW_ROUND=True，这个通常不用填，除非你想复用经验)
    LOAD_REPLAY_BUFFER = False 
    REPLAY_BUFFER_PATH = os.path.join(project_root, "data/models/SAC_Residual_v1/rl_model_replay_buffer_100000_steps.pkl")

    # Base Policy 路径 (保持不变)
    CKPT_RELATIVE_PATH = "diffusion_policy/data/outputs/2026.01.26/16.00.38_train_diffusion_unet_hybrid_peg_in_hole/checkpoints/base_policy.ckpt"
    BASE_CKPT_PATH = os.path.join(project_root, CKPT_RELATIVE_PATH)

    # 3. 新一轮参数 (Round 4 配置)
    NEW_RUN_NAME = "SAC_Residual_v2"  # 新的 Log 名字
    NEW_TOTAL_TIMESTEPS = 200_000      # 新一轮跑多少步
    
    # 👉 [关键修改] Round 4 我们把 Scale 加大到 0.03
    NEW_RESIDUAL_SCALE = 0.02 if START_NEW_ROUND else 0.01 
    
    # 其他环境参数 (保持和训练脚本一致)
    ACTION_CHUNK_SIZE = 4
    MAX_STEPS = 200     # 训练脚本里你改成了 200
    RESIDUAL_CLIP = 0.2

    # ==============================================================================
    # 🚀 脚本逻辑开始
    # ==============================================================================
    
    # 1. 检查 Base Policy
    if not os.path.exists(BASE_CKPT_PATH):
        print(f"❌ 错误: 找不到 Base Policy 文件: {BASE_CKPT_PATH}")
        return

    # 2. 创建环境 (Factory 模式，适配 VecEnv)
    print(f"🔧 初始化环境: Scale = {NEW_RESIDUAL_SCALE}, Steps = {MAX_STEPS}")
    
    def make_env():
        env = ResidualPegEnv(
            base_ckpt_path=BASE_CKPT_PATH,
            residual_scale=NEW_RESIDUAL_SCALE,  # <--- 应用新的 Scale
            residual_clip=RESIDUAL_CLIP,
            action_chunk_size=ACTION_CHUNK_SIZE,
            max_steps=MAX_STEPS,
            device='cuda:0'
        )
        # 设置 Monitor 记录 Log
        log_dir = os.path.join(project_root, "data", "logs", NEW_RUN_NAME)
        os.makedirs(log_dir, exist_ok=True)
        return Monitor(env, log_dir)

    # 包装成 DummyVecEnv (非常重要！因为 SAC.load 需要环境结构匹配)
    env = DummyVecEnv([make_env])

    # 3. 加载模型
    print(f"📥 正在加载模型: {LOAD_MODEL_PATH}")
    # custom_objects 可以用来覆盖旧模型里的一些参数，但这里我们主要靠环境改变
    model = SAC.load(
        LOAD_MODEL_PATH, 
        env=env, 
        device='cuda:0',
        print_system_info=True
    )

    # 4. 处理 Replay Buffer
    if LOAD_REPLAY_BUFFER and not START_NEW_ROUND:
        if os.path.exists(REPLAY_BUFFER_PATH):
            print(f"📥 正在加载 Replay Buffer: {REPLAY_BUFFER_PATH}")
            model.load_replay_buffer(REPLAY_BUFFER_PATH)
            print(f"✅ Buffer 加载完成，当前大小: {model.replay_buffer.size()}")
        else:
            print(f"⚠️ 警告: 找不到 Buffer 文件 {REPLAY_BUFFER_PATH}，将从空 Buffer 开始。")
    else:
        print("🆕 开启新一轮 / 不加载旧 Buffer，将从空 Buffer 开始重新收集适应新 Scale 的数据。")

    # 5. 设置回调函数
    save_dir = os.path.join(project_root, "data", "models", NEW_RUN_NAME)
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=save_dir,
        name_prefix="rl_model"
    )

    # 6. 开始训练
    # reset_num_timesteps=True: Tensorboard 从 0 开始画新图 (适合 Round 4)
    # reset_num_timesteps=False: 接在旧图后面 (适合断点续训)
    print(f"\n🚀 开始训练: {NEW_RUN_NAME}")
    print(f"🎯 目标步数: {NEW_TOTAL_TIMESTEPS}")
    print(f"📈 Tensorboard Log: data/logs/{NEW_RUN_NAME}")

    try:
        model.learn(
            total_timesteps=NEW_TOTAL_TIMESTEPS,
            callback=[checkpoint_callback],
            tb_log_name=NEW_RUN_NAME,
            reset_num_timesteps=START_NEW_ROUND, 
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n🛑 用户手动停止训练，正在保存最后一步...")

    # 保存最终模型
    model.save(os.path.join(save_dir, "rl_model_final"))
    print("✅ 续训/微调结束！")
    env.close()

if __name__ == "__main__":
    main()