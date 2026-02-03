import os
import sys
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from tqdm import tqdm

# =========================
# 路径 Hack
# =========================
current_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(script_dir)
source_root = os.path.join(project_root, 'diffusion_policy')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if source_root not in sys.path:
    sys.path.insert(0, source_root)

# 引入你的环境
from scripts.residual_env import ResidualPegEnv 

def main():
    # ================= 配置区域 =================
    # 1. 填入你 Base Policy 的权重路径
    BASE_CKPT = "diffusion_policy/data/outputs/2026.01.26/16.00.38_train_diffusion_unet_hybrid_peg_in_hole/checkpoints/base_policy.ckpt" 
    
    # 2. 填入你刚刚训练好的 SAC 模型路径 (best_model.zip 或 latest)
    # 通常在 tensorboard_logs/你的实验名/model_checkpoints/ 或者是 final_model.zip
    RESIDUAL_MODEL_PATH = "data/models/SAC_Residual_v1/rl_model_105000_steps.zip" 

    CHUNK_SIZE = 4
    MAX_STEPS = 200
    N_EPISODES = 50  # 测 50 次看看实力
    # ===========================================

    print(f"🧊 Loading Environment with Base Policy...")
    # 注意：测试时 residual_scale 要和训练时保持一致！(0.01)
    env = ResidualPegEnv(
        base_ckpt_path=BASE_CKPT,
        residual_scale=0.01,    # 👈 必须是 0.01
        residual_clip=0.2,
        max_steps=MAX_STEPS,
        action_chunk_size=CHUNK_SIZE,
        device="cuda:0"
    )

    print(f"🔥 Loading Residual Policy (SAC) from: {RESIDUAL_MODEL_PATH}")
    model = SAC.load(RESIDUAL_MODEL_PATH)

    success_count = 0
    pbar = tqdm(range(N_EPISODES))

    for i in pbar:
        obs, _ = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            # 🟢 关键：deterministic=True (关掉 hand shaking)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            if info.get('is_success', False):
                success_count += 1
                done = True # 提前结束

        current_sr = success_count / (i + 1)
        pbar.set_postfix({"Success Rate": f"{current_sr:.1%}"})

    print("\n" + "="*50)
    print(f"📊 最终成绩 (Chunk={CHUNK_SIZE}, Scale=0.01)")
    print(f"✅ Success Rate: {success_count/N_EPISODES:.2%} ({success_count}/{N_EPISODES})")
    print("="*50)

if __name__ == "__main__":
    main()