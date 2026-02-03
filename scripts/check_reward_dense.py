import os
import sys
import numpy as np

# 路径 Hack
current_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(script_dir)
source_root = os.path.join(project_root, 'diffusion_policy')
if project_root not in sys.path: sys.path.insert(0, project_root)
if source_root not in sys.path: sys.path.insert(0, source_root)

from scripts.residual_env import ResidualPegEnv

def main():
    CKPT_PATH = "diffusion_policy/data/outputs/2026.01.26/16.00.38_train_diffusion_unet_hybrid_peg_in_hole/checkpoints/base_policy.ckpt"
    abs_ckpt_path = os.path.join(project_root, CKPT_PATH)

    print("🚀 初始化最终检查...")
    env = ResidualPegEnv(
        base_ckpt_path=abs_ckpt_path,
        residual_scale=0.05,
        action_chunk_size=1,
        max_steps=200,
        device='cuda:0'
    )
    
    env.reset()
    print("\n✅ 开始运行 (跑 15 步，等待机器人启动)...")
    
    for i in range(15):
        # 采样一个随机动作
        action = env.action_space.sample()
        
        # ✅ 正确调用：调用 wrapper 的 step，而不是 env.env.step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 获取距离
        ag = obs['achieved_goal']
        dg = obs['desired_goal']
        dist = np.linalg.norm(ag - dg)
        
        print(f"Step {i:02d}: Dist={dist:.6f} | Reward={reward:.6f}")
        
    env.close()

if __name__ == "__main__":
    main()