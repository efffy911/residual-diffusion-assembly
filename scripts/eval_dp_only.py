import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# ==============================================================================
# 🟢 路径设置
# ==============================================================================
current_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入环境
from scripts.residual_env import ResidualPegEnv

def main():
    # --------------------------------------------------------------------------
    # 🔧 [测试配置] - 请在这里修改
    # --------------------------------------------------------------------------
    # 1. 你的 Base Policy 路径
    CKPT_RELATIVE_PATH = "diffusion_policy/data/outputs/2026.01.26/16.00.38_train_diffusion_unet_hybrid_peg_in_hole/checkpoints/base_policy.ckpt"
    BASE_CKPT_PATH = os.path.join(project_root, CKPT_RELATIVE_PATH)
    
    # 2. 关键变量：测试不同的 Chunk Size！
    # 建议分别测试 1, 4, 8，看看成功率怎么变
    TEST_CHUNK_SIZE = 4  
    
    # 3. 测试次数
    NUM_EPISODES = 50
    MAX_STEPS = 200
    
    # --------------------------------------------------------------------------
    # 1. 初始化环境
    # --------------------------------------------------------------------------
    if not os.path.exists(BASE_CKPT_PATH):
        print(f"❌ 错误: 找不到文件: {BASE_CKPT_PATH}")
        return

    print(f"🧊 加载 Base Policy 进行纯净测试 (No RL)...")
    print(f"📏 Action Chunk Size: {TEST_CHUNK_SIZE}")

    env = ResidualPegEnv(
        base_ckpt_path=BASE_CKPT_PATH,
        residual_scale=0.0,      # 👈 关键！设为 0，彻底屏蔽 RL 的影响
        action_chunk_size=TEST_CHUNK_SIZE,
        max_steps=MAX_STEPS,
        device='cuda:0'
    )
    
    # --------------------------------------------------------------------------
    # 2. 开始循环测试
    # --------------------------------------------------------------------------
    success_count = 0
    total_steps = 0
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(range(NUM_EPISODES), desc="Testing Base Policy")
    
    for i in pbar:
        obs, info = env.reset()
        terminated = False
        truncated = False
        step = 0
        
        while not (terminated or truncated):
            # 👇 核心：完全不给任何 RL 动作，只传 0
            # 这样 env 内部就会只执行 base_action + 0
            zero_action = np.zeros(3, dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(zero_action)
            step += 1
            
        total_steps += step
        
        if info.get("is_success", False):
            success_count += 1
            
        # 实时更新进度条上的成功率
        current_acc = (success_count / (i + 1)) * 100
        avg_len = total_steps / (i + 1)
        pbar.set_description(f"Success: {current_acc:.1f}% | AvgLen: {avg_len:.0f}")

    # --------------------------------------------------------------------------
    # 3. 输出最终结果
    # --------------------------------------------------------------------------
    final_acc = (success_count / NUM_EPISODES) * 100
    avg_len = total_steps / NUM_EPISODES
    
    print("\n" + "="*40)
    print(f"🏁 测试结果报告 (Chunk Size = {TEST_CHUNK_SIZE})")
    print("="*40)
    print(f"✅ 总成功率: {final_acc:.2f}% ({success_count}/{NUM_EPISODES})")
    print(f"⏱️ 平均步数: {avg_len:.1f}")
    print("="*40)
    
    if final_acc < 50.0:
        print("⚠️ 警告: Base Policy 在此环境下的表现显著低于 63%。")
        print("可能原因:")
        print("1. Action Chunk Size 设置不合理 (尝试改成 1 或 8 对比)")
        print("2. 图像归一化 (Normalize) 问题 (检查 residual_env.py 是否除以了255)")
        print("3. Z轴防撞逻辑 (Z-Gating) 误伤了 Base Policy (测试时可暂时注释掉 env 里的防撞逻辑)")
    else:
        print("🎉 Base Policy 表现正常！环境封装没有问题。")

    env.close()

if __name__ == "__main__":
    main()