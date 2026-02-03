import os
import sys
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from tqdm import tqdm
import pandas as pd # 稍微用一下 pandas 来打印漂亮的表格，如果没有安装可以注释掉

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

# =========================
# 🛠️ 噪声注入函数 (核心修改)
# =========================
def inject_observation_noise(obs, noise_level):
    """
    模拟传感器/标定误差。
    假设 obs 的前3维是末端位置或者位置误差 (x, y, z)。
    """
    if noise_level <= 0.0:
        return obs
    
    # 创建噪声：均匀分布 [-noise_level, +noise_level]
    # 模拟每一帧的传感器抖动 (Sensor Noise)
    # 如果想模拟固定的标定偏差 (Calibration Bias)，可以在 reset 时生成一次固定噪声
    noise = np.random.uniform(-noise_level, noise_level, size=3)
    
    noisy_obs = obs.copy()
    noisy_obs[:3] += noise # 只污染位置信息
    return noisy_obs

# =========================
# 🧪 单轮评估函数
# =========================
def evaluate_policy_with_noise(env, model, n_episodes, noise_level, deterministic=True):
    success_count = 0
    
    # 进度条描述
    pbar = tqdm(range(n_episodes), desc=f"Testing Noise ±{noise_level*1000:.1f}mm")

    for i in pbar:
        real_obs, _ = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            # 🔴 关键步骤：欺骗 Agent
            # 1. 获取加了噪声的观测 (Agent 以为自己在的位置)
            noisy_obs = inject_observation_noise(real_obs, noise_level)
            
            # 2. Agent 基于错误的观测做出决策
            # 注意：Round 2 Scale 0.01 我们需要它的 Dither 效果
            # 如果 SAC 训练得非常确定性，这里用 deterministic=True 可能会导致不震动
            # 如果发现效果不好，可以尝试改成 deterministic=False 试试
            action, _ = model.predict(noisy_obs, deterministic=deterministic)
            
            # 3. 环境执行动作 (基于真实的物理世界)
            real_obs, reward, done, truncated, info = env.step(action)

            if info.get('is_success', False):
                success_count += 1
                done = True 

        current_sr = success_count / (i + 1)
        pbar.set_postfix({"Success": f"{current_sr:.1%}"})
    
    return success_count / n_episodes

def main():
    # ================= 配置区域 =================
    # 1. Base Policy
    BASE_CKPT = "diffusion_policy/data/outputs/2026.01.26/16.00.38_train_diffusion_unet_hybrid_peg_in_hole/checkpoints/base_policy.ckpt" 
    
    # 2. 你的 Round 2 最强模型 (Scale 0.01)
    RESIDUAL_MODEL_PATH = "data/models/SAC_Residual_v1/rl_model_105000_steps.zip" 
    
    CHUNK_SIZE = 4
    MAX_STEPS = 200
    N_EPISODES_PER_LEVEL = 100  # 每个等级测 30 次 (节省时间，正式跑可以改 50-100)
    
    # 3. 🎯 定义要测试的噪声等级列表 (单位: 米)
    # 0mm (基准), 1mm, 2mm (目标), 3mm (极端)
    NOISE_LEVELS = [0.0, 0.001, 0.002, 0.003] 
    
    # 4. 开关：是否对比 Base Policy (也就是 Residual Scale = 0 的情况)
    # 如果想看纯 Base 的抗扰性，设为 True，会跑两遍
    TEST_BASELINE = True
    # ===========================================

    print(f"🧊 Loading Environment...")
    # 注意：这里 Scale 设为 0.01
    env = ResidualPegEnv(
        base_ckpt_path=BASE_CKPT,
        residual_scale=0.01,    # Scale 0.01
        residual_clip=0.2,
        max_steps=MAX_STEPS,
        action_chunk_size=CHUNK_SIZE,
        device="cuda:0"
    )

    print(f"🔥 Loading SAC Model: {RESIDUAL_MODEL_PATH}")
    model = SAC.load(RESIDUAL_MODEL_PATH)

    results = []

    print("\n" + "="*60)
    print("🚀 开始鲁棒性压力测试 (Robustness Stress Test)")
    print("="*60)

    for noise in NOISE_LEVELS:
        print(f"\n[Test Case] Noise Level: ±{noise*1000:.1f} mm")
        
        # --- 测试 Hybrid (SAC + Base) ---
        # 恢复 Scale = 0.01
        env.residual_scale = 0.01 
        sr_hybrid = evaluate_policy_with_noise(env, model, N_EPISODES_PER_LEVEL, noise)
        
        # --- 测试 Baseline (Only Base Policy) ---
        sr_base = 0.0
        if TEST_BASELINE:
            # 把 Scale 设为 0，相当于关掉 SAC，只测 Base Policy + 螺旋搜索
            env.residual_scale = 0.0 
            # 这里的 model.predict 输出什么不重要了，因为 scale 是 0
            sr_base = evaluate_policy_with_noise(env, model, N_EPISODES_PER_LEVEL, noise)
        
        # 记录数据
        results.append({
            "Noise (mm)": noise * 1000,
            "Base Policy (SR)": sr_base,
            "Hybrid (Base+SAC) (SR)": sr_hybrid,
            "Improvement": sr_hybrid - sr_base
        })

    # ================= 打印最终报表 =================
    print("\n\n" + "="*60)
    print("📊 最终鲁棒性测试报告")
    print("="*60)
    
    try:
        df = pd.DataFrame(results)
        # 格式化输出百分比
        df["Base Policy (SR)"] = df["Base Policy (SR)"].apply(lambda x: f"{x:.1%}")
        df["Hybrid (Base+SAC) (SR)"] = df["Hybrid (Base+SAC) (SR)"].apply(lambda x: f"{x:.1%}")
        df["Improvement"] = df["Improvement"].apply(lambda x: f"{x:+.1%}")
        print(df.to_string(index=False))
    except ImportError:
        # 如果没装 pandas，用普通打印
        print(f"{'Noise':<10} | {'Base Policy':<15} | {'Hybrid (SAC)':<15} | {'Improvement'}")
        print("-" * 55)
        for r in results:
            print(f"{r['Noise (mm)']:<10.1f} | {r['Base Policy (SR)']:<15.1%} | {r['Hybrid (Base+SAC) (SR)']:<15.1%} | {r['Improvement']:+.1%}")
    print("="*60)

if __name__ == "__main__":
    main()