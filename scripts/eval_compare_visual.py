import sys
import os
import time
import cv2
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import mujoco

# =========================
# 🟢 路径终极修正
# =========================
current_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(script_dir)
source_root = os.path.join(project_root, 'diffusion_policy')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if source_root not in sys.path:
    sys.path.insert(0, source_root)

# 引入你的残差环境
from scripts.residual_env import ResidualPegEnv 

def run_visual_eval(mode_name, residual_scale, n_episodes=3):
    # ================= 配置区域 =================
    # 1. Base Policy 路径
    BASE_CKPT = "diffusion_policy/data/outputs/2026.01.26/16.00.38_train_diffusion_unet_hybrid_peg_in_hole/checkpoints/base_policy.ckpt"
    
    # 2. SAC 模型路径 (Hybrid 模式需要)
    SAC_MODEL_PATH = "data/models/SAC_Residual_v1/rl_model_105000_steps.zip" # 替换为你最好的模型路径
    
    CHUNK_SIZE = 4
    MAX_STEPS = 250
    # ===========================================

    print(f"\n{'='*60}")
    print(f"🎬 正在初始化可视化: {mode_name} (Scale={residual_scale})")
    print(f"{'='*60}")

    # 初始化环境 (记得把 render_mode 注释掉)
    env = ResidualPegEnv(
        base_ckpt_path=BASE_CKPT,
        residual_scale=residual_scale, 
        residual_clip=0.2,
        max_steps=MAX_STEPS,
        action_chunk_size=CHUNK_SIZE,
        device="cuda:0"
        # render_mode='rgb_array' # 👈 确保这里已注释/删除
    )

    print(f"🔥 Loading SAC Model from: {SAC_MODEL_PATH}")
    model = SAC.load(SAC_MODEL_PATH)

    # =========================================================
    # 🛠️ [核心修复]：一次性获取底层的 model 和 data
    # =========================================================
    # 1. 穿透 Wrapper 找到真实环境
    if hasattr(env, 'env'):
        # ResidualPegEnv 通常把 gym env 存在 .env 属性里
        real_env = env.env.unwrapped
    else:
        real_env = env.unwrapped
    
    # 2. 提取 MuJoCo 对象 (供渲染器使用)
    # 如果这行报错，说明 real_env 找错了，但通常 env.env.unwrapped 是对的
    mujoco_model = real_env.model
    mujoco_data = real_env.data   # 👈 这就是循环里缺少的 data

    # 3. 设置高清分辨率
    mujoco_model.vis.global_.offwidth = 1920
    mujoco_model.vis.global_.offheight = 1080
    
    # 创建渲染器
    renderer = mujoco.Renderer(mujoco_model, height=720, width=960)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        steps = 0
        
        print(f"  > Episode {ep+1}/{n_episodes} Start...")

        while not (done or truncated):
            # 1. SAC 预测残差动作
            # 关键：即使 Scale=0，我们也跑一遍流程，保证对比的严谨性
            # deterministic=True 关闭 SAC 的探索噪声，只保留 Policy 的确定性输出(和Dithering效果)
            action, _ = model.predict(obs, deterministic=True)
            
            # 2. 环境步进 (内部会叠加 Base + Scale * SAC)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1

            # ================= 🎥 渲染与可视化 =================
            # 更新渲染器场景
            renderer.update_scene(mujoco_data, camera="watching")
            
            img = renderer.render()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # --- 添加 HUD 信息 (平视显示器) ---
            # 1. 策略名称
            color = (0, 255, 255) if residual_scale > 0 else (200, 200, 200)
            cv2.putText(img, f"MODE: {mode_name}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # 2. Scale 参数
            cv2.putText(img, f"Residual Scale: {residual_scale}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 3. 步数
            cv2.putText(img, f"Step: {steps}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 4. 成功提示 (大大的绿色 SUCCESS)
            if info.get('is_success', False):
                cv2.putText(img, "SUCCESS!", (300, 360), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                # 成功后多停顿一会展示
                cv2.imshow("Policy Comparison", img)
                cv2.waitKey(1500)
                done = True # 强制结束

            # 显示图像
            cv2.imshow("Policy Comparison", img)
            
            # 按 'q' 提前退出当前 Episode
            # 按空格暂停
            key = cv2.waitKey(20) # 20ms 延时，约 50fps
            if key & 0xFF == ord('q'):
                done = True
            elif key == 32: # Space bar to pause
                cv2.waitKey(0)

        print(f"  > Episode finished. Steps: {steps}, Success: {info.get('is_success', False)}")
        time.sleep(0.5)

    # 清理资源，准备下一个模式
    env.close()

def main():
    # ==========================================
    # ⚔️ 对决开始：Base vs Hybrid
    # ==========================================
    
    # Round 1: Base Policy (纯螺旋搜索)
    # 观察重点：动作非常平滑，但一旦顶住孔边，就会“定住”推不动 (死锁)
    run_visual_eval(
        mode_name="Base Policy (Only Spiral)", 
        residual_scale=0.0, 
        n_episodes=3
    )

    # Round 2: Hybrid Policy (SAC 介入)
    # 观察重点：动作会有高频抖动(帕金森感)，在孔边卡住时会疯狂抽搐，然后进去了
    run_visual_eval(
        mode_name="Hybrid Policy (Base+SAC)", 
        residual_scale=0.01, 
        n_episodes=3
    )
    
    cv2.destroyAllWindows()
    print("\n✅ 所有演示结束。")

if __name__ == "__main__":
    main()