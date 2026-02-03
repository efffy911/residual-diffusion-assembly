import sys
import os
import time
import numpy as np
import cv2
import gymnasium as gym
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import custom_envs 
from scripts.scripted_policy import ScriptedPegInHolePolicy

# ================= 配置区域 =================
DATA_SAVE_DIR = "data/demo_npz"    # 数据保存路径
NUM_EPISODES = 300                 # 采集多少条成功的轨迹
MAX_STEPS = 400                    # 每回合最大步数
IMG_SIZE = 96                      # (注意：CustomEnv 已经输出了 96x96，这里主要用于检查)
RENDER_MODE = "rgb_array"          
# ===========================================

def collect_demonstrations():
    # 1. 准备目录
    if not os.path.exists(DATA_SAVE_DIR):
        os.makedirs(DATA_SAVE_DIR)
    
    # 2. 初始化环境
    env = gym.make("FrankaPegInHole-v0", render_mode=RENDER_MODE)
    policy = ScriptedPegInHolePolicy(verbose=False)

    # 3. 数据Buffer
    # 🟢 [修改] 准备两个列表存两个相机的图
    all_imgs_global = []  # 存全局 watching 相机
    all_imgs_wrist = []   # 存手眼 wrist_camera 相机
    
    all_states = []
    all_actions = []
    episode_ends = []
    
    collected_cnt = 0
    seed_counter = 0 

    pbar = tqdm(total=NUM_EPISODES, desc="Collecting")

    while collected_cnt < NUM_EPISODES:
        # A. Reset 环境
        seed = seed_counter
        seed_counter += 1
        
        # Reset 返回的 obs 里已经包含了第一帧图像
        obs, _ = env.reset(seed=seed)
        policy.reset()
        
        ep_imgs_global = []
        ep_imgs_wrist = []
        ep_states = []
        ep_actions = []
        
        done = False
        is_success = False
        
        # B. 执行策略循环
        for t in range(MAX_STEPS):
            # 🟢 [修改] 直接从 obs 获取图像 (CustomEnv 已经渲染好了)
            # obs['image'] 是 (3, 96, 96) 的 float32 [0,1]
            img_g = obs['image']       
            img_w = obs['image_wrist'] 

            # 🟢 [优化] 转回 uint8 [0,255] 以节省空间
            # 注意: 这里的形状是 (C, H, W)，符合 Diffusion Policy 的习惯
            img_g = (img_g * 255).clip(0, 255).astype(np.uint8)
            img_w = (img_w * 255).clip(0, 255).astype(np.uint8)

            # 获取状态 (State)
            # 假设 obs 依然保留了底层 observation (qpos, qvel)
            # 如果 custom_envs 里没有 key 'observation'，可能需要用 agent_pos 或其他
            # 这里先假设你的 env 继承自 FrankaEnv，会有 observation
            if 'observation' in obs:
                state = obs['observation']
            else:
                # 备选: 如果没有 observation，就存末端位置
                state = obs['agent_pos'] 

            # 获取动作
            action = policy.act(obs)
            policy.step_phase_counter()

            # 执行一步
            next_obs, reward, terminated, truncated, info = env.step(action)

            # 存入临时 Buffer
            ep_imgs_global.append(img_g)
            ep_imgs_wrist.append(img_w) # 🟢 存手眼图
            ep_states.append(state)
            ep_actions.append(action)

            # 更新 Obs
            obs = next_obs

            # 检查结束
            if info.get("is_success", False):
                is_success = True
            
            if terminated or truncated:
                break
        
        # C. 数据过滤与保存
        if is_success:
            # 存入总 Buffer
            all_imgs_global.extend(ep_imgs_global)
            all_imgs_wrist.extend(ep_imgs_wrist) # 🟢
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)
            
            # 记录 Cumulative Index
            current_len = len(all_imgs_global)
            episode_ends.append(current_len)
            
            collected_cnt += 1
            pbar.update(1)
            pbar.set_postfix({"seed": seed, "steps": len(ep_actions)})
        else:
            # print(f"⚠️ Seed {seed} failed. Discarding.")
            pass

    pbar.close()
    env.close()

    # 4. 转换为 Numpy 数组
    print("Converting to Numpy arrays...")
    # 注意形状: (N, C, H, W) -> 这是 DP 喜欢的格式
    np_imgs_global = np.array(all_imgs_global, dtype=np.uint8) 
    np_imgs_wrist = np.array(all_imgs_wrist, dtype=np.uint8)   # 🟢
    
    np_states = np.array(all_states, dtype=np.float32) 
    np_actions = np.array(all_actions, dtype=np.float32) 
    np_episode_ends = np.array(episode_ends, dtype=np.int32) 

    # 5. 保存为 .npz
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_name = f"peg_in_hole_demo_dual_cam_{NUM_EPISODES}eps_{timestamp}.npz"
    save_path = os.path.join(DATA_SAVE_DIR, file_name)
    
    print(f"Saving to {save_path} ...")
    np.savez_compressed(
        save_path,
        image=np_imgs_global,         # 全局图 key
        image_wrist=np_imgs_wrist,    # 🟢 手眼图 key
        state=np_states,
        action=np_actions,
        episode_ends=np_episode_ends
    )
    print("✅ Data collection complete!")
    print(f"Total Steps: {len(np_imgs_global)}")
    print(f"Global Img Shape: {np_imgs_global.shape}")
    print(f"Wrist  Img Shape: {np_imgs_wrist.shape}") # 🟢

if __name__ == "__main__":
    collect_demonstrations()