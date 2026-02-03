import sys
import os
import cv2
import time # 引入 time 库

# 将当前脚本的父目录加入路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gymnasium as gym
import numpy as np
import mujoco # 引入 mujoco 用于创建高清渲染器
import panda_mujoco_gym 
from gymnasium.wrappers import TimeLimit
import custom_envs 
from scripts.scripted_policy import ScriptedPegInHolePolicy

def force_time_limit(env, max_episode_steps: int):
    base = env
    while base.__class__.__name__ == "TimeLimit":
        base = base.env
    return TimeLimit(base, max_episode_steps=max_episode_steps)

def main(
    n_episodes: int = 5,
    max_steps: int = 500,
    render: bool = True,
    seed: int = 0,
    verbose: bool = True,
):
    # 使用 rgb_array 模式
    env = gym.make(
        "FrankaPegInHole-v0", 
        render_mode="rgb_array", 
        disable_env_checker=True
    )
    
    env = force_time_limit(env, max_steps)
    rng = np.random.default_rng(seed)

    success_cnt = 0
    valid_cnt = 0
    policy = ScriptedPegInHolePolicy(verbose=verbose)

    # 🟢 [新增] 创建一个专用的高清渲染器 (640x480) 给人类看
    # 注意: env.unwrapped.model 才能拿到原始 mujoco 模型
    if render:
        human_renderer = mujoco.Renderer(env.unwrapped.model, height=480, width=480)

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
            policy.reset()

            ep_success = False
            ep_valid = True
            success_hold_steps = 0 # 🟢 [新增] 成功后保持步数计数器
            
            print(f"--- Episode {ep} Start ---")

            for t in range(max_steps):
                action = policy.act(obs)
                policy.step_phase_counter()

                obs, _, terminated, truncated, info = env.step(action)

                if info.get("is_success", False):
                    ep_success = True

                # =========================================================
                # 🟢 [修改] 高清渲染逻辑
                # =========================================================
                if render:
                    # 1. 使用高清渲染器更新场景
                    human_renderer.update_scene(env.unwrapped.data, camera="wrist_camera") # 也可以改成 "watching" 看全局
                    img_wrist_hd = human_renderer.render()
                    
                    human_renderer.update_scene(env.unwrapped.data, camera="watching")
                    img_global_hd = human_renderer.render()

                    # 2. 转 BGR
                    img_wrist_hd = cv2.cvtColor(img_wrist_hd, cv2.COLOR_RGB2BGR)
                    img_global_hd = cv2.cvtColor(img_global_hd, cv2.COLOR_RGB2BGR)

                    # 3. 拼接 (因为是 640宽，两个拼起来有点宽，我们把它们缩小一点点或者上下拼)
                    # 这里演示左右拼接，如果屏幕放不下，可以把 width 改小
                    combined = np.hstack([img_global_hd, img_wrist_hd])

                    # 4. 加文字提示状态
                    status_text = f"Step: {t} Phase: {policy.phase}"
                    color = (0, 255, 0) if ep_success else (0, 165, 255)
                    if ep_success: status_text += " [SUCCESS]"
                    
                    cv2.putText(combined, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.imshow("HD Verification", combined)
                    
                    # 🟢 [关键] 只有还没成功时才快速播放
                    # 如果成功了，我们不立即退出，而是让用户按任意键继续，或者慢放
                    key_delay = 1
                    if ep_success:
                        # 成功后，稍微慢一点，让你看清最后几步
                        key_delay = 50 
                    
                    if cv2.waitKey(key_delay) & 0xFF == ord('q'):
                        return

                # =========================================================
                # 🟢 [关键] 成功后不立即 Break，而是“贪恋”几十步
                # =========================================================
                if ep_success:
                    success_hold_steps += 1
                    # 让它再跑 20 步，确保 Push 动作做完，且让你看清插进去的状态
                    if success_hold_steps > 20: 
                        print("✅ Success confirmed. Moving to next episode...")
                        # 这一步暂停住，让你按任意键才进入下一集 (彻底看清)
                        print("Press any key on the window to continue...")
                        if render:
                            cv2.waitKey(0) 
                        break
                
                # 如果没成功，但环境判定结束了 (比如撞墙)，那就退
                elif terminated or truncated:
                    break

            if ep_success and t <= 2:
                ep_valid = False

            if ep_success and ep_valid:
                success_cnt += 1
            if ep_valid:
                valid_cnt += 1

            if verbose:
                print(f"[EP {ep:03d}] steps={t:03d} success={ep_success}")
                
    finally:
        env.close()
        if render:
            cv2.destroyAllWindows()

    denom = max(1, valid_cnt)
    print(f"Success rate: {success_cnt}/{denom} = {success_cnt / denom:.2f}")

if __name__ == "__main__":
    main(
        n_episodes=5,    
        max_steps=500, 
        render=True,
        verbose=True,    
        seed=0
    )