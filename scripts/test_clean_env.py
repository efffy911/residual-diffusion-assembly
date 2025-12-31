import time
import gymnasium as gym
import panda_mujoco_gym  # 触发 env 注册

env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")
obs, info = env.reset()

for _ in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
    time.sleep(0.02)

env.close()
