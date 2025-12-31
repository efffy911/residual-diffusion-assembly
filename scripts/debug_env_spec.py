import gymnasium as gym
import panda_mujoco_gym
import numpy as np

env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")
obs, info = env.reset()

print("action_space:", env.action_space)
print("  shape:", env.action_space.shape)
print("  low:", env.action_space.low)
print("  high:", env.action_space.high)

print("obs type:", type(obs))
if isinstance(obs, dict):
    for k, v in obs.items():
        print(f"obs['{k}'] shape:", np.asarray(v).shape)
else:
    print("obs shape:", np.asarray(obs).shape)

env.close()