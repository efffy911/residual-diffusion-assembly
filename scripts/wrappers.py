import numpy as np
import gymnasium as gym

class BoolDoneWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, bool(terminated), bool(truncated), info

class NonTrivialResetWrapper(gym.Wrapper):
    """
    Resample reset until ||achieved_goal - desired_goal|| >= min_goal_dist
    so we don't get trivial 1-step success.
    """
    def __init__(self, env, min_goal_dist=0.10, max_tries=50):
        super().__init__(env)
        self.min_goal_dist = float(min_goal_dist)
        self.max_tries = int(max_tries)

    def reset(self, **kwargs):
        last = None
        for _ in range(self.max_tries):
            obs, info = self.env.reset(**kwargs)
            ag = obs["achieved_goal"]
            dg = obs["desired_goal"]
            d = float(np.linalg.norm(ag - dg))
            last = (obs, info, d)
            if d >= self.min_goal_dist:
                info = dict(info)
                info["init_ag_dg_dist"] = d
                return obs, info
        # fallback
        obs, info, d = last
        info = dict(info)
        info["init_ag_dg_dist"] = d
        info["nontrivial_reset_failed"] = True
        return obs, info