import os, time
import numpy as np
import gymnasium as gym
import panda_mujoco_gym  # noqa: F401

from scripts.wrappers import BoolDoneWrapper

ENV_ID = "FrankaPickAndPlaceSparse-v0"

def flatten_obs(obs_dict):
    # 你也可以只用 obs_dict["observation"]，但先全都存下来最稳
    o  = obs_dict["observation"].astype(np.float32)
    ag = obs_dict["achieved_goal"].astype(np.float32)
    dg = obs_dict["desired_goal"].astype(np.float32)
    return o, ag, dg

def main(
    out_dir="data/pickplace_npz",
    n_episodes=50,
    ep_len=200,
    seed=0,
    render=False,
):
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make(ENV_ID, render_mode="human" if render else None)
    env = BoolDoneWrapper(env)

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(seed + ep))
        o0, ag0, dg0 = flatten_obs(obs)

        obs_buf = np.zeros((ep_len, 19), dtype=np.float32)
        ag_buf  = np.zeros((ep_len, 3), dtype=np.float32)
        dg_buf  = np.zeros((ep_len, 3), dtype=np.float32)
        act_buf = np.zeros((ep_len, 4), dtype=np.float32)
        rew_buf = np.zeros((ep_len,), dtype=np.float32)
        done_buf= np.zeros((ep_len,), dtype=np.bool_)

        success_ep = 0.0

        for t in range(ep_len):
            # 先用 random policy 把管线跑通；后面再换 scripted / expert
            act = env.action_space.sample()

            obs_buf[t] = o0
            ag_buf[t]  = ag0
            dg_buf[t]  = dg0
            act_buf[t] = act.astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(act)

            rew_buf[t] = float(reward)
            done_buf[t]= bool(terminated or truncated)

            if "is_success" in info:
                success_ep = max(success_ep, float(info["is_success"]))

            o0, ag0, dg0 = flatten_obs(obs)

            if render:
                time.sleep(0.02)

            if terminated or truncated:
                break

        # 截断到真实长度
        T = t + 1
        path = os.path.join(out_dir, f"ep_{ep:05d}.npz")
        np.savez_compressed(
            path,
            obs=obs_buf[:T],
            achieved_goal=ag_buf[:T],
            desired_goal=dg_buf[:T],
            act=act_buf[:T],
            rew=rew_buf[:T],
            done=done_buf[:T],
            success=np.array(success_ep, dtype=np.float32),
        )
        print(f"[{ep+1}/{n_episodes}] saved {path} (T={T}, success={success_ep})")

    env.close()

if __name__ == "__main__":
    main()
