import gymnasium as gym
import numpy as np
import panda_mujoco_gym 
from gymnasium.wrappers import TimeLimit

from scripts.scripted_policy import ScriptedPickPlacePolicy


def force_time_limit(env, max_episode_steps: int):
    """Force override TimeLimit even if env is already wrapped."""
    base = env
    while base.__class__.__name__ == "TimeLimit":
        base = base.env
    return TimeLimit(base, max_episode_steps=max_episode_steps)


def main(
    n_episodes: int = 50,
    max_steps: int = 300,
    render: bool = False,
    seed: int = 0,
    verbose: bool = False,
):
    # render_mode 必须在 make 时指定
    env = gym.make(
        "FrankaPickAndPlaceSparse-v0",
        render_mode="human",
    )
    env = force_time_limit(env, max_steps)

    rng = np.random.default_rng(seed)

    success_cnt = 0
    valid_cnt = 0

    policy = ScriptedPickPlacePolicy(verbose=verbose)

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
            policy.reset()

            ep_success = False
            ep_valid = True

            for t in range(max_steps):
                action = policy.act(obs)
                policy.step_phase_counter()

                obs, _, terminated, truncated, info = env.step(action)

                if info.get("is_success", False):
                    ep_success = True

                if render:
                    env.render()

                if terminated or truncated:
                    break

            # 过滤「1~2 step 假成功」
            if ep_success and t <= 2:
                ep_valid = False

            if ep_success and ep_valid:
                success_cnt += 1
            if ep_valid:
                valid_cnt += 1

            if verbose:
                print(
                    f"[EP {ep:03d}] steps={t:03d} "
                    f"success={ep_success} valid={ep_valid}"
                )
    finally:
        env.close()

    denom = max(1, valid_cnt)
    print(f"Valid episodes: {valid_cnt}/{n_episodes}")
    print(
        f"Success rate (valid only): "
        f"{success_cnt}/{denom} = {success_cnt / denom:.2f}"
    )


if __name__ == "__main__":
    main(n_episodes=50, render=True, verbose=True)
