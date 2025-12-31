import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import gymnasium as gym
import numpy as np
import panda_mujoco_gym  # ⚠️ 必须保留：触发 env 注册
from gymnasium.wrappers import TimeLimit

from scripts.scripted_policy import ScriptedPickPlacePolicy


# =========================
# ✅ 固定配置（不需要命令行参数）
# =========================
ENV_ID = "FrankaPickAndPlaceSparse-v0"
OUT_DIR = "data/pickplace_scripted_npz"

N_EPISODES = 300
MAX_STEPS = 300
SEED = 0

RENDER = False          # 采集建议 False（快）；debug 可改 True
VERBOSE = True          # 是否打印每条 episode 的保存信息

FILTER_TRIVIAL_SUCCESS = True
TRIVIAL_SUCCESS_STEPS = 2  # success 出现在 <=2 step 视为无效 episode


ObsType = Union[np.ndarray, Dict[str, Any]]


def force_time_limit(env, max_episode_steps: int):
    """Force override TimeLimit even if env is already wrapped."""
    base = env
    while base.__class__.__name__ == "TimeLimit":
        base = base.env
    return TimeLimit(base, max_episode_steps=max_episode_steps)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _is_dict_obs(obs: ObsType) -> bool:
    return isinstance(obs, dict)


def _obs_keys(obs: Dict[str, Any]) -> List[str]:
    return sorted(list(obs.keys()))


@dataclass
class EpisodeBuffer:
    obs_dict: Dict[str, List[np.ndarray]] = None
    obs_list: List[np.ndarray] = None

    next_obs_dict: Dict[str, List[np.ndarray]] = None
    next_obs_list: List[np.ndarray] = None

    actions: List[np.ndarray] = None
    rewards: List[float] = None
    terminated: List[bool] = None
    truncated: List[bool] = None
    is_success: List[bool] = None

    def __post_init__(self):
        self.actions = []
        self.rewards = []
        self.terminated = []
        self.truncated = []
        self.is_success = []


def _init_episode_buffer(first_obs: ObsType) -> EpisodeBuffer:
    buf = EpisodeBuffer()
    if _is_dict_obs(first_obs):
        keys = _obs_keys(first_obs)
        buf.obs_dict = {k: [] for k in keys}
        buf.next_obs_dict = {k: [] for k in keys}
    else:
        buf.obs_list = []
        buf.next_obs_list = []
    return buf


def _append_obs(buf: EpisodeBuffer, obs: ObsType, next_obs: ObsType):
    if buf.obs_dict is not None:
        for k in buf.obs_dict.keys():
            buf.obs_dict[k].append(_to_numpy(obs[k]))
            buf.next_obs_dict[k].append(_to_numpy(next_obs[k]))
    else:
        buf.obs_list.append(_to_numpy(obs))
        buf.next_obs_list.append(_to_numpy(next_obs))


def _stack_episode(buf: EpisodeBuffer) -> Dict[str, np.ndarray]:
    data = {}
    if buf.obs_dict is not None:
        for k, v in buf.obs_dict.items():
            data[f"obs/{k}"] = np.stack(v, axis=0)
        for k, v in buf.next_obs_dict.items():
            data[f"next_obs/{k}"] = np.stack(v, axis=0)
    else:
        data["obs"] = np.stack(buf.obs_list, axis=0)
        data["next_obs"] = np.stack(buf.next_obs_list, axis=0)

    data["action"] = np.stack([_to_numpy(a) for a in buf.actions], axis=0)
    data["reward"] = np.asarray(buf.rewards, dtype=np.float32)
    data["terminated"] = np.asarray(buf.terminated, dtype=np.bool_)
    data["truncated"] = np.asarray(buf.truncated, dtype=np.bool_)
    data["is_success"] = np.asarray(buf.is_success, dtype=np.bool_)
    return data


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _clear_dir(path: str):
    for fname in os.listdir(path):
        if fname.endswith(".npz") or fname == "manifest.json":
            os.remove(os.path.join(path, fname))

def main():
    _ensure_dir(OUT_DIR)
    _clear_dir(OUT_DIR)

    env = gym.make(ENV_ID, render_mode="human" if RENDER else None)
    env = force_time_limit(env, MAX_STEPS)

    rng = np.random.default_rng(SEED)
    policy = ScriptedPickPlacePolicy(verbose=False)  # 采集时通常不需要 policy 内部 verbose

    manifest = {
        "env_id": ENV_ID,
        "n_episodes": int(N_EPISODES),
        "max_steps": int(MAX_STEPS),
        "seed": int(SEED),
        "filter_trivial_success": bool(FILTER_TRIVIAL_SUCCESS),
        "trivial_success_steps": int(TRIVIAL_SUCCESS_STEPS),
        "episodes": [],
    }

    kept = 0
    tried = 0

    while kept < N_EPISODES:
        tried += 1
        ep_seed = int(rng.integers(0, 1_000_000))

        obs, _ = env.reset(seed=ep_seed)
        policy.reset()

        buf = _init_episode_buffer(obs)
        ep_success_any = False
        t_end = -1

        for t in range(MAX_STEPS):
            action = policy.act(obs)
            policy.step_phase_counter()

            next_obs, reward, terminated, truncated, info = env.step(action)

            success = bool(info.get("is_success", False))
            if success:
                ep_success_any = True

            _append_obs(buf, obs, next_obs)
            buf.actions.append(_to_numpy(action))
            buf.rewards.append(float(reward))
            buf.terminated.append(bool(terminated))
            buf.truncated.append(bool(truncated))
            buf.is_success.append(success)

            obs = next_obs

            if RENDER:
                env.render()

            if terminated or truncated:
                t_end = t
                break

        if t_end < 0:
            t_end = MAX_STEPS - 1

        ep_valid = True
        if FILTER_TRIVIAL_SUCCESS and ep_success_any and t_end <= TRIVIAL_SUCCESS_STEPS:
            ep_valid = False

        if not ep_valid:
            if VERBOSE:
                print(
                    f"[SKIP] seed={ep_seed} steps={t_end:03d} "
                    f"success_any={ep_success_any} valid={ep_valid}"
                )
            continue

        ep_data = _stack_episode(buf)
        ep_len = int(ep_data["action"].shape[0])

        fname = f"episode_{kept:06d}.npz"
        fpath = os.path.join(OUT_DIR, fname)

        np.savez_compressed(
            fpath,
            **ep_data,
            episode_seed=np.asarray(ep_seed, dtype=np.int64),
            episode_len=np.asarray(ep_len, dtype=np.int32),
        )

        manifest["episodes"].append(
            {
                "file": fname,
                "episode_seed": ep_seed,
                "episode_len": ep_len,
                "success_any": bool(ep_success_any),
                "valid": bool(ep_valid),
            }
        )

        kept += 1

        if VERBOSE:
            print(
                f"[SAVE] {fname} seed={ep_seed} len={ep_len:03d} "
                f"success_any={ep_success_any} ({kept}/{N_EPISODES}, tried={tried})"
            )

    env.close()

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nSaved demos to: {OUT_DIR}")
    print(f"Manifest: {manifest_path}")
    print(f"Kept episodes: {kept}/{N_EPISODES} (tried={tried})")


if __name__ == "__main__":
    main()
