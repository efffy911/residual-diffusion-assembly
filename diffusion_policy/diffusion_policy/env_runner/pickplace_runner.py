from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

# ---- import BaseLowdimRunner (repo versions may differ) ----
BaseLowdimRunner = None
_import_errors = []
for _p in [
    "diffusion_policy.env_runner.base_runner",          # common
    "diffusion_policy.env_runner.base_lowdim_runner",   # alt
    "diffusion_policy.env_runner.runner_base",          # alt
]:
    try:
        mod = __import__(_p, fromlist=["BaseLowdimRunner"])
        BaseLowdimRunner = getattr(mod, "BaseLowdimRunner")
        break
    except Exception as e:
        _import_errors.append((_p, repr(e)))

if BaseLowdimRunner is None:
    raise ImportError(
        "Cannot import BaseLowdimRunner. Tried:\n"
        + "\n".join([f"  - {_p}: {err}" for _p, err in _import_errors])
        + "\nPlease locate where BaseLowdimRunner is defined in your diffusion_policy repo "
          "and update the import paths list accordingly."
    )

# env 注册：允许通过 PYTHONPATH 找到你项目里的包
try:
    import panda_mujoco_gym  # noqa: F401
except ImportError:
    panda_mujoco_gym = None


@dataclass
class PickPlaceRunner(BaseLowdimRunner):
    env_id: str = "FrankaPickAndPlaceSparse-v0"
    n_eval_episodes: int = 50
    max_steps: int = 300
    render: bool = False

    n_obs_steps: int = 2
    horizon: int = 16
    seed: int = 0

    output_dir: Optional[str] = None

    # workspace may pass extra args; ignore them safely
    def __init__(self, **kwargs):
        # output_dir is required by BaseLowdimRunner
        output_dir = kwargs.get("output_dir", None)

        # some BaseLowdimRunner expects output_dir positional, some expects keyword
        try:
            super().__init__(output_dir=output_dir)
        except TypeError:
            super().__init__(output_dir)

        # set dataclass defaults
        for k, v in PickPlaceRunner.__dataclass_fields__.items():  # type: ignore
            setattr(self, k, v.default)

        # apply known kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    # ---------- TimeLimit utilities ----------
    @staticmethod
    def _force_time_limit(env, max_episode_steps: int):
        """
        Remove any existing outer TimeLimit wrappers and re-apply one with max_episode_steps.
        This ensures env truncation happens at `max_episode_steps`, not the env's default (often 50).
        """
        base = env
        # unwrap ALL outer TimeLimit wrappers
        while base.__class__.__name__ == "TimeLimit":
            base = base.env
        # re-wrap with desired limit
        return TimeLimit(base, max_episode_steps=max_episode_steps)

    @staticmethod
    def _get_time_limit_steps(env) -> Optional[int]:
        """
        Inspect wrapper chain to find an active TimeLimit and return its max_episode_steps if present.
        """
        cur = env
        visited = 0
        while cur is not None and visited < 50:
            if cur.__class__.__name__ == "TimeLimit":
                return getattr(cur, "_max_episode_steps", None)
            cur = getattr(cur, "env", None)
            visited += 1
        return None

    def _make_env(self):
        env = gym.make(self.env_id, render_mode="human" if self.render else None)

        # Force TimeLimit to runner.max_steps
        env = self._force_time_limit(env, self.max_steps)

        # Print a one-liner sanity check (shows actual wrapper limit)
        tl = self._get_time_limit_steps(env)
        spec_steps = getattr(getattr(env, "spec", None), "max_episode_steps", None)
        print(f"[PickPlaceRunner] TimeLimit={tl}  env.spec.max_episode_steps={spec_steps}  (expected={self.max_steps})")

        return env

    # ---------- obs/action helpers ----------
    @staticmethod
    def _flat_obs(o) -> np.ndarray:
        if isinstance(o, dict):
            parts = []
            for k in ["observation", "achieved_goal", "desired_goal"]:
                if k in o:
                    parts.append(np.asarray(o[k], dtype=np.float32).reshape(-1))
            if not parts:
                raise ValueError(f"Empty obs dict keys={list(o.keys())}")
            return np.concatenate(parts, axis=0)
        return np.asarray(o, dtype=np.float32).reshape(-1)

    @staticmethod
    def _to_numpy(x):
        # torch -> numpy (important: move to CPU)
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().to("cpu").numpy()
        except Exception:
            pass
        return np.asarray(x)

    @staticmethod
    def _take_first_action(action_out) -> np.ndarray:
        a = PickPlaceRunner._to_numpy(action_out)

        # handle shapes
        if a.ndim == 3:   # (B, H, A)
            a = a[0, 0]
        elif a.ndim == 2: # (B, A)
            a = a[0]
        elif a.ndim == 1: # (A,)
            pass
        else:
            raise ValueError(f"Unexpected action shape: {a.shape}")

        return a.astype(np.float32)

    def _policy_act(self, policy: Any, obs_hist: np.ndarray) -> np.ndarray:
        obs_in = obs_hist.astype(np.float32)

        if hasattr(policy, "predict_action"):
            out = policy.predict_action({"obs": obs_in[None, ...]})
            if isinstance(out, dict) and "action" in out:
                return self._take_first_action(out["action"])

        # fallback: callable
        out = policy({"obs": obs_in[None, ...]})
        if isinstance(out, dict) and "action" in out:
            return self._take_first_action(out["action"])

        raise RuntimeError(
            "Policy interface not recognized (need predict_action or callable returning dict['action'])."
        )

    # BaseLowdimRunner usually expects `run(policy)` that returns dict of metrics
    def run(self, policy: Any) -> Dict[str, Any]:
        env = self._make_env()
        rng = np.random.default_rng(self.seed)

        success_list = []
        ep_len_list = []

        for _ in range(self.n_eval_episodes):
            obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
            obs_vec = self._flat_obs(obs)

            obs_hist = np.repeat(obs_vec[None, :], self.n_obs_steps, axis=0)

            success_any = False
            t_final = 0

            for t in range(self.max_steps):
                t_final = t + 1
                act = self._policy_act(policy, obs_hist)

                obs, reward, terminated, truncated, info = env.step(act.astype(np.float32))
                if "is_success" in info:
                    success_any = success_any or bool(info["is_success"])

                obs_vec = self._flat_obs(obs)
                obs_hist = np.concatenate([obs_hist[1:], obs_vec[None, :]], axis=0)

                if self.render:
                    import time
                    time.sleep(1/60)

                if terminated or truncated:
                    break

            success_list.append(float(success_any))
            ep_len_list.append(float(t_final))

        env.close()

        success_rate = float(np.mean(success_list)) if success_list else 0.0
        mean_len = float(np.mean(ep_len_list)) if ep_len_list else 0.0

        return {
            "test_success_rate": success_rate,
            "test_mean_ep_len": mean_len,
            "test_mean_score": success_rate,
        }
