# scripts/train_residual.py

import os
import time
from dataclasses import dataclass
from typing import Callable, Any, Optional, Tuple

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from scripts.residual_env import ResidualPickPlaceEnv, ResidualConfig


# ===================== EDIT CONFIG HERE =====================

# ✅ 1) 直接用 residual_env.py 里的 ResidualConfig 作为“环境 cfg”
ENV_CFG = ResidualConfig(
    checkpoint="./diffusion_policy/data/outputs/2025.12.18/21.05.00_train_diffusion_unet_lowdim_pickplace_lowdim/checkpoints/latest.ckpt",
    device="cuda:0",
    env_id="FrankaPickAndPlaceSparse-v0",
    max_steps=150,
    n_obs_steps=2,

    # residual
    delta_scale=0.005,
    freeze_gripper=True,
    gripper_dim=None,   # None -> env 内自动推断为最后一维

    # DP rollout / speed-stability knobs
    dp_inference_steps=4,
    dp_replan_every=4,
    warmup_steps=3_000,
    ramp_steps=2_000,
    seed=0,
    train_seed_pool=[12345, 54321, 11111, 22222, 33333, 44444, 55555, 66666],
)


# ✅ 2) 训练层 cfg：只管 SAC / eval / 保存
@dataclass
class TrainCfg:
    # --- training ---
    timesteps: int = 50_000
    save_dir: str = "./data/residual_sac_runs/run_0"
    render: bool = False  # training should be False

    # --- eval ---
    eval_every_steps: int = 5_000
    n_eval_episodes: int = 50
    eval_seed: int = 12345
    strict_success_end: bool = False
    save_with_step: bool = True

    # --- SB3 SAC hyperparams ---
    device: str = "cuda:0"  # SB3 model device（env 内 DP 的 device 由 ENV_CFG.device 控制）
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    learning_starts: int = 10_000
    ent_coef: str = "auto"
    verbose: int = 2
    seed: int = 0
    train_seed_pool: Optional[list[int]] = None


CFG = TrainCfg()

# ============================================================


def make_env(res_cfg: ResidualConfig, render: bool = False):
    def _init():
        env = ResidualPickPlaceEnv(res_cfg, render=render)
        env = Monitor(env)
        return env
    return _init


class StepSpeedCallback(BaseCallback):
    def __init__(self, print_every=2000):
        super().__init__()
        self.print_every = int(print_every)
        self.t0 = None
        self.last_t = None
        self.last_steps = 0

    def _on_training_start(self):
        self.t0 = time.time()
        self.last_t = self.t0
        self.last_steps = 0

    def _on_step(self) -> bool:
        steps = self.num_timesteps
        if steps % self.print_every == 0:
            now = time.time()
            dt = now - self.last_t
            ds = steps - self.last_steps
            sps = ds / max(dt, 1e-6)
            total = now - self.t0
            print(f"[PROGRESS] steps={steps}  sps={sps:.1f}  elapsed={total:.1f}s")
            self.last_t = now
            self.last_steps = steps
        return True


class ResidualEvalCallback(BaseCallback):
    """
    Deterministic evaluation with FIXED seeds to reduce variance.

    Saves best model by:
      1) higher success_rate
      2) if tie, lower mean_steps
    """
    def __init__(
        self,
        eval_env_fn: Callable[[], Any],
        eval_every_steps: int,
        n_eval_episodes: int,
        save_dir: str,
        best_name: str = "best_sac_residual",
        eval_seed: int = 12345,
        eval_seed_list: Optional[list[int]] = None,
        strict_success_end: bool = False,
        save_with_step: bool = False,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_every_steps = int(eval_every_steps)
        self.n_eval_episodes = int(n_eval_episodes)
        self.save_dir = save_dir
        self.best_name = best_name

        self.eval_seed = int(eval_seed)
        self.eval_seed_list = eval_seed_list  # e.g., [12345, 54321]
        self.strict_success_end = bool(strict_success_end)
        self.save_with_step = bool(save_with_step)

        self.best_success = -1.0
        self.best_mean_steps = 1e9

        os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def _unwrap_info(info):
        # DummyVecEnv returns list[dict]
        if isinstance(info, (list, tuple)) and len(info) > 0 and isinstance(info[0], dict):
            return info[0]
        return info if isinstance(info, dict) else {}

    def _reset_env(self, env, seed: Optional[int] = None):
        try:
            out = env.reset(seed=seed) if seed is not None else env.reset()
        except TypeError:
            out = env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            return out[0]
        return out

    def _step_env(self, env, action):
        out = env.step(action)
        if len(out) == 4:
            obs, rew, done, info = out
            terminated, truncated = bool(done), False
        else:
            obs, rew, terminated, truncated, info = out
        return obs, rew, bool(terminated), bool(truncated), info

    def _run_eval(self) -> Tuple[float, float]:
        """
        Eval over multiple seed0 blocks to reduce overfitting to a single seed range.

        If self.eval_seed_list is provided (e.g. [12345, 54321]),
        we will run n_eval_episodes episodes for EACH seed0 in the list.
        Total episodes = n_eval_episodes * len(eval_seed_list).

        Returns:
        success_rate: mean over all episodes
        mean_steps:   mean over all episodes
        """
        seed0_list = self.eval_seed_list if self.eval_seed_list else [self.eval_seed]

        success_list = []
        steps_list = []

        # ✅ 每个 seed0 用一个独立 env（避免 env 内部状态/total_steps 干扰）
        for seed0 in seed0_list:
            env = self.eval_env_fn()
            try:
                for ep_i in range(self.n_eval_episodes):
                    obs = self._reset_env(env, seed=int(seed0) + ep_i)

                    steps = 0
                    success_any = False
                    success_terminal = False

                    while True:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, rew, terminated, truncated, info = self._step_env(env, action)

                        info0 = self._unwrap_info(info)
                        is_succ = bool(info0.get("is_success", False))

                        if not self.strict_success_end:
                            success_any = success_any or is_succ

                        steps += 1

                        if terminated or truncated:
                            if self.strict_success_end:
                                success_terminal = is_succ
                            break

                        if steps > 10_000:
                            break

                    succ = success_terminal if self.strict_success_end else success_any
                    success_list.append(float(succ))
                    steps_list.append(float(steps))
            finally:
                try:
                    env.close()
                except Exception:
                    pass

        success_rate = float(np.mean(success_list)) if success_list else 0.0
        mean_steps = float(np.mean(steps_list)) if steps_list else 0.0
        return success_rate, mean_steps


    def _maybe_save_best(self, success_rate: float, mean_steps: float, steps: int):
        improved = False
        if success_rate > self.best_success + 1e-9:
            improved = True
        elif abs(success_rate - self.best_success) < 1e-9 and mean_steps < self.best_mean_steps:
            improved = True

        if not improved:
            return

        self.best_success = success_rate
        self.best_mean_steps = mean_steps

        path = os.path.join(self.save_dir, self.best_name)
        self.model.save(path)
        print(f"[EVAL][BEST] success={success_rate:.3f} mean_steps={mean_steps:.1f} -> saved: {path}.zip")

        if self.save_with_step:
            snap = os.path.join(
                self.save_dir,
                f"{self.best_name}_step{steps}_succ{success_rate:.3f}_len{mean_steps:.1f}"
            )
            self.model.save(snap)
            print(f"[EVAL][BEST][SNAP] saved: {snap}.zip")

    def _on_step(self) -> bool:
        if self.eval_every_steps > 0 and (self.num_timesteps % self.eval_every_steps == 0):
            success_rate, mean_steps = self._run_eval()
            print(f"[EVAL] steps={self.num_timesteps} success={success_rate:.3f} mean_steps={mean_steps:.1f}")
            self._maybe_save_best(success_rate, mean_steps, steps=self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        success_rate, mean_steps = self._run_eval()
        print(f"[EVAL][FINAL] steps={self.num_timesteps} success={success_rate:.3f} mean_steps={mean_steps:.1f}")
        self._maybe_save_best(success_rate, mean_steps, steps=self.num_timesteps)


def main():
    os.makedirs(CFG.save_dir, exist_ok=True)

    # -------- train env --------
    env = DummyVecEnv([make_env(ENV_CFG, render=CFG.render)])

    # -------- eval env factory (NEW env each time) --------
    def make_eval_env():
        # eval 用“残差全开”口径：warmup=0, ramp=0
        res_cfg_eval = ResidualConfig(
            checkpoint=ENV_CFG.checkpoint,
            device=ENV_CFG.device,
            env_id=ENV_CFG.env_id,
            max_steps=ENV_CFG.max_steps,
            n_obs_steps=ENV_CFG.n_obs_steps,

            delta_scale=ENV_CFG.delta_scale,
            freeze_gripper=ENV_CFG.freeze_gripper,
            gripper_dim=ENV_CFG.gripper_dim,

            dp_inference_steps=ENV_CFG.dp_inference_steps,
            dp_replan_every=ENV_CFG.dp_replan_every,

            warmup_steps=0,
            ramp_steps=0,

            # 如果你 residual_env.py 里加了 seed/train_seed_pool，这里也建议对齐
            seed=getattr(ENV_CFG, "seed", 0),
            train_seed_pool=None,  # eval 时我们显式传 seed，所以不需要 pool
        )
        return ResidualPickPlaceEnv(res_cfg_eval, render=False)

    # -------- model --------
    model = SAC(
        "MlpPolicy",
        env,
        device=CFG.device,
        learning_rate=CFG.learning_rate,
        buffer_size=CFG.buffer_size,
        batch_size=CFG.batch_size,
        tau=CFG.tau,
        gamma=CFG.gamma,
        train_freq=CFG.train_freq,
        gradient_steps=CFG.gradient_steps,
        learning_starts=CFG.learning_starts,
        ent_coef=CFG.ent_coef,
        verbose=CFG.verbose,
    )

    # -------- callbacks --------
    speed_cb = StepSpeedCallback(print_every=2000)
    eval_cb = ResidualEvalCallback(
        eval_env_fn=make_eval_env,
        eval_every_steps=CFG.eval_every_steps,
        n_eval_episodes=CFG.n_eval_episodes,
        save_dir=CFG.save_dir,
        best_name="best_sac_residual",
        eval_seed=CFG.eval_seed,
        eval_seed_list=[13579, 86420],  # ✅ 新增：多 seed0
        strict_success_end=CFG.strict_success_end,
        save_with_step=CFG.save_with_step,
    )

    cb = CallbackList([speed_cb, eval_cb])

    t0 = time.time()
    model.learn(total_timesteps=CFG.timesteps, callback=cb)
    print("[SPEED] train_wall_time_sec =", time.time() - t0)

    out_path = os.path.join(CFG.save_dir, "sac_residual_last")
    model.save(out_path)
    print(f"[OK] Saved LAST model to: {out_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
