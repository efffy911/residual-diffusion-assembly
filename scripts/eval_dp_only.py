# scripts/eval_dp_only.py
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from scripts.residual_env import ResidualPickPlaceEnv, ResidualConfig


@dataclass
class EvalCfg:
    # ===== same as your ENV_CFG =====
    ckpt: str = "./diffusion_policy/data/outputs/2025.12.18/21.05.00_train_diffusion_unet_lowdim_pickplace_lowdim/checkpoints/latest.ckpt"
    device: str = "cuda:0"
    env_id: str = "FrankaPickAndPlaceSparse-v0"
    max_steps: int = 150
    n_obs_steps: int = 2

    # DP rollout knobs (match training)
    dp_inference_steps: int = 4
    dp_replan_every: int = 4

    # Important: for DP-only eval, warmup/ramp don't matter (delta=0),
    # but keep them explicit so it's clear.
    warmup_steps: int = 0
    ramp_steps: int = 0

    # residual params (won't be used because delta=0)
    delta_scale: float = 0.02
    freeze_gripper: bool = True
    gripper_dim: Optional[int] = None

    # ===== eval =====
    n_eval_episodes: int = 100
    eval_seed: int = 54321
    render: bool = False
    strict_success_end: bool = False   # False: success if ever succeeded during episode
    print_each_episode: bool = True    # set False if you want clean output


CFG = EvalCfg()


def run_dp_only_eval(cfg: EvalCfg) -> Tuple[float, float, List[float], List[float]]:
    res_cfg = ResidualConfig(
        checkpoint=cfg.ckpt,
        device=cfg.device,
        env_id=cfg.env_id,
        max_steps=cfg.max_steps,
        n_obs_steps=cfg.n_obs_steps,
        delta_scale=cfg.delta_scale,
        freeze_gripper=cfg.freeze_gripper,
        gripper_dim=cfg.gripper_dim,
        dp_inference_steps=cfg.dp_inference_steps,
        dp_replan_every=cfg.dp_replan_every,
        warmup_steps=cfg.warmup_steps,
        ramp_steps=cfg.ramp_steps,
    )

    env = ResidualPickPlaceEnv(res_cfg, render=cfg.render)

    success_list: List[float] = []
    steps_list: List[float] = []

    try:
        for ep_i in range(cfg.n_eval_episodes):
            obs, info = env.reset(seed=cfg.eval_seed + ep_i)

            steps = 0
            success_any = False
            success_terminal = False

            while True:
                # DP-only: delta = 0
                delta = np.zeros(env.action_space.shape, dtype=np.float32)

                obs, rew, terminated, truncated, info = env.step(delta)

                is_succ = bool(info.get("is_success", False))
                if not cfg.strict_success_end:
                    success_any = success_any or is_succ

                steps += 1

                if terminated or truncated:
                    if cfg.strict_success_end:
                        success_terminal = is_succ
                    break

                if steps > 10_000:
                    # safety break
                    break

            succ = success_terminal if cfg.strict_success_end else success_any
            success_list.append(float(succ))
            steps_list.append(float(steps))

            if cfg.print_each_episode:
                print(f"[DP EVAL] ep={ep_i:03d} seed={cfg.eval_seed+ep_i} succ={int(succ)} steps={steps}")

        success_rate = float(np.mean(success_list)) if success_list else 0.0
        mean_steps = float(np.mean(steps_list)) if steps_list else 0.0
        return success_rate, mean_steps, success_list, steps_list

    finally:
        env.close()


def main():
    sr, ms, succs, lens = run_dp_only_eval(CFG)
    print("=" * 60)
    print(f"[DP ONLY RESULT] episodes={CFG.n_eval_episodes} seed0={CFG.eval_seed}")
    print(f"success_rate={sr:.3f}  mean_steps={ms:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
