# scripts/eval_residual_sac.py
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from stable_baselines3 import SAC

from scripts.residual_env import ResidualPickPlaceEnv, ResidualConfig


@dataclass
class EvalCfg:
    # ===== model =====
    # ✅ 用 last 模型评估（支持带/不带 .zip）
    model_path: str = "./data/residual_sac_runs/run_0/best_sac_residual.zip"

    # ===== env / DP checkpoint (match training) =====
    ckpt: str = "./diffusion_policy/data/outputs/2025.12.18/21.05.00_train_diffusion_unet_lowdim_pickplace_lowdim/checkpoints/latest.ckpt"
    device: str = "cuda:0"
    env_id: str = "FrankaPickAndPlaceSparse-v0"
    max_steps: int = 150
    n_obs_steps: int = 2

    # residual params (should match training; gripper_dim None -> infer last dim)
    delta_scale: float = 0.005
    freeze_gripper: bool = True
    gripper_dim: Optional[int] = None

    # DP rollout knobs (match training)
    dp_inference_steps: int = 4
    dp_replan_every: int = 4

    # ✅ eval 口径：warmup/ramp 全关（残差从第 0 步就生效）
    warmup_steps: int = 0
    ramp_steps: int = 0

    # ===== eval =====
    n_eval_episodes: int = 100
    eval_seed: int = 12345
    render: bool = False
    strict_success_end: bool = False   # False: 任意时刻成功都算成功
    print_each_episode: bool = True


CFG = EvalCfg()


def _normalize_model_path(p: str) -> str:
    """
    Make SB3 load robust:
    - accept 'xxx.zip' or 'xxx'
    - if file exists with/without .zip, return a usable path
    """
    p = os.path.expanduser(p)
    if p.endswith(".zip"):
        if os.path.exists(p):
            return p
        # fallback: strip .zip
        return p[:-4]
    else:
        if os.path.exists(p):
            return p
        if os.path.exists(p + ".zip"):
            return p + ".zip"
        return p


def run_eval(cfg: EvalCfg) -> Tuple[float, float, List[float], List[float]]:
    # --- build eval env with warmup/ramp disabled ---
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

    # --- load model (best or last) ---
    model_path = _normalize_model_path(cfg.model_path)
    model = SAC.load(model_path, device=cfg.device)

    success_list: List[float] = []
    steps_list: List[float] = []

    try:
        for ep_i in range(cfg.n_eval_episodes):
            obs, info = env.reset(seed=cfg.eval_seed + ep_i)
            steps = 0
            success_any = False
            success_terminal = False

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, rew, terminated, truncated, info = env.step(action)

                is_succ = bool(info.get("is_success", False))
                if not cfg.strict_success_end:
                    success_any = success_any or is_succ

                steps += 1

                if terminated or truncated:
                    if cfg.strict_success_end:
                        success_terminal = is_succ
                    break

                if steps > 10_000:
                    break

            succ = success_terminal if cfg.strict_success_end else success_any
            success_list.append(float(succ))
            steps_list.append(float(steps))

            if cfg.print_each_episode:
                print(f"[RESIDUAL EVAL] ep={ep_i:03d} seed={cfg.eval_seed+ep_i} succ={int(succ)} steps={steps}")

        success_rate = float(np.mean(success_list)) if success_list else 0.0
        mean_steps = float(np.mean(steps_list)) if steps_list else 0.0
        return success_rate, mean_steps, success_list, steps_list

    finally:
        env.close()


def main():
    sr, ms, succs, lens = run_eval(CFG)
    print("=" * 60)
    print("[RESIDUAL SAC + (warmup=0,ramp=0) RESULT]")
    print(f"model={CFG.model_path}")
    print(f"episodes={CFG.n_eval_episodes} seed0={CFG.eval_seed}")
    print(f"success_rate={sr:.3f}  mean_steps={ms:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
