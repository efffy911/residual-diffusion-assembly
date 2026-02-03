"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
# import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # ===================== FAST OVERRIDES =====================
    # 1) diffusion 推理步数：这是最主要的慢点（强烈建议先 8）
    if hasattr(cfg, "policy") and hasattr(cfg.policy, "num_inference_steps"):
        cfg.policy.num_inference_steps = 8   # 想更快可改 4；想更准可改 16/32
    # 2) eval 回合数与最大步数：你已经改过 config，但这里强制保险
    if hasattr(cfg, "task") and hasattr(cfg.task, "env_runner"):
        if hasattr(cfg.task.env_runner, "n_eval_episodes"):
            cfg.task.env_runner.n_eval_episodes = 10
        if hasattr(cfg.task.env_runner, "max_steps"):
            cfg.task.env_runner.max_steps = 100
        if hasattr(cfg.task.env_runner, "render"):
            cfg.task.env_runner.render = False  # True 会更慢（viewer 同步）
    print("[FAST EVAL] num_inference_steps =", getattr(cfg.policy, "num_inference_steps", None))
    print("[FAST EVAL] n_eval_episodes =", getattr(cfg.task.env_runner, "n_eval_episodes", None),
          "max_steps =", getattr(cfg.task.env_runner, "max_steps", None),
          "render =", getattr(cfg.task.env_runner, "render", None))
    # ==========================================================

    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    # for key, value in runner_log.items():
    #     if isinstance(value, wandb.sdk.data_types.video.Video):
    #         json_log[key] = value._path
    #     else:
    #         json_log[key] = value
    # no wandb dependency
    for key, value in runner_log.items():
        # handle wandb Video if it exists (duck-typing)
        if hasattr(value, "_path"):
            json_log[key] = getattr(value, "_path")
        else:
            json_log[key] = value

    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
