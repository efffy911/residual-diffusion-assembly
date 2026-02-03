import sys
import os
import time
import numpy as np
import torch
import collections
from omegaconf import OmegaConf
import gymnasium as gym
from tqdm import tqdm

# ==============================================================================
# 🟢 [路径设置] 
# ==============================================================================
current_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(script_dir)
source_root = os.path.join(project_root, 'diffusion_policy')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if source_root not in sys.path:
    sys.path.insert(0, source_root)
# ==============================================================================

import hydra
import custom_envs

def get_checkpoint_by_epoch(output_dir, target_epoch=None):
    """
    智能查找 Checkpoint，兼容 epoch=100 和 epoch-110 两种格式
    """
    search_dirs = [os.path.join(output_dir, 'checkpoints'), output_dir]
    ckpts = []
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for f in files:
                    if f.endswith('.ckpt'):
                        ckpts.append(os.path.join(root, f))
    
    if not ckpts:
        raise FileNotFoundError(f"在 {output_dir} 里没找到 .ckpt 文件")

    def parse_epoch(path):
        name = os.path.basename(path)
        if 'latest' in name: return 999999
        
        # 兼容 epoch=xxx
        if 'epoch=' in name:
            try:
                part = name.split('epoch=')[1]
                num_str = part.split('-')[0].split('.')[0]
                return int(num_str)
            except:
                pass
        
        # 兼容 epoch-xxx (新的格式)
        if 'epoch-' in name:
            try:
                part = name.split('epoch-')[1]
                num_str = part.split('-')[0].split('.')[0]
                return int(num_str)
            except:
                pass
        return -1

    if target_epoch is not None:
        found = [p for p in ckpts if parse_epoch(p) == target_epoch]
        if len(found) == 0:
            print(f"⚠️ 警告: 没找到 Epoch {target_epoch} 的权重！")
            print("   文件夹里的文件解析结果:")
            for p in ckpts:
                print(f"   - {os.path.basename(p)} -> 解析为 Epoch: {parse_epoch(p)}")
            raise FileNotFoundError(f"找不到 Epoch {target_epoch}")
        found.sort(key=os.path.getmtime, reverse=True)
        return found[0]

    ckpts.sort(key=parse_epoch, reverse=True)
    return ckpts[0]

def main():
    # ==========================================================================
    # 🟢 [手动配置区域]
    # ==========================================================================
    TARGET_EPOCH = 100   # <--- 修改这里指定你要测的 Epoch
    
    N_EPISODES = 100
    MAX_STEPS = 200      # 🔴 已遵照指示，保持 200 步
    
    relative_run_dir = "diffusion_policy/data/outputs/2026.01.26/16.00.38_train_diffusion_unet_hybrid_peg_in_hole"
    run_dir = os.path.join(project_root, relative_run_dir)
    # ==========================================================================

    print("\n" + "#"*60)
    print(f"🔥 正在评估 Epoch: {TARGET_EPOCH}")
    print("#"*60)

    # 1. 加载 Config
    cfg_path = os.path.join(run_dir, '.hydra', 'config.yaml')
    if not os.path.exists(cfg_path):
        print(f"❌ 找不到配置文件: {cfg_path}")
        return
    cfg = OmegaConf.load(cfg_path)

    # 2. 加载模型
    try:
        ckpt_path = get_checkpoint_by_epoch(run_dir, TARGET_EPOCH)
        print(f"🚀 Loading checkpoint: {os.path.basename(ckpt_path)}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    try:
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
    except Exception as e:
        from diffusion_policy.workspace.base_workspace import BaseWorkspace
        workspace = BaseWorkspace(cfg)
        
    workspace.load_checkpoint(ckpt_path)
    policy = workspace.model
    policy.eval()
    
    device = torch.device('cuda:0')
    policy.to(device)

    # 3. 创建环境
    env = gym.make('FrankaPegInHole-v0', render_mode='rgb_array', control_mode='ee', disable_env_checker=True)

    # 4. 开始测试
    success_count = 0
    success_steps = []
    rewards = []

    # 🟢 进度条设置
    pbar = tqdm(range(N_EPISODES), desc=f"Ep {TARGET_EPOCH}")
    
    for i in pbar:
        raw_obs, _ = env.reset()
        
        n_obs_steps = 2
        obs_deque = collections.deque([raw_obs] * n_obs_steps, maxlen=n_obs_steps)
        
        done = False
        steps = 0
        policy.reset()
        episode_reward = 0
        is_success = False
        
        while not done and steps < MAX_STEPS:
            batch = {'image': [], 'image_wrist': [], 'state': []}
            for o in obs_deque:
                batch['image'].append(o['image']) 
                batch['image_wrist'].append(o['image_wrist'])
                
                if 'state' in o: s = o['state']
                elif 'observation' in o: s = o['observation']
                else: s = np.zeros(19, dtype=np.float32)
                batch['state'].append(s)
            
            t_img = torch.from_numpy(np.stack(batch['image'])).float().unsqueeze(0).to(device)
            t_wri = torch.from_numpy(np.stack(batch['image_wrist'])).float().unsqueeze(0).to(device)
            t_state = torch.from_numpy(np.stack(batch['state'])).float().unsqueeze(0).to(device)
            
            inp = {'image': t_img, 'image_wrist': t_wri, 'state': t_state}
            
            with torch.no_grad():
                result = policy.predict_action(inp)
            
            action = result['action'][0, 0].cpu().numpy()
            action_to_env = action.flatten()[:4]
            
            raw_obs, reward, terminated, truncated, info = env.step(action_to_env)
            obs_deque.append(raw_obs)
            
            episode_reward += reward
            steps += 1
            
            if info.get('is_success', False):
                is_success = True
                break
        
        if is_success:
            success_count += 1
            success_steps.append(steps)
        
        rewards.append(episode_reward)
        
        # 🟢 [这里恢复了原来的显示逻辑]
        current_sr = success_count / (i + 1)
        current_avg_steps = np.mean(success_steps) if success_steps else 0
        
        pbar.set_postfix({
            "SR": f"{current_sr:.1%}", 
            "Avg Steps": f"{current_avg_steps:.1f}",  # 实时平均步数
            "Last Steps": steps if is_success else "Fail" # 上一次步数
        })

    env.close()
    
    print("\n" + "="*50)
    print(f"📊 评估结果 (Epoch {TARGET_EPOCH})")
    print("="*50)
    print(f"✅ 成功率: {success_count/N_EPISODES:.2%} ({success_count}/{N_EPISODES})")
    print(f"⚡ 平均步数: {np.mean(success_steps) if success_steps else 0:.2f}")
    print(f"🚀 最快步数: {np.min(success_steps) if success_steps else 0}")
    print(f"🐢 最慢步数: {np.max(success_steps) if success_steps else 0}")
    print("="*50)

if __name__ == "__main__":
    main()