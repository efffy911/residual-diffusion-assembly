import os
import zarr
from omegaconf import OmegaConf
# 尝试导入数据集类
try:
    from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
except ImportError:
    print("❌ Import 失败，请检查环境安装")
    exit()

# 🔴 这里填你刚才 ls -d 确认过的绝对路径
dataset_path = "/home/wtf/projects/residual-diffusion-assembly/data/demo_npz/peg_in_hole_demo_300eps_20260121_200553.zarr"

print(f"\n🔍 1. 系统层检查:")
print(f"   路径字符串: '{dataset_path}'")
print(f"   是否存在: {os.path.exists(dataset_path)}")
print(f"   是否为目录: {os.path.isdir(dataset_path)}")

if not os.path.exists(dataset_path):
    print("❌ Python 找不到该路径！请检查路径拼写或权限。")
    exit()

print(f"\n🔍 2. Zarr 库检查:")
try:
    f = zarr.open(dataset_path, mode='r')
    print(f"   ✅ Zarr 打开成功！Tree结构:")
    print(f"   {f.tree()}")
except Exception as e:
    print(f"❌ Zarr 打开失败: {e}")
    exit()

print(f"\n🔍 3. Dataset 类实例化检查:")
# 模拟 Config
shape_meta = OmegaConf.create({
    'obs': {
        'img': {'shape': [3, 96, 96], 'type': 'rgb'},
        'state': {'shape': [19], 'type': 'low_dim'}
    },
    'action': {'shape': [7]}
})

try:
    ds = RobomimicReplayImageDataset(
        dataset_path=dataset_path,
        shape_meta=shape_meta,
        n_obs_steps=2,
        horizon=16,
        pad_before=1,
        pad_after=7
    )
    print("🎉🎉🎉 成功！Dataset 类可以正常加载该路径！")
    print("👉 结论：你的路径和文件是完美的。问题出在 YAML 配置文件没写对。")
except Exception as e:
    print(f"❌ Dataset 实例化崩溃: {e}")
    # 这里可能会报错缺少 metadata，如果是这个问题，我会教你修 Zarr