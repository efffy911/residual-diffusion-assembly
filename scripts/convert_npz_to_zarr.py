import sys
import os
import numpy as np
import zarr
from numcodecs import Blosc

# ================= 配置区域 =================
# 🔴 [关键] 请修改为你刚刚采集生成的 .npz 文件路径
# 例如: "data/demo_npz/peg_in_hole_demo_dual_cam_300eps_20260124_xxxxxx.npz"
INPUT_NPZ_PATH = "data/demo_npz/peg_in_hole_demo_dual_cam_300eps_20260126_152236.npz"

# 输出文件路径 (自动替换后缀)
OUTPUT_ZARR_PATH = INPUT_NPZ_PATH.replace(".npz", ".zarr")
# ===========================================

def convert_npz_to_zarr(npz_path, zarr_path):
    print(f"🔄 Loading NPZ from: {npz_path}")
    
    # 1. 加载 NPZ 数据
    data = np.load(npz_path)
    
    # 读取所有键值 (注意键名要和 collect_data_npz.py 里存的一致)
    # collect_data_npz 存的是: image, image_wrist, state, action, episode_ends
    np_img_global = data['image']       # 全局图
    np_img_wrist = data['image_wrist']  # 🟢 手眼图
    np_state = data['state']
    np_action = data['action']
    np_episode_ends = data['episode_ends']
    
    print(f"   Shape Check:")
    print(f"   - Global Img: {np_img_global.shape}")
    print(f"   - Wrist Img : {np_img_wrist.shape}") # 🟢
    print(f"   - State     : {np_state.shape}")
    print(f"   - Action    : {np_action.shape}")
    print(f"   - Eps Ends  : {np_episode_ends.shape}")

    # 2. 创建 Zarr 根组
    print(f"📂 Creating Zarr group at: {zarr_path}")
    # mode='w' 会覆盖旧文件，请小心
    root = zarr.open(zarr_path, mode='w')
    
    # 3. 定义压缩器
    compressor = Blosc(cname='zstd', clevel=3, shuffle=1)

    # 4. 创建 'data' 组
    data_group = root.create_group('data')
    
    # --- 🟢 写入全局图像 (image) ---
    # 形状通常是 (N, C, H, W) 或 (N, H, W, C)，根据采集时的格式
    # 我们这里假设采集时已经是 (N, C, H, W)
    chunks_img = (100,) + np_img_global.shape[1:] 
    data_group.create_dataset(
        'image', 
        data=np_img_global, 
        chunks=chunks_img, 
        compressor=compressor, 
        dtype=np_img_global.dtype
    )
    print("   ✅ Wrote 'data/image'")

    # --- 🟢 写入手眼图像 (image_wrist) ---
    data_group.create_dataset(
        'image_wrist', 
        data=np_img_wrist, 
        chunks=chunks_img, # 使用相同的 chunk 大小
        compressor=compressor, 
        dtype=np_img_wrist.dtype
    )
    print("   ✅ Wrote 'data/image_wrist'")

    # --- 写入状态 (state) ---
    chunks_state = (100, np_state.shape[1])
    data_group.create_dataset(
        'state', 
        data=np_state, 
        chunks=chunks_state, 
        compressor=compressor, 
        dtype=np_state.dtype
    )
    print("   ✅ Wrote 'data/state'")

    # --- 写入动作 (action) ---
    chunks_action = (100, np_action.shape[1])
    data_group.create_dataset(
        'action', 
        data=np_action, 
        chunks=chunks_action, 
        compressor=compressor, 
        dtype=np_action.dtype
    )
    print("   ✅ Wrote 'data/action'")

    # 5. 创建 'meta' 组
    meta_group = root.create_group('meta')
    
    # --- 写入 episode_ends ---
    meta_group.create_dataset(
        'episode_ends', 
        data=np_episode_ends, 
        dtype=np_episode_ends.dtype
    )
    print("   ✅ Wrote 'meta/episode_ends'")

    print("🎉 Conversion Complete!")
    print(f"   Output saved to: {zarr_path}")

def verify_zarr(zarr_path):
    print(f"\n🔍 Verifying Zarr file...")
    root = zarr.open(zarr_path, mode='r')
    print("   Zarr Tree Structure:")
    print(root.tree())
    
    # 简单的读取测试
    img_g = root['data']['image']
    img_w = root['data']['image_wrist']
    print(f"   Read Test - Global Img Shape: {img_g.shape}")
    print(f"   Read Test - Wrist Img Shape : {img_w.shape}")
    
    if img_g.shape[0] != img_w.shape[0]:
        print("❌ Warning: Global and Wrist image counts do not match!")
    else:
        print("   ✅ Verification Passed.")

if __name__ == "__main__":
    if not os.path.exists(INPUT_NPZ_PATH):
        print(f"❌ Error: Input file not found: {INPUT_NPZ_PATH}")
        print("Please update INPUT_NPZ_PATH in the script.")
    else:
        convert_npz_to_zarr(INPUT_NPZ_PATH, OUTPUT_ZARR_PATH)
        verify_zarr(OUTPUT_ZARR_PATH)