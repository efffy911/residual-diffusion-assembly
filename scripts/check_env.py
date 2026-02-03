import mujoco
import numpy as np
import cv2
import time
import os

# ================= 配置区域 =================
# 这里请填入你的主 XML 文件路径 (即包含 <include file="panda_mocap.xml"/> 的那个文件)
# 假设你的文件结构是:
# root/
#   |- assets/peg_in_hole.xml
#   |- assets/panda_mocap.xml
#   |- scripts/check_env.py
XML_PATH = "panda_mujoco_gym/assets/peg_in_hole.xml"  # 请根据实际情况修改路径

# 相机分辨率 (仅用于预览)
WIDTH = 640
HEIGHT = 480
# ===========================================

def main():
    print(f"🔍 正在加载模型: {XML_PATH}")
    
    # 1. 加载模型
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except ValueError as e:
        print(f"❌ 加载失败: {e}")
        print("请检查 XML_PATH 路径是否正确，以及 panda_mocap.xml 是否在同一目录下。")
        return

    # 2. 检查相机列表
    camera_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    print(f"✅ 检测到的相机列表: {camera_names}")

    if "wrist_camera" not in camera_names:
        print("❌ 错误: 未找到 'wrist_camera'！")
        print("请检查是否已保存 panda_mocap.xml，并且 XML 标签拼写正确。")
        return
    else:
        print("🎉 成功找到 'wrist_camera'！准备渲染...")

    # 3. 初始化渲染器
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    print("\n🎥 正在打开预览窗口... (按 'q' 键退出)")
    print("左图: 全局视角 (watching) | 右图: 手眼相机 (wrist_camera)")

    # 4. 循环渲染
    while True:
        # 让物理引擎跑一步 (让机器人受重力自然下垂，或者你可以加控制逻辑)
        mujoco.mj_step(model, data)

        # --- 渲染全局相机 ---
        # 如果你的全局相机名字不是 'watching'，请在这里修改，或者用 camera_names[0]
        try:
            renderer.update_scene(data, camera="watching")
            img_global = renderer.render()
        except Exception:
            # 如果找不到 watching，就用默认视角
            renderer.update_scene(data) 
            img_global = renderer.render()

        # --- 渲染手眼相机 ---
        renderer.update_scene(data, camera="wrist_camera")
        img_wrist = renderer.render()

        # --- 图像处理与拼接 ---
        # MuJoCo 输出是 RGB，OpenCV 需要 BGR
        img_global = cv2.cvtColor(img_global, cv2.COLOR_RGB2BGR)
        img_wrist = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)

        # 在图像上加文字标签
        cv2.putText(img_global, "Global View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_wrist, "Wrist View (New)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 水平拼接两张图
        combined_img = np.hstack((img_global, img_wrist))

        # --- 显示 ---
        cv2.imshow("Check Environment - Press 'q' to exit", combined_img)

        # 每 10ms 刷新一次 (按 q 退出)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 检查 XML 文件是否存在
    if not os.path.exists(XML_PATH):
        print(f"❌ 找不到文件: {XML_PATH}")
        print("请修改脚本中的 XML_PATH 变量！")
    else:
        main()