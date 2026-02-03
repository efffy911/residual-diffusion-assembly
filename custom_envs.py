import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces # 🟢 用于手动注册观测空间
import mujoco  

# 导入相关类
from panda_mujoco_gym.envs.pick_and_place import FrankaPickAndPlaceEnv
from panda_mujoco_gym.envs.panda_env import FrankaEnv

class FrankaPegInHoleEnv(FrankaPickAndPlaceEnv):
    # 强制元数据匹配 20Hz
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, reward_type="sparse", **kwargs):
        # 1. 构造 XML 路径
        project_root = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(project_root, "panda_mujoco_gym", "assets", "peg_in_hole.xml")
        
        if not os.path.exists(xml_path):
            print(f"⚠️ 警告: 找不到 XML 文件: {xml_path}")

        self.reward_type = reward_type

        # =========================================================
        # 🟢 [关键修复 1] 必须在父类 init 之前定义这些参数！
        # 因为父类 init 会调用 _get_obs，而 _get_obs 需要用到它们。
        # =========================================================
        self.img_width = 96
        self.img_height = 96

        # 2. 初始化底层环境
        FrankaEnv.__init__(
            self,
            model_path=xml_path,
            n_substeps=25,          
            block_gripper=False,
            render_mode=kwargs.get("render_mode"),
        )
        
        # =========================================================
        # 🟢 [关键修复 2] 更新 Observation Space
        # 告诉 Gym 我们会多返回两个图像数据，否则 passive checker 会报错
        # =========================================================
        # 获取父类已经定义好的空间字典
        obs_spaces = self.observation_space.spaces
        
        # 手动注册 'image' (全局) 和 'image_wrist' (手眼)
        obs_spaces["image"] = spaces.Box(
            low=0.0, high=1.0, shape=(3, 96, 96), dtype=np.float32
        )
        obs_spaces["image_wrist"] = spaces.Box(
            low=0.0, high=1.0, shape=(3, 96, 96), dtype=np.float32
        )
        
        # 重新打包赋值给 self.observation_space
        self.observation_space = spaces.Dict(obs_spaces)

    def reset(self, seed=None, options=None):
        # 1. 确保随机数生成器同步
        super().reset(seed=seed) 

        # ====================================================
        # 🟢 坐标随机化与安全距离检查 (Rejection Sampling)
        # ====================================================
        x_min, x_max = 0.3, 0.6
        y_min, y_max = -0.25, 0.25
        min_dist = 0.15 

        while True:
            hole_xy = self.np_random.uniform(low=[x_min, y_min], high=[x_max, y_max])
            peg_xy = self.np_random.uniform(low=[x_min, y_min], high=[x_max, y_max])
            dist = np.linalg.norm(hole_xy - peg_xy)
            if dist > min_dist:
                break
        
        # ====================================================
        # 🟢 应用位置到 MuJoCo
        # ====================================================
        # 1. 移动方孔 (Hole)
        hole_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hole_base")
        hole_mocap_id = self.model.body_mocapid[hole_body_id]
        
        self.data.mocap_pos[hole_mocap_id][0] = hole_xy[0]
        self.data.mocap_pos[hole_mocap_id][1] = hole_xy[1]
        self.data.mocap_pos[hole_mocap_id][2] = 0.0 

        self.data.mocap_quat[hole_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)
        
        # 2. 移动轴 (Peg)
        peg_z = 0.05 
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_joint")
        qpos_adr = self.model.jnt_qposadr[jnt_id]
        qvel_adr = self.model.jnt_dofadr[jnt_id]

        self.data.qpos[qpos_adr] = peg_xy[0]
        self.data.qpos[qpos_adr + 1] = peg_xy[1]
        self.data.qpos[qpos_adr + 2] = peg_z
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0

        # 3. 动态增强夹爪摩擦力
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and "finger" in name:
                self.model.geom_friction[i] = np.array([5.0, 0.005, 0.0001])
                self.model.geom_condim[i] = 4 

        # 4. 刷新物理引擎
        mujoco.mj_forward(self.model, self.data)

        # 5. 重新获取观测
        obs = self._get_obs()
        return obs, {}
    
    def _sample_goal(self):
        try:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hole_target")
            return self.data.site_xpos[site_id].copy()
        except Exception:
            return np.array([0.5, 0.0, 0.05])

    def _get_obs(self):
        """
        🟢 [重写] 获取观测数据，包含双相机图像
        """
        # =========================================================
        # 🟢 [修复] 延迟初始化 & 改名避免冲突 (mujoco_renderer -> custom_renderer)
        # =========================================================
        if not hasattr(self, "custom_renderer") or self.custom_renderer is None:
            # 创建 DeepMind 原生渲染器
            self.custom_renderer = mujoco.Renderer(self.model, height=self.img_height, width=self.img_width)
        
        # 1. 获取底层观测
        obs = super()._get_obs()
        
        # 2. 渲染图像 (双视角)
        # (A) 渲染全局相机
        self.custom_renderer.update_scene(self.data, camera="watching") 
        image_global = self.custom_renderer.render()                    

        # (B) 渲染手眼相机 (必须和 XML 里的名字一致)
        self.custom_renderer.update_scene(self.data, camera="wrist_camera") 
        image_wrist = self.custom_renderer.render()                         

        # 3. 数据处理
        # 转换格式: (H, W, C) -> (C, H, W)
        image_global = np.moveaxis(image_global, -1, 0)
        image_wrist = np.moveaxis(image_wrist, -1, 0)

        # 归一化
        image_global = image_global.astype(np.float32) / 255.0
        image_wrist = image_wrist.astype(np.float32) / 255.0

        # 4. 存入字典
        obs["image"] = image_global
        obs["image_wrist"] = image_wrist
        obs["desired_goal"] = self._sample_goal()
        
        return obs
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
            
        d_xy = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=-1)
        d_z = np.abs(achieved_goal[:, 2] - desired_goal[:, 2])

        # XY < 3mm, Z < 5cm
        success_mask = (d_xy < 0.003) & (d_z < 0.05)
        
        if self.reward_type == "sparse":
            return success_mask.astype(np.float32) - 1.0
        else:
            dist = d_xy + d_z 
            return -dist
            
    def step(self, action):
        obs, _, _, truncated, info = super().step(action)
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())
            
        is_success = (reward == 0.0)
        terminated = is_success
        info["is_success"] = is_success
        
        return obs, reward, bool(terminated), bool(truncated), info

# 注册环境
gym.register(
    id="FrankaPegInHole-v0",
    entry_point="custom_envs:FrankaPegInHoleEnv",
    max_episode_steps=400,
)