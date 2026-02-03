import numpy as np

def _clip(a, lo=-1.0, hi=1.0):
    return np.clip(a, lo, hi).astype(np.float32)

class ScriptedPegInHolePolicy:
    """
    专门为 Peg-in-Hole (方孔插轴) 改造的脚本策略。
    流程: 抓取 -> 抬高 -> 移动到孔正上方(精细对齐) -> 垂直插入 -> 松手撤退
    """

    def __init__(
        self,
        kp_xy=12.0,              
        kp_z=10.0,
        xy_tol=0.002,           # 对齐容差

        grasp_offset=np.array([-0.004, 0.0, 0.0]),
        hover_height=0.15,      
        grasp_depth=-0.03,      
        lift_height=0.20,       
        
        # 插孔相关参数
        hole_hover_height=0.15, 
        insert_depth_margin=0.01, 

        close_steps=20,
        open_steps=20,

        max_steps_per_phase=300, 
        descend_steps=100,
        insert_steps=150,        

        # contact stabilization
        press_dz=0.0,         
        z_safety_max=0.60,      

        verbose=False,
    ):
        self.kp_xy = kp_xy
        self.kp_z = kp_z
        self.xy_tol = xy_tol

        self.hover_height = hover_height
        self.grasp_offset = grasp_offset
        self.grasp_depth = grasp_depth
        self.lift_height = lift_height
        self.hole_hover_height = hole_hover_height
        self.insert_depth_margin = insert_depth_margin

        self.close_steps = close_steps
        self.open_steps = open_steps

        self.max_steps_per_phase = max_steps_per_phase
        self.descend_steps = descend_steps
        self.insert_steps = insert_steps

        self.press_dz = press_dz
        self.z_safety_max = z_safety_max

        self.verbose = verbose
        self.reset()

    def reset(self):
        self.phase = "HOVER_OBJ"
        self.phase_step = 0

        self.obj_z0 = None          
        self.lock_obj_xy = None     
        self.lock_goal_xy = None    

        self.lock_ee_pos = None
        self.lift_start_xy = None  # 🟢 [新增] LIFT阶段专用的起飞坐标

        # 🟢 [新增] 每一集随机生成“脾气不一样”的插入策略
        # 幅度: 3mm ~ 6mm 之间随机
        self.rand_w_amp = np.random.uniform(0.003, 0.006)
        # 频率: 0.3 ~ 0.8 之间随机 (有的快有的慢)
        self.rand_w_freq = np.random.uniform(0.3, 0.8)
        # 方向: 随机顺时针或逆时针 (1.0 或 -1.0)
        self.rand_w_dir = np.random.choice([1.0, -1.0])

    def step_phase_counter(self):
        self.phase_step += 1

    def _ee_pos(self, obs):
        return obs["observation"][:3].astype(np.float32)

    # 修改 _goto 定义，增加 override 参数
    def _goto(self, ee, target, grip, kp_xy_override=None, kp_z_override=None):
        target = target.copy()
        target[2] = float(np.clip(target[2], -1e9, self.z_safety_max))

        d = target - ee
        a = np.zeros(4, dtype=np.float32)
        
        # 如果有传入自定义刚度，就用自定义的，否则用默认的
        use_kp_xy = kp_xy_override if kp_xy_override is not None else self.kp_xy
        use_kp_z  = kp_z_override  if kp_z_override  is not None else self.kp_z
        
        a[0] = use_kp_xy * d[0]
        a[1] = use_kp_xy * d[1]
        a[2] = use_kp_z  * d[2]
        a[3] = grip
        return _clip(a)

    def _bump(self, nxt, obj=None, goal=None):
        if self.verbose:
            print(f"[Policy] {self.phase} -> {nxt}")

        # 清理 LIFT 锁
        if nxt != "LIFT":
            self.lift_start_xy = None
        
        # 🟢 [关键修改] 如果要去 CLOSE 或 RELEASE，都保留 lock_ee_pos
        # 只要 nxt 不在这些状态里，才清空
        if nxt not in ("CLOSE", "RELEASE"):
            self.lock_ee_pos = None

        self.phase = nxt
        self.phase_step = 0

        if obj is not None and nxt in ("DESCEND", "CLOSE", "LIFT"):
            self.lock_obj_xy = np.array([obj[0], obj[1]], dtype=np.float32)
        
        if goal is not None and nxt in ("ALIGN_HOLE", "INSERT"):
            self.lock_goal_xy = np.array([goal[0], goal[1]], dtype=np.float32)

    def act(self, obs):
        ee = self._ee_pos(obs)
        obj = obs["achieved_goal"].astype(np.float32)
        goal = obs["desired_goal"].astype(np.float32) 

        if self.obj_z0 is None:
            self.obj_z0 = float(obj[2])

        # ---------------------------------------------------------
        # 1-4. HOVER, DESCEND, SETTLE, CLOSE, LIFT, MOVE_TO_GOAL, ALIGN_HOLE 保持原有逻辑
        # ---------------------------------------------------------
        if self.phase == "HOVER_OBJ":
            grasp_center = obj[:3] + self.grasp_offset
            target = np.array([grasp_center[0], grasp_center[1], self.obj_z0 + self.hover_height], dtype=np.float32)
            a = self._goto(ee, target, grip=+1.0)
            dist_xy = np.linalg.norm((ee[:2] - grasp_center[:2]))
            if (dist_xy < 0.008 and abs(ee[2] - target[2]) < 0.02):
                self._bump("DESCEND", obj=obj)
            return a

        if self.phase == "DESCEND":
            grasp_center = obj[:3] + self.grasp_offset
            grasp_z = self.obj_z0 - self.grasp_depth
            target = np.array([grasp_center[0], grasp_center[1], grasp_z], dtype=np.float32)
            a = self._goto(ee, target, grip=+1.0)
            dist_xy = np.linalg.norm((ee[:2] - grasp_center[:2]))
            dist_z = abs(ee[2] - grasp_z)
            if dist_z < 0.006 and dist_xy < 0.006:
                self._bump("CLOSE", obj=obj)
            elif self.phase_step > 100:
                self._bump("CLOSE", obj=obj)
            return a

        if self.phase == "CLOSE":
            if self.lock_ee_pos is None:
                self.lock_ee_pos = ee.copy() 
            target = self.lock_ee_pos
            a = self._goto(ee, target, grip=-1.0)
            if self.phase_step > 5:
                self._bump("LIFT", obj=obj)
            return a

        if self.phase == "LIFT":
            if self.lift_start_xy is None:
                self.lift_start_xy = ee[:2].copy()
            target_z = self.obj_z0 + self.lift_height
            target = np.array([self.lift_start_xy[0], self.lift_start_xy[1], target_z], dtype=np.float32)
            a = self._goto(ee, target, grip=-1.0)
            if ee[2] > (target_z - 0.02) or self.phase_step > 50:
                self._bump("MOVE_TO_GOAL", goal=goal)
            return a

        if self.phase == "MOVE_TO_GOAL":
            target_z = self.obj_z0 + self.lift_height
            target = np.array([goal[0], goal[1], target_z], dtype=np.float32)
            a = self._goto(ee, target, grip=-1.0)
            dist_xy_goal = np.linalg.norm((ee[:2] - goal[:2]))
            if dist_xy_goal < 0.02: 
                self._bump("ALIGN_HOLE", goal=goal)
            return a

        # ---------------------------------------------------------
        # 5. ALIGN_HOLE: 最终稳定版 (回归本真)
        # ---------------------------------------------------------
        if self.phase == "ALIGN_HOLE":
            error_xy = goal[:2] - obj[:2]
            dist = np.linalg.norm(error_xy)
            hover_z = goal[2] + self.hole_hover_height
            
            if dist > 0.01: 
                # [靠近阶段] 距离 > 1cm
                # 低刚度平滑靠近
                kp_val = 15.0
                mode = "Approach"
            else:
                # [锁死阶段] 距离 < 1cm
                # 🟢 [关键] 只用高刚度，坚决不用 Factor 欺骗！
                # 之前验证过，KP=30 能稳在 2.3mm，这就够了。
                kp_val = 30.0
                mode = "Lock-in"

            # 始终保持 1.0，不骗机器人，防止震荡
            target_xy = ee[:2] + error_xy * 1.0
            target = np.array([target_xy[0], target_xy[1], hover_z], dtype=np.float32)
            
            a = self._goto(ee, target, grip=-1.0, kp_xy_override=kp_val)

            if self.verbose and self.phase_step % 10 == 0:
                print(f"[ALIGN] Dist={dist:.4f} Mode={mode} KP={kp_val}")

            # 🟢 [最终阈值] 0.003 (3mm)
            # 既然物理平衡点在 2.3mm，我们就把线划在 3mm。
            # 这不是妥协，这是工程智慧。2.3mm 的精度配合 Wiggle 100% 能插进去。
            if dist < 0.0025:
                if self.phase_step > 5:
                    print(f"✅ Aligned! Err: {dist:.5f}")
                    self._bump("INSERT", goal=goal)
            
            # 超时保护
            elif self.phase_step > 80:
                print(f"⚠️ Timeout. Err: {dist:.5f}")
                self._bump("INSERT", goal=goal)
                
            return a
        
        # ---------------------------------------------------------
        # 6. INSERT: 垂直下插 (配合宽阈值，增加Wiggle)
        # ---------------------------------------------------------
        if self.phase == "INSERT":
            if self.lock_goal_xy is None:
                 self.lock_goal_xy = ee[:2].copy()
            
            target_xy = self.lock_goal_xy
            
            # 目标高度
            partial_insert_z = goal[2] + 0.07 
            target = np.array([target_xy[0], target_xy[1], partial_insert_z], dtype=np.float32)
            
            # 🟢 [优化4] 因为对齐阈值放宽了，Wiggle 幅度稍微给大一点点 (0.3 -> 0.5)
            # 帮它"晃"进去
            w_amp = self.rand_w_amp * 0.5 
            w_freq = self.rand_w_freq
            direction = self.rand_w_dir
            target[0] += w_amp * np.sin(self.phase_step * w_freq * direction)
            target[1] += w_amp * np.cos(self.phase_step * w_freq * direction)

            a = self._goto(ee, target, grip=-1.0, kp_xy_override=10.0, kp_z_override=15.0)
            
            if ee[2] < (partial_insert_z + 0.005):
                # 这里不需要记录 lock_ee_pos 了，因为下一步我们会动态读取
                self._bump("RELEASE")
            return a

        # ---------------------------------------------------------
        # 7. RELEASE: 动态松手 + 瞬时微抬 (彻底解决漂移)
        # ---------------------------------------------------------
        if self.phase == "RELEASE":
            # 🟢 [核心修复] 不要锁死旧坐标！读取"当前"坐标！
            # 每一帧都把目标设为当前的 XY，意味着 XY 轴完全顺从物理引擎，
            # 这样就不会出现"往孔的方向平移"这种对抗动作。
            current_xy = ee[:2] 
            
            # 目标：一边松手，一边利用这一瞬间稍微往上提 1cm
            # 这样既松开了物体，又为下一步 LIFT 做了预备，非常丝滑
            lift_z = ee[2] + 0.01
            target = np.array([current_xy[0], current_xy[1], lift_z], dtype=np.float32)
            
            # grip = +1.0 (打开)
            a = self._goto(ee, target, grip=+1.0)
            
            # 🟢 [优化5] 只要 10 步 (0.5秒) 让夹爪张开即可
            if self.phase_step > 10: 
                # 顺手把之前的锁清空
                self.lock_ee_pos = None 
                self.lock_goal_xy = None
                self._bump("LIFT_FIST")
            return a

        # ---------------------------------------------------------
        # 8. LIFT_FIST: 低空变拳 (节省时间)
        # ---------------------------------------------------------
        if self.phase == "LIFT_FIST":
            # 🟢 [优化6] 降低抬升高度
            # 墙高0.10，只要抬到 0.14 就足够变拳头了 (之前是0.18)
            fist_target_z = goal[2] + 0.14 
            
            # XY 依然对准孔中心 (goal)，准备下压
            target = np.array([goal[0], goal[1], fist_target_z], dtype=np.float32)
            
            # 变拳头
            a = self._goto(ee, target, grip=-1.0)
            
            # 判据：高度到位 (放宽到 2cm 误差)
            if abs(ee[2] - fist_target_z) < 0.02:
                self._bump("PUSH")
            return a

        # ---------------------------------------------------------
        # 9. PUSH: 垂直下压 (保持 0.108 不变)
        # ---------------------------------------------------------
        if self.phase == "PUSH":
            safe_push_z = 0.108
            target = np.array([goal[0], goal[1], safe_push_z], dtype=np.float32)
            a = self._goto(ee, target, grip=-1.0, kp_z_override=20.0)
            
            if abs(ee[2] - safe_push_z) < 0.002:
                return self._goto(ee, target, grip=-1.0)
            elif self.phase_step > 100:
                return self._goto(ee, target, grip=-1.0)
            return a
            
        return np.zeros(4, dtype=np.float32)