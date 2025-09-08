import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from pyquaternion import Quaternion

#'d:/Reasearch/ACT/imitation_learning_dp-master/imitation_learning_dp/assets/scenes/scene.xml'
'd:/Reasearch/robotics/gello_software/experiments/sim_task_ur/scenes/ee_transfer_cube.xml'
m = mujoco.MjModel.from_xml_path('D:/Reasearch/VEact-plus-plus/assets/sim_task_ur/ee_insert_block.xml')
d = mujoco.MjData(m)

def set_mocap_pose(pos, quat):
    """设置 mocap 体的位置和姿态
    
    Args:
        pos: [x,y,z] 位置 (米)
        quat: [w,x,y,z] 四元数 (单位四元数)
    """
    d.mocap_pos[0] = pos
    d.mocap_quat[0] = quat
  
# Get the position of the body from the data
def get_info(body_id):
    # Get the position of the body from the data
    body_pos = d.xpos[body_id]
    body_quat = d.xquat[body_id]
    return body_pos, body_quat



class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['pos']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['pos']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts=0.02):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)
            self.curr_left_waypoint = self.left_trajectory.pop(0)

        if not self.left_trajectory:
            return np.concatenate([self.curr_left_waypoint['pos'], 
                                 self.curr_left_waypoint['quat'],
                                 [self.curr_left_waypoint['gripper']]])

        next_left_waypoint = self.left_trajectory[0]
        if self.step_count >= next_left_waypoint['t']:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
            if not self.left_trajectory:
                return None
            next_left_waypoint = self.left_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])

        self.step_count += 1
        return action_left



class PickPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):

      gripper_pick_quat = Quaternion(d.mocap_quat[0])
      gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 0.0, 1.0], degrees=-90)
      gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90)
      # 结果为 [0.5, -0.5, 0.5, 0.5]（wxyz格式）
      t=500 #如果不加上t，第二个关节的完全会导致机械臂碰撞桌面
      t1=0
      self.left_trajectory = [
      {"t": 0+t1, "pos": np.array(d.mocap_pos[0]), "quat": d.mocap_quat[0], "gripper":0},
      {"t": 1100+t, "pos": np.array([1.3, 0.2, 1.2]), "quat": gripper_pick_quat.elements,"gripper":0},
      {"t": 1500+t, "pos": np.array([1.3, 0.2, 1.0]), "quat": gripper_pick_quat.elements,"gripper":0},
      {"t": 2000+t, "pos": np.array([1.3, 0.2, 1.0]), "quat": gripper_pick_quat.elements,"gripper":255},
      {"t": 2500+t, "pos": np.array([1.1, 0.6, 1.6]), "quat": gripper_pick_quat.elements, "gripper":255}, #should be key_frame
      {"t": 3000+t, "pos": np.array([1.1, 1.0, 1.5]), "quat": gripper_pick_quat.elements, "gripper":255},
      {"t": 3500+t, "pos": np.array([1.3, 1.0, 1.2]), "quat": gripper_pick_quat.elements, "gripper":255},
      {"t": 4000+t, "pos": np.array([1.3, 1.0, 1.1]), "quat": gripper_pick_quat.elements, "gripper":0},
      {"t": 4500+t, "pos": np.array([1.3, 1.0, 1.2]), "quat": gripper_pick_quat.elements, "gripper":0},
      {"t": 5000+t, "pos": np.array([1.1, 0.6, 1.6]), "quat": gripper_pick_quat.elements, "gripper":0}
      ]




# ... 前面的代码保持不变 ...

with mujoco.viewer.launch(m, d) as viewer:
    policy = PickPolicy()

    # 先重置再设置初始姿态
    # mujoco.mj_resetData(m, d)
    # keyframe_qpos = np.array([
    #     -0.759, -0.209, 0.984, 0.786, -1.571, -0.759, 0.189, -0.026,
    #     0.163, -0.155, 0.189, -0.026, 0.163, -0.155, 1.104, 0.498,
    #     1.352, 1.000, 0.000, 0.000, 0.000
    # ])
    # d.qpos[:] = keyframe_qpos
    mujoco.mj_resetDataKeyframe(m, d, 0)
    wrist_1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    # 强制同步渲染状态
    mujoco.mj_forward(m, d)
    print(get_info(wrist_1_id))
    viewer.sync()  # 添加初始同步
    
    # 保持计数器初始化
    step_counter = 0
    
    # # 预设的相机位置
    # viewer.cam.fixedcamid = m.camera("angle").id
    # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    while viewer.is_running():
        # action = policy()
        # if action is not None:
        #     set_mocap_pose(action[:3], action[3:7])
        #     d.ctrl[0] = action[7]  # 假设第一个控制器控制夹具

        mujoco.mj_step(m, d)
        
        # # 打印t=3000时的qpos
        # if step_counter == 3500:
        #     print(f"[t={step_counter}] 所有关节qpos值:")
        #     print(d.qpos.copy())
            
        step_counter += 1  # 计数器递增
        viewer.sync()
        time.sleep(0.001)