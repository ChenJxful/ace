import cv2
import time
# import torch
import numpy as np
import open3d as o3d

from franka.control.franka_control import OSC_Control, Joint_Control
from franka.sensors.realsense import RealSenseCamera
from franka.perception import depth_2_point, single_point_to_pc, trans_point_to_base, get_camera_in_base_mat
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.transform_utils import quat_slerp
from deoxys import config_root
from franka.pre_defined_pose import *
from vgn.perception import TSDFVolume
from vgn.utils.transform import Rotation, Transform
from franka.bbox import AABBox

pos = None
ori = None


class Franka():
    def __init__(self) -> None:
        self.camera = RealSenseCamera()
        self.intr = self.camera.get_intr()

        osc_controller_type = "OSC_POSE"
        joint_controller_type = "JOINT_POSITION"
        controller_config = "charmander.yml"  # 控制模式 和 配置文件
        robot_interface = FrankaInterface(
            config_root + f"/{controller_config}", use_visualizer=False)

        self.osc_controller = OSC_Control(robot_interface, osc_controller_type)
        self.joint_controller = Joint_Control(robot_interface, joint_controller_type)
        self.reset()

    def reset(self):
        # reset the robot
        time.sleep(0.2)
        self.joint_controller.control(p_reset, grasp=False)
        print("Waiting for the robot to be ready...")
        # avoid the error depth image, skip first 5 frames
        for k in range(5):
            self.camera.get_sensor_info()
            time.sleep(0.2)

        print("Robot is ready!")

    def get_extrinsic(self):
        # get extrinsic matrix of camera in base frame?
        camera_in_base = Transform.from_matrix(
            get_camera_in_base_mat(self.osc_controller.last_eef_pos))
        return camera_in_base

    def goto_extrinsic(self, extrinsic: np.ndarray, grasp=False):
        '''segmented move to target extrinsic matrix
        '''
        current_quat, current_pose = self.osc_controller.last_eef_quat_and_pos
        distance = np.linalg.norm(extrinsic[:3, 3] - current_pose)
        # target_quat = Rotation.from_matrix(extrinsic[:3, :3] @ np.array([[0,0,1],[0,-1,0],[1,0,0]])).as_quat()
        target_quat = Rotation.from_matrix(extrinsic[:3, :3]).as_quat()

        seg_step = max(1, int(distance / 0.11))
        for i in range(seg_step):
            t_pose = (current_pose.transpose() + (extrinsic[:3, 3] - current_pose.transpose()) * (i + 1) / seg_step).transpose()
            t_quat = quat_slerp(current_quat, target_quat, (i + 1) / seg_step)
            self.osc_controller.osc_move((t_pose, t_quat), grasp=grasp, num_steps=6 if i < seg_step - 1 else 40)


    def go_grasp(self, grasp_pose: Transform, pre_depth=0.1):
        grasp_xyz = grasp_pose.translation
        grasp_rot = grasp_pose.rotation

        # pre-grasp
        # pre_grasp_pose = Transform(grasp_rot, grasp_xyz + pre_depth * direction_vector)
        # self.goto_extrinsic(pre_grasp_pose.as_matrix())
        self.goto_extrinsic(Transform(grasp_rot,
                                      (grasp_pose * Transform.from_translation([0.0, 0.0, -pre_depth])).translation).as_matrix())
        time.sleep(0.5)
        breakpoint()

        # grasp
        # fix_grasp_pose = Transform(grasp_rot, grasp_xyz - 0.05 * direction_vector)
        # self.goto_extrinsic(fix_grasp_pose.as_matrix())
        self.goto_extrinsic(Transform(grasp_rot,
                                      (grasp_pose * Transform.from_translation([0.0, 0.0, 0.05])).translation).as_matrix())
        time.sleep(0.5)
        breakpoint()
        current_quat, current_pose = self.osc_controller.last_eef_quat_and_pos
        self.osc_controller.osc_move((current_pose, current_quat), grasp=True, num_steps=10)
        breakpoint()

        # post-grasp
        self.goto_extrinsic(Transform(grasp_rot,
                                      (grasp_pose * Transform.from_translation([0.0, 0.0, -2*pre_depth])).translation).as_matrix(), grasp=True)
        time.sleep(0.5)
        breakpoint()

        # back to home
        self.joint_controller.control(p_reset, grasp=False)
        breakpoint()


if __name__ == "__main__":
    franka = Franka()

    ret, color_image, depth_image, point_cloud = franka.camera.get_sensor_info()
    # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('logs/color.png', color_image)
    # cv2.imwrite('logs/depth.png', camera.get_colorize_depth())

    tsdf = TSDFVolume(0.3, 80)  # 创建空的 TSDF 网格
    bbox = AABBox([0.3, -0.15, 0.05], [0.6, 0.15, 0.35])  # 手动测量
    xyz = np.r_[bbox.center[:2] - 0.15, bbox.min[2] - 0.05]  # 以目标物体为中心 扩展一个0.3*0.3的网格 抬高0.05
    T_base_task = Transform.from_translation(xyz)

    fuse_count = 0
    def goto_pose_and_fuse_depth(pose):
        global fuse_count
        fuse_count += 1
        franka.joint_controller.control(pose, grasp=False)
        time.sleep(0.2)
        _, _, depth_image, _ = franka.camera.get_sensor_info()
        camera_in_base_mat = Transform.from_matrix(
            get_camera_in_base_mat(franka.osc_controller.last_eef_pos))
        tsdf.integrate(depth_image.astype(np.float32),
                       franka.intr, camera_in_base_mat.inverse() * T_base_task)
        o3d.io.write_point_cloud(f'logs/pose_{fuse_count}.pcd', tsdf.get_cloud())

    goto_pose_and_fuse_depth(p_circle_0)
    goto_pose_and_fuse_depth(p_circle_1)
    goto_pose_and_fuse_depth(p_circle_2)
    goto_pose_and_fuse_depth(p_circle_3)
    goto_pose_and_fuse_depth(p_circle_4)
    goto_pose_and_fuse_depth(p_circle_5)

