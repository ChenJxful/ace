# Abstract: 测试抓取, 多轮实验管理

import os
import json
import shutil
from typing import Any
import torch
import hydra
import argparse
import itertools
import cv2
import numpy as np
from tqdm import tqdm
from hydra import compose, initialize
from models.networks import Runner
from pathlib import Path
from utils.misc import EasyDict, set_random_seed, blockPrint, enablePrint
from vgn.io import my_IO
from vgn.grasp import Grasp, Label
from vgn.perception import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.misc import apply_noise
from vgn.utils.transform import Rotation, Transform
from franka.bbox import AABBox
import matplotlib.pyplot as plt
from utils.visualization import *
from utils.transform import *
from typing import Any, List, Dict, Set, Tuple, Union
from datetime import datetime


def _angles_to_extrinsic(angles: Tuple[float, float, float], size=0.3) -> Transform:
    '''sphereical coordinate to extrinsic matrix
    '''
    r, theta, phi = angles
    # origin = Transform(Rotation.identity(), np.r_[0.0, 0.0, 0.0])  # 右上角
    origin = Transform(Rotation.identity(), np.r_[size / 2, size / 2, 0.0])
    return camera_on_sphere(origin, r, theta, phi)


def _extrinsic_to_angles(transform: Transform, size=0.3) -> Tuple[float, float, float]:
    '''extrinsic matrix to sphereical coordinate
    '''
    origin = Transform(Rotation.identity(), np.r_[size / 2, size / 2, 0.0])
    m = (transform * origin).inverse().as_matrix()

    eye = m[:3, 3]
    forward = m[:3, 2]
    right = m[:3, 0]
    up = -m[:3, 1]

    radius = np.linalg.norm(eye)
    theta = np.arccos(eye[2] / radius)
    phi = np.arctan2(eye[1], eye[0])

    if phi < 0.0:
        phi += 2.0 * np.pi

    return radius, theta, phi


def _get_random_angles(size=0.3) -> Tuple[float, float, float]:
    '''生成指向 origin 的随机相机旋转角度
    '''
    r = np.random.uniform(1.6, 2.4) * size
    theta = np.random.uniform(np.pi / 4.0, np.pi / 2.0-0.2)  # test
    # theta = np.random.uniform(0.0, np.pi / 3.0)  # default pi/4
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    return r, theta, phi


def _generate_extrinsic_neighbors(origin_cam: Transform, size=0.3, num_direction=6,
                                  min_step=0.0, max_step=0.5*np.pi, num_steps=3) -> List[Transform]:
    '''将当前相机绕圆心朝各个方向旋转 生成随机相机外参
    num_direction: 方向的数量
    num_steps: 旋转的角度的数量
    max_step: 最大旋转角度
    '''
    rotated_cams = []
    for q in np.linspace(0, 2 * np.pi, num_direction, endpoint=False):  # 方向
        for s in np.linspace(max_step, min_step, num_steps, endpoint=False):  # 步长
            rotated_cam = origin_cam * Transform(Rotation.identity(), np.array([0.15, 0.15, 0]))
            optic_axis = rotated_cam.inverse().as_matrix()[:3, 3]  # 原点与相机连线
            optic_axis = optic_axis / np.linalg.norm(optic_axis)
            rot_axis = np.cross(optic_axis, np.array([0, 0, 1]))  # 叉乘 找旋转轴
            q_shifted = q + np.random.uniform(0, 2 * np.pi / num_direction)
            rot_axis = Transform(Rotation.from_rotvec(q_shifted * optic_axis), np.array([0, 0, 0])).as_matrix() @ np.r_[rot_axis, 1]  # 转到不同方向
            rot_axis = rot_axis[:3]
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            rotated_cam = rotated_cam * Transform(Rotation.from_rotvec(s * rot_axis), np.array([0, 0, 0]))  # 将相机沿着轴旋转指定弧度
            rotated_cam = rotated_cam * Transform(Rotation.identity(), np.array([-0.15, -0.15, 0]))
            radius, theta, phi = _extrinsic_to_angles(rotated_cam)
            if 0 < theta < np.pi / 3.0:  # 过滤一部分太低的视角
                # if phi < 0.0:
                #     phi += 2.0 * np.pi
                radius = np.random.uniform(0.45, 0.55)  # for real setup
                rotated_cams.append((radius, theta, phi))
    return rotated_cams


class ExtrinsicMonitor:
    '''可视化展示相机外参
    '''
    def __init__(self, intrinsic):
        self.intrinsic = intrinsic
        self.cam_angles = []
        self.candidates_angles = []

    def add_cam(self, angles):
        self.cam_angles.append(angles)

    def add_candidates(self, angles_list: List[Tuple[float, float, float]]):
        self.candidates_angles = angles_list
        self.quality_list = [0.0] * len(angles_list)

    def add_candidatas_with_quality(self, angles_list: List[Tuple[float, float, float]],
                                    quality_list: List[float]):
        self.candidates_angles = angles_list
        self.quality_list = quality_list

    def clear_candidates(self):
        self.candidates_angles = []
        self.quality_list = []

    def draw_and_save(self, path=None):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        for index, angle in enumerate(self.candidates_angles):
            if self.quality_list[index] == 0.0:  # 未验证的相机视角
                color = 'gray'
            else:
                q_scaled = (self.quality_list[index] - 0.9) / 0.1  # 缩放一下 q
                q_scaled = max(0.0, q_scaled)
                color = (q_scaled, 0.0, 1.0 - q_scaled)
            draw_pyramid(ax, _angles_to_extrinsic(angle).as_matrix(),
                         intrinsic=self.intrinsic, height=0.1,
                         color=color,
                         label="candidate")
        for angle in self.cam_angles:
            draw_pyramid(ax, _angles_to_extrinsic(angle).as_matrix(),
                         intrinsic=self.intrinsic, height=0.1,
                         color='red', label="cam")
        draw_floor_grid(ax)
        set_axis_range(ax, [-0.3, 0.6, -0.3, 0.6, -0.3, 0.6])
        fix_3d_axis_equal(ax)
        mark_axis_label(ax)
        if path is not None:
            plt.savefig(path)

    def show(self):
        plt.show()


class GraspPlanner:
    '''抓取规划器 主要包括网络接口、数据准备、后处理
    '''
    def __init__(self, cfg):
        self.net = Runner(cfg)

    def __call__(self, tsdf, pos_list, extrinsic_list, intrinsic_list, size) -> EasyDict:
        result = EasyDict()
        if len(pos_list) == 0:
            result.grasp_label = torch.tensor([0])
            return result
        batch_size = 1000  # 一次预测的点数量 为了保显存
        is_first_batch = True
        for i in tqdm(range(0, len(pos_list), batch_size), disable=True):  # 分批处理
            batch = self._prepare_batch(tsdf,
                                        pos_list[i:i + batch_size],  # 索引会自动处理最后一批不足 batch_size 的情况
                                        extrinsic_list,
                                        intrinsic_list,
                                        size)
            prediction_batch = self.net.predict_grasp(batch)
            prediction_batch = self._clean_prediction(prediction_batch, size)
            if is_first_batch:
                result = prediction_batch
                is_first_batch = False
            else:
                result.append(prediction_batch)  # feature from EasyDict
        return result

    def _prepare_batch(self, tsdf, pos_list, extrinsics, intrinsics, size) -> EasyDict:
        '''将数据转换为网络输入格式, 空间点 normalize
        tsdf: [1, resolution**3]
        pos_list: [n, 1, 3]
        extrinsics: [1, 7]
        intrinsics: [1, 6]
        '''
        # 数据都转成 tensor
        tsdf = tsdf if isinstance(tsdf, torch.Tensor) else torch.tensor(tsdf)
        pos_list = pos_list if isinstance(pos_list, torch.Tensor) else torch.tensor(pos_list)
        if len(pos_list.shape) == 2:
            pos_list = pos_list.unsqueeze(1)
        extrinsics = extrinsics if isinstance(extrinsics, torch.Tensor) else torch.tensor(extrinsics)
        intrinsics = intrinsics if isinstance(intrinsics, torch.Tensor) else torch.tensor(intrinsics)

        batch_size = len(pos_list)
        assert batch_size != 0
        # tsdf = tsdf.repeat(batch_size, 1, 1, 1)  # 网络自动处理 只需要 encode 一次
        extrinsics = extrinsics[None, None, :].repeat(batch_size, 1, 1)
        intrinsics = intrinsics[None, :].repeat(batch_size, 1)

        # normalize
        pos_list = pos_list / size - 0.5  # [0.0, 0.3] normalize to [-0.5, 0.5]

        batch = EasyDict({'tsdf': tsdf, 'point_grasp': pos_list,
                          'camera_extrinsic': extrinsics,
                          'camera_intrinsic': intrinsics})
        return batch

    def _clean_prediction(self, prediction: EasyDict, size) -> EasyDict:
        '''恢复到世界坐标尺寸
        '''
        # prediction.
        # pose.translation = (pose.translation + 0.5) * size
        prediction.grasp_width *= size
        return prediction


class GraspExp:
    '''实验管理
    '''
    def __init__(self, args, save_dir, rng=None, tsdf_resolution=40):
        self.save_dir = save_dir
        self.mode = args.mode
        self.depth_imgs = []
        self.tsdf = TSDFVolume(args.size, tsdf_resolution)  # 40*40*40 大小的空白 TSDF, 注意这个分辨率就会影响后续的网络 feature grid 分辨率
        self.args = args
        # self.intrinsic = None

        if self.mode == 'sim':  # 随机生成仿真环境的模式
            self.create_sim_scene(args, rng=rng)
            self.gui = args.gui
            self.add_noise = args.add_noise
            self.intrinsic = self.sim.camera.intrinsic
            self.intrinsic_list = [getattr(self.intrinsic, k) for k in ['fx', 'fy', 'cx', 'cy', 'width', 'height']]
            self.target_label = None
            self.angles = _get_random_angles()  # 随机初始视角
            self.extrinsic: Transform = _angles_to_extrinsic(self.angles)  # task & cam frame

        elif self.mode == 'dataset':  # 使用数据集中的观测、抓取点 但此时没有仿真环境可供验证
            self.gui = args.gui
            self.io = my_IO('data/set/data_multiview_grasp_2k')
            self.df = self.io.read_df()
            _, self.intrinsic, _, _ = self.io.read_setup()
            self.intrinsic_list = [getattr(self.intrinsic, k) for k in ['fx', 'fy', 'cx', 'cy', 'width', 'height']]
            scene_id = self.df.loc[0, "scene_id"]
            voxel_grid = self.io.read_voxel_grid(scene_id)
            self.tsdf = torch.tensor(voxel_grid[0], dtype=torch.float32)
            [self.depth_img], [extrinsic_list] = self.io.read_depth_image(scene_id)
            self.depth_imgs.append(self.depth_img)
            self.extrinsic: Transform = Transform.from_list(extrinsic_list)  # task & cam frame
            self.angles = _extrinsic_to_angles(self.extrinsic)
            print(f'Load scene {scene_id} from dataset')

        elif self.mode == 'real':  # 使用真实机械臂
            from franka.my_controller import Franka
            self.franka = Franka()
            self.intrinsic = CameraIntrinsic(self.franka.intr.width,
                                             self.franka.intr.height,
                                             self.franka.intr.fx,
                                             self.franka.intr.fy,
                                             self.franka.intr.ppx,
                                             self.franka.intr.ppy)
            self.intrinsic_list = [getattr(self.intrinsic, k) for k in ['fx', 'fy', 'cx', 'cy', 'width', 'height']]
            xyz = np.r_[0.48 - 0.15,
                        0.00 - 0.15,
                        0.05 - 0.00]  # 以目标物体为中心, 扩展一个0.3*0.3的区域 (需要标定)
            self.t_bbox = AABBox([0.13, 0.13, 0.14],
                                 [0.17, 0.17, 0.18])  # 物体的 bbox (task_frame)

            self.T_base_task = Transform.from_translation(xyz)
            self.extrinsic = self.franka.get_extrinsic().inverse() * self.T_base_task  # task & cam frame
            self.angles = _extrinsic_to_angles(self.extrinsic)
            print('Robot ready!')

    def clear_depth_imgs(self):
        '''场景变化后执行清空已有观测
        '''
        self.depth_imgs = []

    def create_sim_scene(self, args, rng=None):
        '''创建仿真场景
        '''
        assert self.mode == 'sim', 'Only sim mode can create sim scene'

        object_count = 4  # np.random.poisson(args.num_objects) + 1
        self.sim = ClutterRemovalSim(args.scene_type, args.object_set,
                                     gui=args.gui,  # seed=args.seed,
                                     add_noise=args.add_noise,
                                     save_dir=self.save_dir, save_pkl=True)
        if args.record_video:
            self.sim.world.log_renderer.enable()  # 开启离线渲染

        while self.sim.num_objects == 0:  # 注意场景可能会生成失败, 例如物体靠在 box 上, 移去之后落地
            self.sim.world.log_renderer.reset()
            self.sim.reset(object_count, rng=rng)
            self.object_count = self.sim.num_objects  # 场景内物体数量
            print(f"Resetting simulation with {self.object_count} objects")

    def get_tsdf(self):
        '''获取指定视角的 depth 输入, 融合到 TSDF 中
        '''
        if self.mode == 'dataset':
            # 没有点云数据
            return self.tsdf.unsqueeze(0), None

        if self.mode == 'sim':
            depth_img, self.segmentation = self.sim.camera.render(self.extrinsic)[1:3]
            self.depth_img = apply_noise(depth_img, self.add_noise)
            # self.depth_img = depth_img  # w/o noise
            self.depth_imgs.append(depth_img)

        elif self.mode == 'real':
            # 从真实传感器中获取深度图
            _, color_image, depth_img, _ = self.franka.camera.get_sensor_info()
            self.color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            self.depth_img = depth_img.astype(np.float32)
            self.depth_imgs.append(self.depth_img)

        # 融合新视角 生成 TSDF
        self.tsdf.integrate(self.depth_img, self.intrinsic, self.extrinsic)  # 往空白 TSDF 上加入深度图所获得的数据
        pc = self.tsdf.get_cloud()
        # bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.sim.lower, self.sim.upper)
        # pc = pc.crop(bounding_box)
        return self.tsdf.get_grid(), pc

    def get_target_pos(self, pixel: Union[Tuple, List, None] = None, any_point: bool = False,
                       extrinsic=None) -> Tuple[List, List]:
        '''sim: 给定像素点, 获取目标物体的 mask, 随机选点, 转到世界坐标系; dataset: 读取pos, 转到 pixel; real: 给定 bbox, 转到 pixel
        any_point: 允许任意物体(仅仿真模式)
        extrinsic: sim 模式下, 使用 extrinsic 进行深度扩展
        '''
        # 筛选
        if self.mode == 'sim':
            if any_point:
                target_mask = self.segmentation > 0
            else:
                if self.target_label is None:
                    if pixel is None:
                        while True:
                            self.target_label = np.random.randint(1, self.object_count + 1)
                            target_mask = self.segmentation == self.target_label
                            if np.sum(target_mask) > 0:  # 可能会随机到被遮挡物体
                                break
                        # self.target_label = 4  # FOR DEBUG
                    else:
                        self.target_label = self.segmentation[pixel[1], pixel[0]]
                target_mask = self.segmentation == self.target_label

            pixel_list = np.argwhere(target_mask)
            # 随机选部分点分 (参数控制)
            if len(pixel_list) > self.args.sample_points_per_view:
                pixel_list = pixel_list[np.random.choice(len(pixel_list), self.args.sample_points_per_view, replace=False)]

            # pixel -> pos
            pos_list = []
            for y, x in pixel_list:
                z = self.depth_img[y, x]  # 深度图上的深度
                # 像素坐标转世界坐标
                pos = pixel2world(x, y, z, self.intrinsic, self.extrinsic)
                pos_list.append(pos)

            return self._expand_grasp_depth(
                pos_list, list(pixel_list), extrinsic=extrinsic if extrinsic is not None else self.extrinsic)

        elif self.mode == 'dataset':
            # 数据集
            pos_len = len(self.df.index)
            pos_list = []
            pixel_list = []
            # 从 pos_len 中随机选取 num_pts 个点
            num_pts = self.args.sample_points_per_view if self.args.sample_points_per_view < pos_len else pos_len
            for i in np.random.choice(pos_len, num_pts, replace=False):
                pos = self.df.loc[i, "x":"z"].to_numpy(np.single)
                pos_list.append(pos)
                u, v = world2pixel(*pos, self.intrinsic, self.extrinsic)
                pixel_list.append([v, u])
            return pos_list, pixel_list

        elif self.mode == 'real':
            pos_list = []
            pixel_list = []

            # 给定物体 bbox, 随机选取 bbox 内的点
            num_sample_axis = 10
            x_range = np.linspace(self.t_bbox.min[0], self.t_bbox.max[0], num_sample_axis)
            y_range = np.linspace(self.t_bbox.min[1], self.t_bbox.max[1], num_sample_axis)
            z_range = np.linspace(self.t_bbox.min[2], self.t_bbox.max[2], num_sample_axis)
            for point in itertools.product(x_range, y_range, z_range):
                pos_list.append(point)
                y, x = world2pixel(*point, self.intrinsic, self.extrinsic)
                pixel_list.append([x, y])

            # 从 pos 中随机选取 num_pts 个点
            if len(pos_list) > self.args.sample_points_per_view:
                sample_index = np.random.choice(len(pos_list), self.args.sample_points_per_view, replace=False)
                pos_list = list(np.array(pos_list)[sample_index])
                pixel_list = list(np.array(pixel_list)[sample_index])
            return pos_list, pixel_list

    def _expand_grasp_depth(self, pos_list: List, pixel_list: List,
                           extrinsic: Transform,
                           finger_depth=0.05, depth_candidates=5) -> Tuple[List, List]:
        '''叠加抓取深度
        finger_depth: 默认值在 sim.gripper.finger_depth
        '''
        camera_M = Transform(extrinsic.rotation,
                             np.array([0, 0, 1])).as_matrix()  # 旋转到光轴方向 然后沿着 z 前进 1 (不用归一化了)
        direction_vector = np.linalg.inv(camera_M)[:3, 3]  # 前进之后在世界坐标下的位置 or 方向向量

        # 范围 [-0.1, 1.1] 倍的爪深, 取 depth_candidates 个
        pos_list_with_depth = []
        eps = 0.1
        if len(pos_list) == 0:
            return pos_list_with_depth, pixel_list
        for depth in np.linspace(-eps * finger_depth, (1.0 + eps) * finger_depth, depth_candidates):
            pos_list_with_depth.extend(pos_list + direction_vector * depth)

        return pos_list_with_depth, pixel_list * depth_candidates  # pixel_list 直接复制以对应

    def excute_grasp(self, prediction, pos_list):
        '''合成抓取姿态, 驱动机械臂执行抓取
        '''
        index = prediction.grasp_label.argmax()
        ori = Rotation.from_quat(prediction.grasp_rotation[index].cpu())
        candidate = Grasp(Transform(ori, pos_list[index]),
                          width=prediction.grasp_width[index])  # 生成一个候选抓取
        print(f"Try grasp: {candidate.pose.to_dict()}, width: {candidate.width}")
        if self.mode == 'sim':
            outcome, width = self.sim.execute_grasp(candidate, remove=False)  # 执行抓取
            self.sim.world.log_renderer.export_video()
        elif self.mode == 'real':
            breakpoint()
            target_pose = self.T_base_task * candidate.pose  # gripper & task frame -> gripper & base frame
            self.franka.go_grasp(target_pose)
        elif self.mode == 'dataset':  # 不支持
            raise NotImplementedError
        return outcome

    def move_to_next_view(self, next_view):
        '''到下一视角
        '''
        if self.mode == 'sim':
            self.angles = next_view
            self.extrinsic = _angles_to_extrinsic(self.angles)
        elif self.mode == 'real':
            # 驱动机械臂移动到下一视角
            self.angles = next_view
            self.extrinsic = _angles_to_extrinsic(self.angles)  # task & cam frame

            rotated_extrinsic = Transform(Rotation.from_euler("z", np.pi / 2)  * self.extrinsic.rotation,
                                            self.extrinsic.translation)  # rotate camera 90 degree
            
            T_base_task_offset = self.T_base_task*Transform.from_translation([0.3, 0, 0])  # strange offset
            self.franka.goto_extrinsic((rotated_extrinsic * T_base_task_offset.inverse()).inverse().as_matrix())  # base & cam frame

            time.sleep(1.0)  # 等待机械臂到位
            self.extrinsic = self.franka.get_extrinsic().inverse() * self.T_base_task  # 跟踪有误差

        elif self.mode == 'dataset':
            raise NotImplementedError



def visualize_prediction(prediction, pixel_list, pos_list, grasp_exp, depth_expanded=True, best_depth_only=True):
    '''用 opencv 显示 heatmap
    depth_expanded: prediction 中的抓取位置是否已经叠加了抓取深度, 即保持 n*pixel_list, n*pos_list 的结构
                    如果是的话 q 值会选取不同深度中最好的 (pixel_list, pos_list 需要是未经深度扩展的)
    best_depth_only: rw 图是否只显示最好的深度
    '''
    background = grasp_exp.color_image.copy() if hasattr(grasp_exp, 'color_image') else \
        np.expand_dims(grasp_exp.depth_img.copy()*255, -1).repeat(3, -1).astype(np.uint8)

    if len(pos_list) == 0:
        raw_img = background.copy()
        return raw_img, raw_img
    result = EasyDict()
    depth_candidates = int(len(prediction.grasp_label) / len(pixel_list))
    if depth_expanded:  # 选出不同深度中质量最高的
        result.grasp_label = torch.zeros(len(pixel_list))
        result.grasp_rotation = torch.zeros(len(pixel_list), 4)
        result.grasp_width = torch.zeros(len(pixel_list))
        for i in range(len(pixel_list)):
            q_list = [prediction.grasp_label[j * depth_candidates + i] for j in range(depth_candidates)]
            best_q_index = q_list.index(max(q_list))
            result.grasp_label[i] = q_list[best_q_index]
            result.grasp_rotation[i] = prediction.grasp_rotation[best_q_index * depth_candidates + i]
            result.grasp_width[i] = prediction.grasp_width[best_q_index * depth_candidates + i]
    else:
        assert depth_candidates == 1
        result = prediction.to('cpu')

    # 质量图
    prediction_map_q = background.copy()
    for index, (y, x) in enumerate(pixel_list):
        q = int((result.grasp_label[index] - 0.001) / 0.999 * 255)
        cv2.circle(prediction_map_q, (x, y), 2, (255 - q, 0, q), -1)  # 红色质量高 蓝色质量低

    # 角度、宽度图
    prediction_map_rw = background.copy()
    for index, (y, x) in enumerate(pixel_list):
        if result.grasp_label[index] > 0.94:  # 只画质量高的抓取
            if best_depth_only:
                # 抓手在图上画成线
                pos_center = Transform(Rotation.from_quat(result.grasp_rotation[index]),
                                       pos_list[index])
                pixel1, pixel2 = _cal_gripper_pixel(pos_center, result.grasp_width[index],
                                                   grasp_exp.intrinsic.K, grasp_exp.extrinsic)
                cv2.line(prediction_map_rw,
                         (int(pixel1[0]), int(pixel1[1])),
                         (int(pixel2[0]), int(pixel2[1])), (0, 255, 0), 1)

            else:
                for j in range(depth_candidates):
                    pos_center = Transform(Rotation.from_quat(prediction.grasp_rotation[j * depth_candidates + index]),
                                           pos_list[index])
                    pixel1, pixel2 = _cal_gripper_pixel(pos_center, prediction.grasp_width[j * depth_candidates + index],
                                                       grasp_exp.intrinsic.K, grasp_exp.extrinsic)
                    # 用 j_ratio 来表示不同深度, 用红蓝来表示抓取质量
                    q_color = int(prediction.grasp_label[j * depth_candidates + index] * 255)
                    j_ratio = (j+8) / (depth_candidates+10)
                    cv2.line(prediction_map_rw,
                             (int(pixel1[0]), int(pixel1[1])),
                             (int(pixel2[0]), int(pixel2[1])),
                             (int((255 - q_color) * j_ratio),
                              0,
                              int(q_color * j_ratio)),
                             1)
    return prediction_map_q, prediction_map_rw


def _cal_gripper_pixel(pos_center, grasp_width, intrinsic, extrinsic) -> Tuple[float, float]:
    '''pos_center: 抓取中心的世界坐标位置
    '''
    # 抓手两端的世界坐标位置
    gripper_offset1 = Transform(Rotation.identity(), [0,  grasp_width / 2, 0])
    gripper_offset2 = Transform(Rotation.identity(), [0, -grasp_width / 2, 0])
    pos1 = pos_center * gripper_offset1
    pos2 = pos_center * gripper_offset2
    # 世界坐标系 转相机坐标系
    intrinsic = np.hstack((intrinsic, np.zeros((3, 1))))  # 内参添加最后一列零
    pos1_cam = extrinsic * pos1
    pixel1 = (intrinsic @ pos1_cam.as_matrix())[:, 3]
    pixel1 /= pixel1[-1]
    pos2_cam = extrinsic * pos2
    pixel2 = (intrinsic @ pos2_cam.as_matrix())[:, 3]
    pixel2 /= pixel2[-1]
    return pixel1, pixel2


def check_all_angles(tsdf, pos, intrinsic, grasp_planner, size=0.3):
    '''检查某个 pos 在所有视角下的表现
    '''
    # 构造 batch
    step = 100  # resolution = step**2
    r = 2.0 * 0.3
    extrinsics = torch.zeros(step, step, 7)
    for i, theta in enumerate(np.linspace(1e-3, np.pi / 2.0 - 1e-3, step)):
        for j, phi in enumerate(np.linspace(1e-3, np.pi * 2 - 1e-3, step)):
            extrinsics[i, j] = torch.tensor(_angles_to_extrinsic((r, theta, phi)).to_list())
    extrinsics = extrinsics.reshape(step**2, 1, 7)

    pos = pos / args.size - 0.5
    pos = pos[None, None, :]

    tsdf = torch.tensor(tsdf)

    intrinsics = torch.tensor(intrinsic)
    intrinsics = intrinsics[None, :]

    # 分批处理
    batch_size = 200  # 一次预测的点数量 为了保显存
    result = None
    for i in tqdm(range(0, step**2, batch_size)):
        batch = EasyDict({'tsdf': tsdf.repeat(batch_size, 1, 1, 1),
                          'point_grasp': pos.repeat(batch_size, 1, 1),
                          'camera_extrinsic': extrinsics[i:i + batch_size],
                          'camera_intrinsic': intrinsics.repeat(batch_size, 1, 1)})
        prediction_batch = grasp_planner.predict_grasp(batch)
        if result is None:
            result = prediction_batch
        else:
            result.append(prediction_batch)

    q = ((result.grasp_label.reshape(step, step) - 0.001) / 0.999 * 255).int().cpu().numpy()
    prediction_map_q = np.zeros((step, step, 3), dtype=np.uint8)
    prediction_map_q[:, :, 0] = 255 - q  # 抓取质量越高 颜色越红
    prediction_map_q[:, :, 2] = q
    return prediction_map_q


def main(args, cfg):
    grasp_planner = GraspPlanner(cfg)

    # 实验保存路径
    save_dir_root = Path("experiments") / args.experiment_name / "grasp_results" / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if os.path.exists(save_dir_root):  # 删除旧的实验结果
        shutil.rmtree(save_dir_root)
    os.makedirs(save_dir_root, exist_ok=True)

    # 固定每次轮实验的种子
    if args.seed < 0:
        with open('seed_list.txt', 'r') as f:
            seeds_list = [int(seed) for seed in f.readlines()]
        assert len(seeds_list) >= args.num_rounds, 'seed_list.txt 中的种子数量不足'
    else:
        np.random.seed(args.seed)
        seeds_list = np.random.choice(2**20, args.num_rounds, replace=False)

    results = []
    for index in range(args.num_rounds):  # 实验轮数
        set_random_seed(seeds_list[index])  # 固定全局随机种子
        save_dir = save_dir_root / f'round_{index:04d}'
        os.makedirs(save_dir, exist_ok=True)
        print('\033[1;34m' + f'===== Round {index:04d} Start =====' + '\033[0m')
        print(f'Model: {args.experiment_name}  Seed: {seeds_list[index]}')
        grasp_exp = GraspExp(args, save_dir, rng=np.random.default_rng(seeds_list[index]))  # 实验管理器(创建场景、数据接口)

        eplot = ExtrinsicMonitor(grasp_exp.intrinsic_list)  # 可视化相机外参
        os.makedirs(save_dir / 'cameras', exist_ok=True)

        # loop
        success_flag = False
        grasp_count = 0
        look_count = 0
        while not success_flag:
            # 观测
            look_count += 1
            tsdf, pc = grasp_exp.get_tsdf()
            if pc:
                o3d.io.write_point_cloud(f'{save_dir}/pointcloud_{look_count}.pcd', pc)  # 保存点云
                if args.gui:
                    o3d.visualization.draw_geometries([pc])

            print('\033[1;35m' + f'Get depth image from view {grasp_exp.extrinsic.to_list()}.' + '\033[0m')
            # 可视化外参
            eplot.add_cam(grasp_exp.angles)
            eplot.draw_and_save(f'{save_dir}/cameras/{args.ckpt_index}_{look_count}_0.png')
            if args.gui and hasattr(grasp_exp, 'sim'):
                # gui 视角随动
                grasp_exp.sim.world.p.resetDebugVisualizerCamera(
                    cameraDistance=0.6,
                    cameraYaw=90 + grasp_exp.angles[2] / np.pi * 180,
                    cameraPitch=-90 + grasp_exp.angles[1] / np.pi * 180,
                    cameraTargetPosition=[0.15, 0.15, 0.0],
                )
                eplot.show()

            # 分割
            pos_list, pixel_list = grasp_exp.get_target_pos(any_point=args.grasp_any_object)

            # 预测抓取
            prediction = grasp_planner(tsdf, pos_list,
                                       grasp_exp.extrinsic.to_list(),
                                       grasp_exp.intrinsic_list,
                                       args.size)
            print(f'Evaluate {len(pos_list)} grasp positions from current view, '
                  f'and the best possible grasp quality is {prediction.grasp_label.max():.4f}.')

            map_q, map_rw = visualize_prediction(prediction, pixel_list, pos_list, grasp_exp, depth_expanded=grasp_exp.args.mode in ['sim', 'dataset'])
            cv2.imwrite(f'{save_dir}/prediction_map_q_{args.ckpt_index}_{look_count}.png', map_q)
            cv2.imwrite(f'{save_dir}/prediction_map_rw_{args.ckpt_index}_{look_count}.png', map_rw)

            if args.gui:
                cv2.imshow('prediction_map_q', map_q)
                cv2.imshow('prediction_map_rw', map_rw)
                cv2.waitKey(0)
            # breakpoint()

            # 判断抓取质量 决定执行抓取或者预测下一视角
            if prediction.grasp_label.max() < args.grasp_q_thresh and look_count <= args.max_look_time:
                # 预测下一视角
                print('No good grasp found. Start to sample new view.')
                angles_candidate = _generate_extrinsic_neighbors(grasp_exp.extrinsic,
                                                                 num_direction=args.new_view_directions,
                                                                 num_steps=args.new_view_step)
                print(f'Sampled {len(angles_candidate)} new view angles.')

                # 可视化相机
                eplot.add_candidates(angles_candidate)
                eplot.draw_and_save(f'{save_dir}/cameras/{args.ckpt_index}_{look_count}_1.png')
                if args.gui:
                    eplot.show()

                # 评估新视角的抓取质量
                next_view = None
                candidate_q = []
                for angles in angles_candidate:
                    pos_list, pixel_list = grasp_exp.get_target_pos(any_point=args.grasp_any_object, extrinsic=_angles_to_extrinsic(angles))

                    prediction = grasp_planner(tsdf, pos_list,
                                               _angles_to_extrinsic(angles).to_list(),
                                               grasp_exp.intrinsic_list,
                                               args.size)
                    candidate_q.append(prediction.grasp_label.max().item())

                # 可视化相机
                eplot.add_candidatas_with_quality(angles_candidate, candidate_q)
                eplot.draw_and_save(f'{save_dir}/cameras/{args.ckpt_index}_{look_count}_2.png')
                eplot.clear_candidates()
                if args.gui:
                    eplot.show()

                best_q = max(candidate_q)
                next_view = angles_candidate[candidate_q.index(best_q)]
                print(f'Next best predicted grasp quality is {best_q:.4f} in new view {next_view}.')
                grasp_exp.move_to_next_view(next_view)
                # grasp_exp.move_to_next_view([0.6, 0.4, 3.1])  # test
                print(f'Move to next view {next_view}.')

            else:
                # 执行抓取
                print('\033[1;33m' + 'Start to execute grasp.' + '\033[0m')
                grasp_count += 1
                success_flag = grasp_exp.excute_grasp(prediction, pos_list)
                if success_flag:  # 抓取成功 直接结束本次场景
                    print('\033[1;32m' + 'Grasp success!' + '\033[0m')
                    break
                else:  # 抓取不成功
                    grasp_exp.clear_depth_imgs()
                    print('\033[1;31m' + 'Grasp failed!' + '\033[0m' + ' Clear previous observation.')
                    # 如果允许多次尝试
                    # grasp_exp.clear_depth_imgs()
                    # ↓ 目前是只允许一次尝试
                    if grasp_count >= args.max_grasp_time:
                        # 达到最大尝试次数 直接结束本次场景
                        print('\033[1;31m' + 'Tired. Reach max trail count. Stop this scene.' + '\033[0m')
                        break

        print(f'Round {index} finished, looked {look_count} times, tried {grasp_count} times and',
              'succeed' if success_flag else 'failed')
        results.append([success_flag, look_count, grasp_count])
    return results, save_dir_root


def analyze_results(results, save_dir_root):
    '''统计结果
    Grasp Success Rate: success / cnt
    Average Look Count: look_count / cnt
    Average Trail Count: grasp_count / cnt, if success
    '''
    print('\033[1;34m' + f'===== Result Analysis =====' + '\033[0m')
    cnt = len(results)
    success_cnt = sum([1 for success_flag, _, _ in results if success_flag])
    GSR = success_cnt / cnt
    ALC = sum([look_count for _, look_count, _ in results]) / cnt
    if success_cnt:
        ATC = sum([grasp_count for success_flag, _, grasp_count in results if success_flag]) / success_cnt
    else:
        ATC = np.nan
    print(f'GSR: {GSR:.4f}, ALC: {ALC:.4f}, ATC: {ATC:.4f}')

    with open(save_dir_root / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saving results to {save_dir_root / "results.json"}')


def load_hydra_cfg(args):
    '''load configs from experiment log, and override ckpt_name, device, etc.
    '''
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None,
               config_path="experiments/" + args.experiment_name + "/configs",
               job_name="test_app")
    ckpt_name = get_ckpts(args.experiment_name)[args.ckpt_index]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cfg = compose(config_name="config", overrides=[f"load_path='{ckpt_name}'",
                                                   f"device={device}"])
    return cfg


def get_ckpts(experiment, only_epoch_end=False):
    '''获取某个实验的所有 ckpt list
    only_epoch_end: 只用每个 epoch end 保存的 ckpt
    '''
    ckpts_path = 'experiments/' + experiment + '/ckpts/'
    ckpts = [f for f in os.listdir(ckpts_path)
             if not only_epoch_end or f.endswith('end.pt')]
    # ckpts.sort(key=lambda f: os.path.getmtime(os.path.join(ckpts_path, f)))
    ckpts.sort()
    return [os.path.join(ckpts_path, f) for f in ckpts]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--gui", action="store_true")
    # model
    parser.add_argument("experiment_name", type=str)
    parser.add_argument("--ckpt-index", type=int, default=-1)
    # experiment
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--mode", type=str, choices=["sim", "real", "dataset"], default="sim")
    parser.add_argument("--size", type=float, default=0.3)
    # simulation
    parser.add_argument("--load-scene", type=str,
                        default="", help="加载已有仿真场景")
    parser.add_argument("--scene-type", type=str, choices=["pile", "packed"], default="packed")
    parser.add_argument("--object-set", type=str, default="packed/test",
                        help="blocks or packed/test or pile/test")
    parser.add_argument("--num-objects", type=int, default=4)
    parser.add_argument("--add-noise", type=str, default='dex',
                        help="Whether add noise to depth observation, trans | dex | norm | ''")
    parser.add_argument("--record-video", action="store_true")
    # grasp
    parser.add_argument("--grasp-any-object", action="store_true")
    parser.add_argument("--sample-points-per-view", type=int, default=200)
    parser.add_argument("--grasp-q-thresh", type=float, default=0.92)
    parser.add_argument("--new-view-directions", type=int, default=6,
                        help="每次看完之后, 选取多少个新视角方向")
    parser.add_argument("--new-view-step", type=int, default=3,
                        help="新视角方向上[0-90]分几个步")
    parser.add_argument("--max-grasp-time", type=int, default=1)
    parser.add_argument("--max-look-time", type=int, default=6)

    args = parser.parse_args()
    if args.silent:
        print = lambda *args, **kwargs: None
    cfg = load_hydra_cfg(args)

    results, save_dir_root = main(args, cfg)

    if args.silent:
        del print

    analyze_results(results, save_dir_root)
