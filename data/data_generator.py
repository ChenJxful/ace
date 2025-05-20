# Abstract: Save configuration to setup.json;
#           Generate random scene, save mesh_pose_list to mesh_pose_list/[scene_id].npz
#           Render depth image and record extrinsic to depth_imgs/[scene_id].npz;
#           depth images -> tsdf, save to voxel_grid/[scene_id].npz;
#           Test random grasp(direction bound with view) and save to grasps.csv;
#           Clean and balance.
# Reference: GIGA/scripts/generate_data_parallel.py

import cv2
import uuid
import argparse
import numpy as np
import multiprocessing as mp
import scipy.signal as signal
from tqdm import tqdm
from pathlib import Path
try:
    import open3d as o3d
except ImportError:
    print("Running without Open3D.")

from vgn.io import my_IO
from vgn.perception import *
from vgn.grasp import Grasp, Label
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_mesh_pose_list_from_world
from vgn.utils.misc import apply_noise
from utils.misc import set_random_seed
from utils.transform import *


def generate_scene_and_grasp(args, rank):
    """
    args: arguments from the command line
    rank: index for multiprocessing
    """
    seed = np.random.randint(0, 1000) + rank  # 随机数种子，每个进程不同
    np.random.seed(seed)
    # 基于 PyBullet, 创建抓取场景
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui,
                            log_render=True, save_dir=args.root)
    args.size = sim.size  # 默认 0.3
    if args.record_video:
        default_extrinsic = get_default_extrinsic(args.size)  # 默认相机外参
        sim.world.log_renderer.reset(default_extrinsic)  # 用于 log_renderer 的外参
        sim.world.log_renderer.enable()

    grasps_per_worker = args.num_grasps // args.num_proc  # 分配每个进程的工作量
    pbar = tqdm(total=grasps_per_worker, disable=rank != 0)  # 只显示第一个进程的进度条

    myio = my_IO(args.root)  # 初始化 IO, 不支持写入OSS存储
    if rank == 0:
        myio.write_setup(  # 写入 setup.json
            sim.size,  # 默认 0.3
            sim.camera.intrinsic,
            sim.gripper.max_opening_width,  # 默认 0.08
            sim.gripper.finger_depth,  # 默认 0.05
        )
        if args.save_scene:
            (args.root / "mesh_pose_list").mkdir(parents=True, exist_ok=True)

    # 创建场景(多个进程分)
    for scene_idx in range(grasps_per_worker // args.grasps_per_scene):
        scene_id = uuid.uuid4().hex
        object_count = np.random.poisson(args.object_count_lambda) + 1  # 按照泊松分布生成场景中物体数量
        sim.reset(object_count)  # 随机生成物体堆
        sim.save_state()  # 保存场景到 sim._snapshot_id

        # 随机采集一系列 depth image
        depth_imgs, _, segmentation_masks, extrinsics = render_images(sim, viewpoint_count=12)

        # 创建 tsdf 作为观测, 保存到文件(注意这里添加噪声)
        if args.observation_type == "facing":  # 提供正面观测 预测正面抓取、正面深度图
            tsdf = generate_tsdf_from_depth(args.size, args.grid_resolution,
                                            depth_imgs[[0]], sim.camera.intrinsic,
                                            extrinsics[[0]], add_noise='dex')
        elif args.observation_type == "side":  # 提供侧面观测 预测正面抓取、正面深度图
            assert len(depth_imgs) >= 2, "Need at least 2 depth images for side observation"
            tsdf = generate_tsdf_from_depth(args.size, args.grid_resolution,
                                            depth_imgs[[1]], sim.camera.intrinsic,
                                            extrinsics[[1]], add_noise='dex')
        elif args.observation_type == "multiview":  # 提供多视角观测 预测正面抓取、正面深度图
            tsdf = generate_tsdf_from_depth(args.size, args.grid_resolution,
                                            depth_imgs, sim.camera.intrinsic,
                                            extrinsics, add_noise='dex')
        else:
            raise ValueError("Invalid observation type")
        pc = tsdf.get_cloud()
        if args.save_pointcloud:  # 保存点云 仅用于 debug
            (args.root / "scenes").mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(args.root / "scenes" / ("pc_scene_{:03d}".format(scene_idx) + ".pcd")), pc)  # 保存点云
        if args.sim_gui:
            o3d.visualization.draw_geometries([pc])
        if pc.is_empty():  # 如果点云为空, 跳过该场景
            print("Point cloud empty, skipping scene")
            continue
        grid = tsdf.get_grid()
        myio.write_voxel_grid(scene_id, grid)  # 保存到 voxel_grid/[scene_id].npz
        myio.write_depth_image(scene_id, depth_imgs[[0]], extrinsics[[0]])  # 保存正面深度图和外参

        # 保存物体形状, 大小, 位置到 mesh_pose_list/[scene_id].npz
        if args.save_scene:
            mesh_pose_list = get_mesh_pose_list_from_world(sim.world, args.object_set)
            myio.write_mesh_pose_list(scene_id, mesh_pose_list, name="mesh_pose_list")

        # 创建精确点云用于后续抓取点法向判断
        tsdf = generate_tsdf_from_depth(args.size, 120,
                                        depth_imgs, sim.camera.intrinsic,
                                        extrinsics)  # 生成 TSDF
        pc = tsdf.get_cloud()
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)  # default 是一个 26cm*26cm*24.5cm 的长方体
        pc = pc.crop(bounding_box)  # crop surface and borders from point cloud
        # o3d.io.write_point_cloud(str(args.root / "scenes" / ("pc_scene_{:03d}".format(scene_idx) + ".pcd")), pc)  # 保存点云

        # 尝试抓取
        depth = depth_imgs[0]
        extrinsic = extrinsics[0]
        extrinsic_matrix = Transform.from_list(list(extrinsic)).as_matrix()
        camera_M = Transform(Rotation.from_quat(list(extrinsic)[:4]),
                             np.array([0, 0, 1])).as_matrix()  # 旋转到光轴方向 然后沿着 z 前进 1
        direction_vector = np.linalg.inv(camera_M)[:3, 3]  # 前进之后在世界坐标下的位置 or 方向向量
        segmentation_mask = segmentation_masks[0]  # [480, 640] ndarray
        valid_point = np.nonzero(segmentation_mask)  # tuple of ndarray, [y1, y2, ...], [x1, x2, ...]

        if args.with_heatmap:  # 维护一张抓取 heatmap, 转 RGB 图像
            visulizer = GraspVisualizer(depth_imgs[0], sim.camera.intrinsic.K, extrinsic_matrix, camera_M)

        # 在 mask 上随机选点
        for _ in range(args.grasps_per_scene):
            sim.world.log_renderer.reset()

            point, pixel = sample_grasp_point_from_depth(depth, valid_point,
                                                         sim.camera.intrinsic.K, extrinsic_matrix)

            # 随机抓取深度，[-0.1, 1.1] 倍的爪深
            eps = 0.1
            grasp_depth = np.random.uniform(-eps * sim.gripper.finger_depth, (1.0 + eps) * sim.gripper.finger_depth)
            grasp_point = point + direction_vector * grasp_depth

            label, yaw_from_view, width = evaluate_grasp_point(sim, grasp_point, direction_vector, num_rotations=args.rotation_per_pos)

            if args.with_heatmap:
                visulizer.add_grasp(pixel, point, direction_vector, yaw_from_view, label, width)

            # 保存当前场景的抓取(id, 点3, 角度1, 宽度1, 标签1)
            myio.write_grasp_yaw(scene_id, grasp_point, yaw_from_view, width, label)
            if label != 0:
                sim.world.log_renderer.export_video()
            pbar.update()

        if args.with_heatmap:
            visulizer.save(scene_id)

    pbar.close()
    print('Process %d finished!' % rank)


def get_default_extrinsic(size):
    '''一个默认相机外参
    '''
    origin = Transform(Rotation.identity(), np.r_[size / 2, size / 2, size / 3])
    r = 2 * size
    theta = np.pi / 3.0
    phi = - np.pi / 2.0
    return camera_on_sphere(origin, r, theta, phi)


def get_random_extrinsic(size=0.3,
                         origin=Transform(Rotation.identity(), np.r_[0.3 / 2, 0.3 / 2, 0.0])):
    '''生成指向origin的随机相机外参
    '''
    r = np.random.uniform(1.6, 2.4) * size
    theta = np.random.uniform(0.0, np.pi / 3.0)  # default pi/4
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    return camera_on_sphere(origin, r, theta, phi)  # 生成从球面上点看向中心的相机外参


def render_images(sim, viewpoint_count=6):
    """
    Args:
        给定场景 sim (其中包括相机内参)
        viewpoint_count 个随机视角 (球面上)
    Returns:
        viewpoint_count * 渲染深度图 (480*640的一维图, 应该是米为单位)
        viewpoint_count * 相机外参 (7维list)
    """
    # 仿真环境中相机内参
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((viewpoint_count, 7), np.float32)  # 注意没有初始化 随机数值
    depth_imgs = np.empty((viewpoint_count, height, width), np.float32)
    rgb_imgs = np.empty((viewpoint_count, height, width, 3), np.float32)
    segmentation_masks = np.empty((viewpoint_count, height, width), np.float32)
    for i in range(viewpoint_count):  # 均匀分布生成 n 个视角，对应论文公式 12 附近
        extrinsic = get_random_extrinsic(sim.size, origin)  # 生成从球面上点看向中心的相机外参
        rgb_img, depth_img, segmentation_mask = sim.camera.render(extrinsic)  # 渲染深度图
        rgb_img = np.array(rgb_img)

        extrinsics[i] = extrinsic.to_list()
        rgb_imgs[i] = rgb_img
        depth_imgs[i] = depth_img
        segmentation_masks[i] = segmentation_mask

    return depth_imgs, rgb_imgs, segmentation_masks, extrinsics


def generate_tsdf_from_depth(size, grid_resolution, depth_imgs, intrinsic, extrinsics, add_noise=''):
    '''合成多个视角的深度图，合成 TSDF (叠加指定噪声)
    '''
    assert add_noise in ['', 'dex', 'trans', 'norm']
    assert len(depth_imgs) == len(extrinsics)
    depth_imgs = np.array([apply_noise(x, add_noise) for x in depth_imgs])
    tsdf = create_tsdf(size, grid_resolution, depth_imgs, intrinsic, extrinsics)
    return tsdf


def sample_grasp_point_from_pointcloud(point_cloud, intrinsic, extrinsic):
    """从点云中随机采样一个抓取点 (点云的法向作为抓取法向)
    """
    points = np.asarray(point_cloud.points)  # 点云上的点 [n*3]
    normals = np.asarray(point_cloud.normals)  # 点云上的法向 [n*3]
    ok = False
    # 随机采样一个法向朝上的抓取表面
    while not ok:
        # TODO this could result in an infinite loop, though very unlikely
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]
        # ok = normal[2] > -0.1  # make sure the normal is pointing upwards
        ok = True
    # 世界坐标转像素坐标
    pixel = world2pixel(*point, intrinsic, extrinsic)
    return point, pixel, normal


def sample_grasp_point_from_depth(depth, valid_point,
                                  intrinsic, extrinsic):
    """沿着相机方向, 在mask中随机采样一个像素点, 转到世界坐标, 作为抓取点(叠加了一个爪深位移)
    """
    index = np.random.randint(len(valid_point[0]))
    x, y = valid_point[1][index], valid_point[0][index]  # 像素坐标系
    z = depth[y, x]  # 深度图上的深度

    # 像素坐标转世界坐标
    pos = pixel2world(x, y, z, intrinsic, extrinsic)
    return pos, (x, y)


def evaluate_grasp_point(sim, pos, normal, num_rotations=6):
    '''测试抓取结果
    Args:
        sim: 环境
        pos: 抓取点
        normal: 抓取点的法向
        num_rotations: 旋转采样次数
    Returns:
        成功的抓取姿态(经过滤波), 成功与否标签, 抓取姿态的 yaw(仅用于训练)
    '''
    R = normal2RM(normal)  # 法向 → 旋转矩阵 (人工固定一个额外的 x 轴)

    # 以不同角度尝试抓取  try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations, endpoint=False)  # 叠加 6 个不同的 yaw（等间隔）
    yaws += np.random.uniform(0.0, np.pi / num_rotations)  # 整体随机偏移
    outcomes, widths = [], []

    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()  # 从 self._snapshot_id 恢复初始状态
        candidate = Grasp(Transform(ori, pos), width=sim.gripper.max_opening_width)  # 生成一个候选抓取
        outcome, width = sim.execute_grasp(candidate, remove=False)  # 执行抓取
        outcomes.append(outcome)
        widths.append(width)

    # 如果有成功的抓取，则寻找最宽的角度范围  detect mid-point of widest peak of successful yaw angles
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        peaks, properties = signal.find_peaks(  # TODO 没有考虑角度的周期性
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]
        return int(np.max(outcomes)), yaws[idx_of_widest_peak], width
    else:
        return int(np.max(outcomes)), yaw, width  # 失败情况随便给一个角度/宽度


def normal2RM(normal: np.ndarray) -> Rotation:
    '''法向 → 旋转矩阵 (固定一个额外的 x 轴)
    注意 数据生成、后续抓取验证 yaw 转四元数 要遵循同样的写法
    Args:
        normal: [3] 法向
    '''
    # 根据抓取法向建立坐标轴 (给定 z, 随便来个正交 xy)  define initial grasp frame on object surface
    z_axis = -normal  # z 轴：点云法向的反向（朝物体内）
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):  # 判断 xz 内积是否接近 1（差值小于 1e-4）
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)  # 叉乘
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)  # 从法向向量生成旋转矩阵
    return R


class GraspVisualizer():
    def __init__(self, depth_img, intrinsic, extrinsic, camera_M):
        self.depth_img = depth_img
        self.grasp_map_q = np.expand_dims(depth_img * 255, -1).repeat(3, -1).astype(np.uint8)
        self.grasp_map_rw = np.expand_dims(depth_img * 255, -1).repeat(3, -1).astype(np.uint8)
        self.normal_map = np.expand_dims(depth_img * 255, -1).repeat(3, -1).astype(np.uint8)
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.camera_M = camera_M

    def add_grasp(self, pixel, point, direction_vector, yaw_from_view, label, width):
        # === q ===
        color = (0, 0, 255) if label == 1 else (255, 0, 0)  # 红色能抓 蓝色不能抓
        cv2.circle(self.grasp_map_q, (pixel[0], pixel[1]), 2, color, -1)
        cv2.circle(self.grasp_map_rw, (pixel[0], pixel[1]), 1, color, -1)
        # === rw ===
        if label == 1:  # 只能是视角绑定的情况下可用
            R = normal2RM(direction_vector)
            ori = R * Rotation.from_euler("z", yaw_from_view)
            # 抓手两端的世界坐标位置
            pos_center = Transform(ori, point)

            gripper_offset1 = Transform(Rotation.identity(), [0, width / 2, 0])
            gripper_offset2 = Transform(Rotation.identity(), [0, -width / 2, 0])
            pos1 = pos_center * gripper_offset1
            pos2 = pos_center * gripper_offset2
            # 世界坐标系 转相机坐标系
            intrinsic = np.hstack((self.intrinsic, np.zeros((3, 1))))  # 内参添加最后一列零
            extrinsic = Transform.from_matrix(self.extrinsic)
            pos1_cam = extrinsic * pos1
            pixel1 = (intrinsic @ pos1_cam.as_matrix())[:, 3]
            pixel1 /= pixel1[-1]
            pos2_cam = extrinsic * pos2
            pixel2 = (intrinsic @ pos2_cam.as_matrix())[:, 3]
            pixel2 /= pixel2[-1]
            # 在图上画成线
            cv2.line(self.grasp_map_rw,
                     (int(pixel1[0]), int(pixel1[1])),
                     (int(pixel2[0]), int(pixel2[1])), (0, 255, 0), 1)
        # === normal ===
        # normal_color = tuple(map(lambda x: int(x * 127 + 127), direction_vector))
        normal_color = tuple(map(lambda x: int(x * 127 + 127), self.camera_M[:3, :3] @ direction_vector))  # optional 世界坐标转相机坐标
        cv2.circle(self.normal_map, (pixel[0], pixel[1]), 4, normal_color, -1)

    def save(self, scene_id):
        path = args.root / "heat_map"
        path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f'{path}/{scene_id}_grasp_q.png', self.grasp_map_q)
        cv2.imwrite(f'{path}/{scene_id}_grasp_rw.png', self.grasp_map_rw)
        cv2.imwrite(f'{path}/{scene_id}_depth.png', self.depth_img * 255)
        cv2.imwrite(f'{path}/{scene_id}_normal.png', self.normal_map)


def clean_balance_data(root):
    '''筛选合法抓取位置, 平衡正负样本
    '''
    myio = my_IO(args.root)
    df = myio.read_df()  # grasp.csv
    positives = df[df["label"] == 1]  # 筛选出对应标签的行
    negatives = df[df["label"] == 0]

    print("Before clean and balance:")
    print("Number of samples:", len(df.index))
    print("Number of positives:", len(positives.index))
    print("Number of negatives:", len(negatives.index))

    df.drop(df[df["x"] < 0.02].index, inplace=True)
    df.drop(df[df["y"] < 0.02].index, inplace=True)
    df.drop(df[df["z"] < 0.02].index, inplace=True)
    df.drop(df[df["x"] > 0.28].index, inplace=True)
    df.drop(df[df["y"] > 0.28].index, inplace=True)
    df.drop(df[df["z"] > 0.28].index, inplace=True)

    # balance  丢弃部分负样本使得正负样本数量相等
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]
    i = np.random.choice(negatives.index, len(negatives.index) - len(positives.index), replace=False)
    df = df.drop(i)
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]
    myio.write_df(df)

    print("After clean and balance:")
    print("Number of samples:", len(df.index))
    print("Number of positives:", len(positives.index))
    print("Number of negatives:", len(negatives.index))

    # remove unreferenced scenes.  寻找不再保留有效抓取的场景
    # suffix 扩展名，stem 文件名
    grasp_scene = df["scene_id"].values
    for f in (root / "depth_imgs").iterdir():
        if f.suffix == ".npz" and f.stem not in grasp_scene:
            print("Removed scene", f.stem)
            f.unlink()  # 删除文件
            if (args.root / "mesh_pose_list").is_dir():
                (root / "mesh_pose_list").joinpath(f.stem + ".npz").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, default=Path("./data/pile/data_pile_bind_minimal"))

    # scene
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="packed")
    parser.add_argument("--object-count-lambda", type=int, default=4)
    parser.add_argument("--object-set", type=str, default="packed/train",
                        help="物品数据来源")
    parser.add_argument("--observation-type", type=str, choices=["facing", "side", 'multiview'], default="facing",
                        help="观测视角类型: facing 与抓取同方向, side 与抓取方向无关, multiview 多视角综合观测")
    parser.add_argument("--grid-resolution", type=int, default=40)

    # grasp
    parser.add_argument("--num-grasps", type=int, default=4000000)
    parser.add_argument("--grasps-per-scene", type=int, default=240)  # 120
    parser.add_argument("--rotation-per-pos", type=int, default=12)

    # misc
    parser.add_argument("--seed", type=int, default=42,
                        help="固定全局随机种子")
    parser.add_argument("--num-proc", type=int, default=1)  # support multi-process
    parser.add_argument("--save-scene", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--with-heatmap", action="store_true")
    parser.add_argument("--save-pointcloud", action="store_true")
    args = parser.parse_args()

    # rewrite args
    # args.save_scene = True
    # args.num_grasps = 200
    # args.grasps_per_scene = 200

    # check args
    assert args.num_grasps // args.num_proc >= args.grasps_per_scene, 'num_grasps should be larger than grasps_per_scene'

    # remove DEBUG folder
    if args.root.exists():
        breakpoint()  # double check
        import shutil
        shutil.rmtree(args.root)

    # start generating
    print('====== Summary ======')
    print(f'Generate data with {args.object_set} object in {args.scene} scene.')
    print(f'{args.num_grasps} grasps, {args.grasps_per_scene} grasps per scene, {args.num_proc} workers in total.')
    print(f'Output to {args.root}.')
    print('=====================')
    set_random_seed(args.seed)  # 固定全局随机种子

    if args.num_proc > 1:
        pool = mp.Pool(processes=args.num_proc)
        for i in range(args.num_proc):
            pool.apply_async(func=generate_scene_and_grasp, args=(args, i))
        pool.close()
        pool.join()
    else:
        generate_scene_and_grasp(args, 0)

    print('=====================')
    clean_balance_data(args.root)
