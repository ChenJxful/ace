import time

import numpy as np
import trimesh
from scipy import ndimage
import torch

#from vgn import vis
from vgn.grasp import *
from vgn.utils.transform import Transform, Rotation
# from vgn.networks import load_network
# from models.networks import load_network
try:
    from vgn.utils import visual
except ImportError:
    print("Running without vgn.utils.visual")
from vgn.utils.implicit import as_mesh

from utils.inspection import PrepareNet
from utils.misc import EasyDict

LOW_TH = 0.5


class VGNImplicit(object):
    def __init__(self, experiment_name, ckpt_index, best=False, force_detection=False, qual_th=0.9, out_th=0.5, visualize=False, resolution=40, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = PrepareNet(experiment_name, ckpt_index)._get_net()

        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        self.visualize = visualize

        self.resolution = resolution
        x, y, z = torch.meshgrid(torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), 
                                 torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), 
                                 torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution))
        # 1, self.resolution, self.resolution, self.resolution, 3
        pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).to(self.device)
        self.pos = pos.view(1, self.resolution * self.resolution * self.resolution, 3)

    def __call__(self, state, scene_mesh=None, aff_kwargs={}):
        '''输入 TSDF, 输出一个(多个)抓取、计算时间
        Args:
            state: 主要包含 pc 和 tsdf
        '''
        if hasattr(state, 'tsdf_process'):
            tsdf_process = state.tsdf_process
        else:
            tsdf_process = state.tsdf

        if isinstance(state.tsdf, np.ndarray):
            tsdf_vol = state.tsdf
            voxel_size = 0.3 / self.resolution
            size = 0.3
        else:
            tsdf_vol = state.tsdf.get_grid()
            voxel_size = tsdf_process.voxel_size
            tsdf_process = tsdf_process.get_grid()
            size = state.tsdf.size

        # 根据 TSDF, 网络判断整个空间中的抓取质量
        tic = time.time()
        qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.pos, self.net, self.device)
        qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        rot_vol = rot_vol.reshape((self.resolution, self.resolution, self.resolution, 4))
        width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))

        # 抓取质量平滑、过滤等后处理
        qual_vol, rot_vol, width_vol = process(tsdf_process, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        qual_vol = bound(qual_vol, voxel_size)

        if self.visualize:
            colored_scene_mesh = visual.affordance_visual(qual_vol, rot_vol, scene_mesh, size, self.resolution, **aff_kwargs)
        breakpoint()
        # 根据各种条件, 选择一个合适的抓取 (或多个抓取的 list)
        grasps, scores = select(qual_vol.copy(),
                                self.pos.view(self.resolution, self.resolution, self.resolution, 3).cpu(),
                                rot_vol, width_vol,
                                threshold=self.qual_th,
                                force_detection=self.force_detection,
                                max_filter_size=8 if self.visualize else 4)
        toc = time.time() - tic

        grasps, scores = np.asarray(grasps), np.asarray(scores)

        # 重新排序 (或者顺序)(因为后续只会执行第一个), 修正抓取的位置 (重新映射到合适范围)
        new_grasps = []
        if len(grasps) > 0:
            if self.best:
                p = np.arange(len(grasps))
            else:
                p = np.random.permutation(len(grasps))
            for g in grasps[p]:  # 注意这里还是一个 list 只不过索引顺序变了
                pose = g.pose
                pose.translation = (pose.translation + 0.5) * size
                width = g.width * size
                new_grasps.append(Grasp(pose, width))
            scores = scores[p]
        grasps = new_grasps

        if self.visualize:
            grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            composed_scene = trimesh.Scene(colored_scene_mesh)
            for i, g_mesh in enumerate(grasp_mesh_list):
                composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            return grasps, scores, toc, composed_scene
        else:
            return grasps, scores, toc


def bound(qual_vol, voxel_size, limit=[0.02, 0.02, 0.055]):
    '''清空在边缘的几个体素 avoid grasp out of bound [0.02  0.02  0.055]
    '''
    x_lim = int(limit[0] / voxel_size)
    y_lim = int(limit[1] / voxel_size)
    z_lim = int(limit[2] / voxel_size)
    qual_vol[:x_lim] = 0.0
    qual_vol[-x_lim:] = 0.0
    qual_vol[:, :y_lim] = 0.0
    qual_vol[:, -y_lim:] = 0.0
    qual_vol[:, :, :z_lim] = 0.0
    return qual_vol


def predict(tsdf_vol, pos, net, device):
    '''输入 TSDF 和 判断抓取的 pos, 网络输出对应每个点的抓取质量、旋转、宽度
    '''
    assert tsdf_vol.shape == (1, 40, 40, 40)

    # move input to the GPU
    tsdf_vol = torch.from_numpy(tsdf_vol).to(device)

    # forward pass
    batch_dict = EasyDict({"tsdf": tsdf_vol,
                           "point_grasp": pos})  # 输入 TSDF(40**3) 和 pos(-0.5~0.5 三维点)
    with torch.no_grad():
        prediction = net(batch_dict, geometry_branch=False)
        qual_vol, rot_vol, width_vol = prediction["grasp_label"], prediction["grasp_rotation"], prediction["grasp_width"]

    # move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    return qual_vol, rot_vol, width_vol


def process(tsdf_vol, qual_vol, rot_vol, width_vol,
            gaussian_filter_sigma=1.0,
            min_width=0.033,
            max_width=0.233,
            out_th=0.5):
    '''数据后处理: 平滑抓取质量网格, 过滤内部, 过滤太远的, 过滤抓取宽度不合适的
    '''
    tsdf_vol = tsdf_vol.squeeze()

    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > out_th  # 选出 TSDF 大的(距离远的)
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    valid_voxels = ndimage.morphology.binary_dilation(  # 在 mask=True 区域(较远区域), 膨胀候选区域
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select(qual_vol, center_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4, force_detection=False):
    '''
    Args:
        center_vol: -0.5 ~ 0.5 三维网格点
    '''
    best_only = False
    qual_vol[qual_vol < LOW_TH] = 0.0  # 非常非常低质量的直接去掉
    if force_detection and (qual_vol >= threshold).sum() == 0:  # 如果没有合适的抓取点超过阈值
        best_only = True
    else:
        # threshold on grasp quality
        qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression  相当于去掉次优的 每个峰只留下最优
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)  # 找到 8*8*8 窗口中的最大值(相当于膨胀)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)  # 只有局部 8*8*8 窗口最大值满足等号
    mask = np.where(qual_vol, 1.0, 0.0)  # 一系列局部最大 qual 点

    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index(qual_vol, center_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
    sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]  # 按照 score 从大到小排序

    if best_only and len(sorted_grasps) > 0:
        sorted_grasps = [sorted_grasps[0]]
        sorted_scores = [sorted_scores[0]]

    return sorted_grasps, sorted_scores


def select_index(qual_vol, center_vol, rot_vol, width_vol, index):
    '''用 index 取值, 包括格式转换, 返回一个 Grasp 对象(仅用于包含 pos 和 width)
    '''
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[i, j, k])
    # pos = np.array([i, j, k], dtype=np.float64)
    pos = center_vol[i, j, k].numpy()
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score
