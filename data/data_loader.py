import torch
import numpy as np
from pathlib import Path
from scipy import ndimage

import sys
import os
sys.path.append(os.getcwd())
from utils.misc import load_config, EasyDict

from vgn.io import my_IO
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_scene_from_mesh_pose_list
from typing import Any, List, Tuple, Union


class SimOccGrasp_Dataset(torch.utils.data.Dataset):
    """自定义数据集
    """
    def __init__(self, root, with_occ_data=False, num_point_occ=2048, augment=False):
        self.augment = augment  # not used
        self.num_point_occ = num_point_occ
        self.root = root
        self.num_th = 32
        self.io = my_IO(root)
        self.df = self.io.read_df()
        self.size, self.intrinsic, _, _ = self.io.read_setup()  # read setup.json
        self.with_occ_data = with_occ_data
        self.single_yaw = 'yaw' in self.df.columns

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        """read data from batch, change format
        Returns(dict):
            tsdf: N*N*N, tsdf 数据
            point_grasp: 1*3, query grasp 的位置
            grasp_label: 1, 抓取成功标签
            grasp_rotation: 2*4, 抓取姿态(四元数) / 1, yaw 角 (0, pi)
            grasp_width: 1, 抓取宽度
            point_occ: 2048*3, query occ 的位置
            occupancy: 2048, query occ 结果
            camera_extrinsic: n*7, 相机外参, [rotation.as_quaternions, translation]
            camera_intrinsic: 6, 相机内参, ['fx', 'fy', 'cx', 'cy', 'width', 'height']
            depth_img: n*H*W, n张深度图
        """
        batch_dict = EasyDict()

        # read from grasps.csv
        scene_id = self.df.loc[i, "scene_id"]
        batch_dict.scene_id = scene_id

        if self.single_yaw:
            ori = self.df.loc[i, "yaw"]
        else:
            ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))
        pos = self.df.loc[i, "x":"z"].to_numpy(np.single)
        width = self.df.loc[i, "width"].astype(np.single)
        label = self.df.loc[i, "label"].astype(np.int64)
        batch_dict.grasp_label = torch.tensor(label, dtype=torch.float32)

        # read from voxel_grid
        voxel_grid = self.io.read_voxel_grid(scene_id)

        if self.augment:
            voxel_grid, ori, pos = self.apply_transform(voxel_grid, ori, pos)

        batch_dict.tsdf = torch.tensor(voxel_grid[0], dtype=torch.float32)

        pos = pos / self.size - 0.5  # [0.0, 0.3] normalize to [-0.5, 0.5]
        width = width / self.size
        batch_dict.grasp_width = torch.tensor(width, dtype=torch.float32)
        batch_dict.point_grasp = torch.tensor(pos, dtype=torch.float32).unsqueeze_(0)  # [3] → [1, 3]

        if self.single_yaw:
            rotations = ori  # 不加对称的角度
        else:
            rotations = np.empty((2, 4), dtype=np.single)
            R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
            rotations[0] = ori.as_quat()
            rotations[1] = (ori * R).as_quat()
        batch_dict.grasp_rotation = torch.tensor(rotations, dtype=torch.float32)

        # read from occ
        if self.with_occ_data:
            occ_points, occ = self.io.read_occ(scene_id, self.num_point_occ)
            occ_points = occ_points / self.size - 0.5

            batch_dict.point_occ = torch.tensor(occ_points, dtype=torch.float32)
            batch_dict.occupancy = torch.tensor(occ, dtype=torch.float32)

        # read from depth
        batch_dict.camera_intrinsic = torch.tensor([getattr(self.intrinsic, k) for k in ['fx', 'fy', 'cx', 'cy', 'width', 'height']])
        depth_images, extrinsics = self.io.read_depth_image(scene_id)
        batch_dict.depth_img = torch.tensor(depth_images)
        batch_dict.camera_extrinsic = torch.tensor(extrinsics)

        return batch_dict

    # ↓ not used
    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene

    def apply_transform(self, voxel_grid, orientation, position):
        angle = np.pi / 2.0 * np.random.choice(4)
        R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, angle])

        z_offset = np.random.uniform(6, 34) - position[2]

        t_augment = np.r_[0.0, 0.0, z_offset]
        T_augment = Transform(R_augment, t_augment)

        T_center = Transform(Rotation.identity(), np.r_[20.0, 20.0, 20.0])
        T = T_center * T_augment * T_center.inverse()

        # transform voxel grid
        T_inv = T.inverse()
        matrix, offset = T_inv.rotation.as_matrix(), T_inv.translation
        voxel_grid[0] = ndimage.affine_transform(voxel_grid[0], matrix, offset, order=0)

        # transform grasp pose
        position = T.transform_point(position)
        orientation = T.rotation * orientation

        return voxel_grid, orientation, position


def create_train_val_loaders(root: Union[str, List[str]], batch_size: int, val_split: float, kwargs):
    """
    From custom dataset To torch.dataloader
    validation set size: val_split * len(dataset)
    training set size: (1 - val_split) * len(dataset)
    """
    # deal with multiple datasets
    if isinstance(root, str):
        dataset = SimOccGrasp_Dataset(root)
    else:
        assert isinstance(root[0], str)  # may be omegaconf.listconfig
        dataset = torch.utils.data.ConcatDataset([SimOccGrasp_Dataset(r) for r in root])

    # split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # create loaders for both datasets
    # Note: shuffle: random the batch at every epoch
    #       drop_last: drop the last batch if it is smaller than batch_size
    if root[:2] == "s3" and kwargs["num_workers"] > 0:  # if in s3 and petrel_client installed
        try:
            from petrel_client.utils.data import DataLoader
            kwargs = {"batch_size": batch_size, "drop_last": True,
                      "prefetch_factor": 4, "persistent_workers": True,
                      **kwargs}
            train_loader = DataLoader(train_set, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, shuffle=False, **kwargs)
        except ImportError:
            pass
    if 'train_loader' not in dir():
        kwargs = {"batch_size": batch_size, "drop_last": True,
                  **kwargs}
        train_loader = torch.utils.data.DataLoader(
            train_set, shuffle=True, **kwargs
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, shuffle=False, **kwargs
        )
    return train_loader, val_loader


if __name__ == "__main__":
    # Load config
    # cfg = load_config("AGATE.yaml")
    cfg = {
        "dataset": ["data/packed/data_packed_facing_grasp",
                    "data/packed/data_packed_multiview_grasp",
                    "data/packed/data_packed_side_grasp"],
        "batch_size": 3,
        "val_split": 0.1,
    }

    use_cuda = False  # torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 16, "pin_memory": True} if use_cuda else {"num_workers": 0}

    train_loader, val_loader = create_train_val_loaders(
        cfg["dataset"], cfg["batch_size"], cfg["val_split"], kwargs)

    train_loader_iterator = iter(train_loader)
    for i in range(30):
        b = next(train_loader_iterator)
        print(b['scene_id'])
        # breakpoint()
