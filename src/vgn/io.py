import json
import uuid
import io

import numpy as np
import pandas as pd
from pathlib import Path

from vgn.grasp import Grasp
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform


class my_IO(object):
    def __init__(self, root):
        if isinstance(root, Path) or root[:2] != "s3":  # 旧版的 pathlib 对象或者字符串形式本地路径
            self.USE_OSS = False
        else:
            self.USE_OSS = True  # TODO only supported read from OSS
        self.root = root

        if self.USE_OSS:
            from petrel_client.client import Client
            # 读写文件配置
            conf_path = '/mnt/petrelfs/petreloss.conf'
            self.client = Client(conf_path)
        else:
            self.client = None

        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    def write_setup(self, size, intrinsic, max_opening_width, finger_depth):
        '''
        e.g. {'size': 0.30000000000000004,
            'intrinsic': {'width': 640, 'height': 480,
                'K': [540.0, 0.0, 320.0, 0.0, 540.0, 240.0, 0.0, 0.0, 1.0]},
            'max_opening_width': 0.08, 'finger_depth': 0.05}
        '''
        def write_json(data, path):
            with path.open("w") as f:
                json.dump(data, f, indent=4)

        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "size": size,  # 0.3
            "intrinsic": intrinsic.to_dict(),
            "max_opening_width": max_opening_width,
            "finger_depth": finger_depth,
        }
        if self.USE_OSS:
            self.client.put(self.root + '/setup.json', json.dumps(data, indent=4))
        else:
            write_json(data, self.root / "setup.json")

    def read_setup(self):
        '''read setup.json, return (size, intrinsic, max_opening_width, finger_depth)
        '''
        def read_json(path):
            with path.open("r") as f:
                data = json.load(f)
            return data

        if self.USE_OSS:
            d = self.client.get(self.root + "/setup.json")
            data = json.loads(d)
        else:
            data = read_json(Path(self.root) / "setup.json")
        size = data["size"]
        intrinsic = CameraIntrinsic.from_dict(data["intrinsic"])
        max_opening_width = data["max_opening_width"]
        finger_depth = data["finger_depth"]
        return size, intrinsic, max_opening_width, finger_depth

    def write_mesh_pose_list(self, scene_id, mesh_pose_list, name="mesh_pose_list"):
        path = self.root / name / (scene_id + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, mesh_pose_list=mesh_pose_list)

    def read_mesh_pose_list(self, scene_id):
        if self.USE_OSS:
            npz_file = self.root + "/mesh_pose_list/" + scene_id + ".npz"
            npz_bytes = self.client.get(npz_file)
            data = np.load(io.BytesIO(npz_bytes), allow_pickle=True)
            return data["mesh_pose_list"]
        else:
            path = self.root / "mesh_pose_list" / (scene_id + ".npz")
            return np.load(path, allow_pickle=True)["mesh_pose_list"]

    def write_depth_image(self, scene_id, depth_imgs, extrinsics, name="depth_imgs"):
        path = self.root / name / (scene_id + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics)

    def read_depth_image(self, scene_id, name="depth_imgs"):
        if self.USE_OSS:
            npz_file = self.root + "/" + name + "/" + scene_id + ".npz"
            npz_bytes = self.client.get(npz_file)
            data = np.load(io.BytesIO(npz_bytes), allow_pickle=True)
            return data["depth_imgs"], data["extrinsics"]
        else:
            data = np.load(Path(self.root) / name / (scene_id + ".npz"))
            return data["depth_imgs"], data["extrinsics"]

    def write_grasp(self, scene_id, grasp, label):
        '''创建并写入 grasps.csv
        '''
        # TODO concurrent writes could be an issue | BIG PROBLEM do not support multi-process
        csv_path = self.root / "grasps.csv"
        if not csv_path.exists():  # 写入首行
            self.__create_csv(
                csv_path,
                ["scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label"],
            )
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        self.__append_csv(csv_path, scene_id, qx, qy, qz, qw, x, y, z, width, label)

    def write_grasp_yaw(self, scene_id, pos, yaw, width, label):
        '''创建并写入 grasps.csv, 只写入 yaw 而不是完整四元数
        '''
        # TODO concurrent writes could be an issue | BIG PROBLEM do not support multi-process
        csv_path = self.root / "grasps.csv"
        if not csv_path.exists():  # 写入首行
            self.__create_csv(
                csv_path,
                ["scene_id", "x", "y", "z", "yaw", "width", "label"],
            )
        self.__append_csv(csv_path, scene_id, pos[0], pos[1], pos[2], yaw, width, label)

    def __create_csv(self, path, columns):
        with path.open("w") as f:
            f.write(",".join(columns))
            f.write("\n")

    def __append_csv(self, path, *args):
        row = ",".join([str(arg) for arg in args])
        with path.open("a") as f:
            f.write(row)
            f.write("\n")


    # not used!
    def read_grasp(df, i):
        scene_id = df.loc[i, "scene_id"]
        orientation = Rotation.from_quat(df.loc[i, "qx":"qw"].to_numpy(np.double))
        position = df.loc[i, "x":"z"].to_numpy(np.double)
        width = df.loc[i, "width"]
        label = df.loc[i, "label"]
        grasp = Grasp(Transform(orientation, position), width)
        return scene_id, grasp, label

    def read_df(self):
        '''使用 pandas 读取 grasps.csv
        '''
        if self.USE_OSS:
            d = self.client.get(self.root + "/grasps.csv")
            return pd.read_csv(io.BytesIO(d))
        else:
            return pd.read_csv(Path(self.root) / "grasps.csv")

    def write_df(self, df):
        df.to_csv(self.root / "grasps.csv", index=False)

    def write_voxel_grid(self, scene_id, voxel_grid):
        path = self.root / "voxel_grid" / (scene_id + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, grid=voxel_grid)

    def read_voxel_grid(self, scene_id):
        if self.USE_OSS:
            npz_file = self.root + "/voxel_grid/" + scene_id + ".npz"
            npz_bytes = self.client.get(npz_file)
            data = np.load(io.BytesIO(npz_bytes), allow_pickle=True)
            return data["grid"]
        else:
            path = Path(self.root) / "voxel_grid" / (scene_id + ".npz")
            return np.load(path)["grid"]

    def write_point_cloud(self, scene_id, point_cloud, name="point_clouds_with_noise_crop"):
        # TODO data not used, maybe .pcd better
        path = self.root / name / (scene_id + ".pcd")
        path.parent.mkdir(parents=True, exist_ok=True)

        o3d.io.write_point_cloud(str(path), point_cloud)
        point_cloud = np.asarray(point_cloud.points)

        path = self.root / name / (scene_id + ".npz")
        np.savez_compressed(path, pc=point_cloud)

    def read_point_cloud(self, scene_id, name="point_clouds"):
        if self.USE_OSS:
            npz_file = self.root + "/" + name + "/" + scene_id + ".npz"
            npz_bytes = self.client.get(npz_file)
            data = np.load(io.BytesIO(npz_bytes), allow_pickle=True)
            return data["pc"]
        else:
            path = Path(self.root) / name / (scene_id + ".npz")
            return np.load(path)["pc"]

    def write_occ(self, scene_id, occ, points, num_file, name="occ"):
        '''save occ, split the data into multiple files
        '''
        for i in range(num_file):
            path = self.root / 'occ' / scene_id / ('%04d.npz' % (i,))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, points=points[i], occ=occ[i])

    def read_occ(self, scene_id, num_point):
        '''从多个 occ 文件中随机选择一个, 随机采样其中 num_point 个点
        '''
        # randomly select one occ file
        if self.USE_OSS:
            occ_paths = list(self.client.list(self.root + "/occ/" + scene_id))
        else:
            occ_paths = list((Path(self.root) / 'occ' / scene_id).glob('*.npz'))
        path_idx = np.random.randint(len(occ_paths))
        occ_path = occ_paths[path_idx]
        if self.USE_OSS:
            npz_bytes = self.client.get(self.root + "/occ/" + scene_id + '/' + occ_path)
            occ_data = np.load(io.BytesIO(npz_bytes), allow_pickle=True)
        else:
            occ_data = np.load(occ_path)
        points = occ_data['points']
        occ = occ_data['occ']
        # randomly sample num_point points (if num_point > num_point_all, then allow repeat)
        num_point_all = points.shape[0]
        idxs = np.random.choice(np.arange(num_point_all), size=(num_point,), replace=num_point > num_point_all)
        return points[idxs], occ[idxs]


if __name__ == "__main__":
    root = "s3://Agate_data/data_pile_train_4000000"
    scene_id = "4fb20b66a4454430947553e8e80ce516"

    myio = my_IO(root)
    df = myio.read_df()
    imgs, exts = myio.read_depth_image(scene_id)
    for i in range(imgs.shape[0]):
        img = imgs[i]
        ext = exts[i]
        print(img.shape)
        print(ext.shape)
