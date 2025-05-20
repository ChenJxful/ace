import open3d as o3d
import numpy as np

def get_coordinate(size=0.5, origin=[0, 0, 0]):
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
    return coord

def get_pcd_list_from_mesh_pose_list(mesh_pose_list):
    obj_list = []
    for mesh_path, scale, pose in mesh_pose_list:
        obj = o3d.io.read_triangle_mesh(mesh_path)
        obj.scale(scale, center=obj.get_center())
        obj.transform(pose)
        obj.paint_uniform_color(np.random.rand(3))
        obj_list.append(obj)
    return obj_list

def get_pcd_from_np(np_array, color=None):
    """
    np_array: (..., 3)
    color: RGB like [0, 0, 1]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array.reshape(-1, 3))
    if color:
        pcd.paint_uniform_color(color)
    return pcd

def get_rays_from_np(starts, ends):
    """
    starts / ends: (..., 3)
    """
    lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
                o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(starts.reshape(-1, 3))),
                o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(ends.reshape(-1, 3))),
                o3d.cpu.pybind.utility.Vector2iVector(np.arange(0, 3072).reshape(-1, 2)))
    return lines    


def get_table():
    box = o3d.geometry.TriangleMesh.create_box(width=0.3, height=0.3, depth=0.000001)
    box.transform(np.array([[ 1.    ,  0.    ,  0.    ,  0.    ],
                                [ 0.    ,  1.    ,  0.    ,  0.    ],
                                [ 0.    ,  0.    ,  1.    ,  0.05  ],
                                [ 0.    ,  0.    ,  0.    ,  1.    ]]))
    return box

def get_camera_list(extrinsic, intrinsic=[540.0, 540.0, 320.0, 240.0, 640, 480], 
                size=0.1, color="linear_g"):
    """
    extrinsic: list of (4, 4)
    intrinsic: (fx, fy, cx, cy, W, H)
    """
    fx, fy, cx, cy, W, H = intrinsic
    cam_list = []
    for index, ext in enumerate(extrinsic):
        cam = o3d.geometry.LineSet.create_camera_visualization(
            o3d.cpu.pybind.camera.PinholeCameraIntrinsic(W, H, np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])), 
            ext, scale=size)
        if color == "linear_r":
            cam.paint_uniform_color([1.0 / len(extrinsic) * (index+1), 0, 0])
        elif color == "linear_g":
            cam.paint_uniform_color([0, 1.0 / len(extrinsic) * (index+1), 0])
        elif color == "linear_b":
            cam.paint_uniform_color([0, 0, 1.0 / len(extrinsic) * (index+1)])
        elif color == "random":
            cam.paint_uniform_color(np.random.rand(3))            
        cam_list.append(cam)
    return cam_list