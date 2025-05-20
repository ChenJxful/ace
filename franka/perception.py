import numpy as np
from scipy.spatial.transform import Rotation as R
from deoxys.utils.transform_utils import pose2mat,mat2pose,pose_in_A_to_pose_in_B,quat2axisangle,axisangle2quat
import open3d as o3d

SUNHAN_POSE_CAMERA_IN_EE = [0.050911, -0.036769, -0.041399]
SUNHAN_QUAT_CAMERA_IN_EE = [-0.0030886, -0.0056089, 0.7171562, 0.696578]

# POSE_CAMERA_IN_EE = SUNHAN_POSE_CAMERA_IN_EE
POSE_CAMERA_IN_EE = [0.0467162 , -0.0388289 , -0.0419308]  
# POSE_CAMERA_IN_EE = [0.0436099 , -0.0439066 , -0.0420663] 
# POSE_CAMERA_IN_EE = [0.0469654,  -0.0389453,  -0.041804] 
# POSE_CAMERA_IN_EE = [0.036801 ,  -0.0320831,  -0.0484176]
# # the quat is [x,y,z,w]

# QUAT_CAMERA_IN_EE = SUNHAN_QUAT_CAMERA_IN_EE
QUAT_CAMERA_IN_EE = [-0.008896206339478014, 0.001529932861133454, 0.7130445599416992, 0.7010606053371948]
# QUAT_CAMERA_IN_EE = [-0.009638185029200455, 0.0050344256353956895, 0.7027567510587952, 0.71134710851254]
# QUAT_CAMERA_IN_EE = [-0.008965061839253031, 0.0014138577443180145, 0.7130851281413529, 0.701014807018385]
# QUAT_CAMERA_IN_EE = [-0.0028981363311248102, -0.008938693827707178, 0.7161214816004796,0.6979124043522906]

# new quat is [x,y,z,w] 
# POSE_CAMERA_IN_EE = [0.0569333,-0.0957159,-0.04444]  
# # the quat is [x,y,z,w]
# QUAT_CAMERA_IN_EE = [0.0217634, -0.0073362, 0.7481789, 0.663099]


CAMERA_IN_EE_MAT = pose2mat([POSE_CAMERA_IN_EE, QUAT_CAMERA_IN_EE])

def depth_2_point(depth,color_image,intr):
    fx,fy,cx,cy = intr.fx,intr.fy,intr.ppx,intr.ppy

    x,y = np.meshgrid(np.arange(depth.shape[1]),np.arange(depth.shape[0]))
    color = color_image.reshape(-1,3) / 255.0
    uv_depth = np.zeros((depth.shape[0],depth.shape[1],3))
    uv_depth[:,:,0] = x
    uv_depth[:,:,1] = y
    uv_depth[:,:,2] = depth
    uv_depth = np.reshape(uv_depth,[-1,3])

    color = color[np.where(uv_depth[:,2]!=0),:].squeeze()
    uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze()

    color = color[np.where(uv_depth[:,2]<2.0),:].squeeze()
    uv_depth = uv_depth[np.where(uv_depth[:,2]<2.0),:].squeeze()
    
    n = uv_depth.shape[0]
    points = np.ones((n,4))
    print(n)
    X = (uv_depth[:,0]-cx)*uv_depth[:,2]/fx
    Y = (uv_depth[:,1]-cy)*uv_depth[:,2]/fy
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = uv_depth[:,2]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(color)
    
    return pcd, points

def single_point_to_pc(depth,intr,x,y):
    fx,fy,cx,cy = intr.fx,intr.fy,intr.ppx,intr.ppy
    X = (x-cx)*depth/fx
    Y = (y-cy)*depth/fy
    return np.array([X,Y,depth,1])


def trans_point_to_base(point_in_camera,ee_in_base_mat):
            
    camera_in_base_mat = pose_in_A_to_pose_in_B(CAMERA_IN_EE_MAT, ee_in_base_mat)
    point_in_base = point_in_camera.dot(camera_in_base_mat.T)
    
    return point_in_base

def trans_pose_to_base(pose_in_camera,ee_in_base_mat):
            
    camera_in_base_mat = pose_in_A_to_pose_in_B(CAMERA_IN_EE_MAT, ee_in_base_mat)
    pose_in_base = pose_in_A_to_pose_in_B( pose_in_camera,camera_in_base_mat)
    return pose_in_base

def get_camera_in_base_mat(ee_in_base_mat):
    camera_in_base_mat = pose_in_A_to_pose_in_B(CAMERA_IN_EE_MAT, ee_in_base_mat)
    return camera_in_base_mat

