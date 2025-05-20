import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

# depth = cv2.imread('output/depth_0.png', cv2.IMREAD_ANYDEPTH)

pipeline = rs.pipeline()
profile = pipeline.start()

frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()
pc = rs.pointcloud()
pc.map_to(color_frame)
points = pc.calculate(depth_frame)

vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # xyz


print(pc)

