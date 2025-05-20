from .sensor import Realsense_Camera
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import pyrealsense2 as rs

import time
import open3d as o3d
# from open3d.camera import PinholeCameraIntrinsic
from open3d.visualization import Visualizer
import os


class RealSenseCamera(Realsense_Camera):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def get_sensor_info(self):
        ret = True
        point_cloud = None
        
        frames = self.pipeline.wait_for_frames(9999)
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        # color_image = np.asanyarray(color_frame.get_data())
        depth_frame = aligned_frames.get_depth_frame()
        # depth_frame = rs.decimation_filter(1).process(depth_frame)
        # depth_frame = rs.disparity_transform(True).process(depth_frame)
        # depth_frame = rs.spatial_filter().process(depth_frame)
        # depth_frame = rs.temporal_filter().process(depth_frame)
        # depth_frame = rs.disparity_transform(False).process(depth_frame)
        # depth_frame = rs.hole_filling_filter().process(depth_frame)
        
        # color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        # depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image * self.depth_scale

        # points = rs.pointcloud()
        # points.calculate(depth_frame)
        # points.map_to(color_frame)
        
        # vtx = np.asanyarray(points.get_vertices_2d())
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(vtx)
        
        # print(pcd)
        # # color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        
        # print("color shape is:", color_image.shape)
        # print("depth shape is:", depth_image.shape)
        
        return ret, color_image, depth_image, point_cloud

    def get_colorize_depth(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()

        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(aligned_depth_frame)
        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())
        return colorized_depth


def get_all():
    align = rs.align(rs.stream.color)
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    # get camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    # pinhole_camera_intrinsic = PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    # print(type(pinhole_camera_intrinsic))
    print(intr)
    vis = Visualizer()
    vis.create_window("Pointcloud",640,480)
    pointcloud = o3d.geometry.PointCloud()
    i = 0

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("depth scale is: ", depth_scale)

    try:
        while True:
            time_start = time.time()
            pointcloud.clear()

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_frame = aligned_frames.get_depth_frame()

            depth_frame = rs.decimation_filter(1).process(depth_frame)
            depth_frame = rs.disparity_transform(True).process(depth_frame)
            depth_frame = rs.spatial_filter().process(depth_frame)
            depth_frame = rs.temporal_filter().process(depth_frame)
            depth_frame = rs.disparity_transform(False).process(depth_frame)
            # depth_frame = rs.hole_filling_filter().process(depth_frame)
            

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            # depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.namedWindow('color image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('color image', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
            cv2.namedWindow('depth image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('depth image', depth_image )

            depth = o3d.geometry.Image(depth_image)
            color = o3d.geometry.Image(color_image)

            # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
            # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
            # pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            # # pcd = voxel_down_sample(pcd, voxel_size = 0.003)

            # pointcloud = pcd

            # if not geometrie_added:
            #     vis.add_geometry(pointcloud)
            #     geometrie_added = True


            # vis.update_geometry(pointcloud)
            # vis.poll_events()
            # vis.update_renderer()
            time_end = time.time()

            key = cv2.waitKey(1)

            # print("FPS = {0}".format(int(1/(time_end-time_start))))

            # press 's' to save current RGBD images and pointcloud.
            if key & 0xFF == ord('s'):
                if not os.path.exists('./output/'): 
                    os.makedirs('./output')
                cv2.imwrite('./output/depth_'+str(i)+'.png',depth_image)
                cv2.imwrite('./output/color_'+str(i)+'.png',color_image1)
                # o3d.io.write_point_cloud('./output/pointcloud_'+str(i)+'.pcd', pcd)
                print('No.'+str(i) + ' shot is saved.' )
                i += 1


            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                vis.destroy_window()

                break
    finally:
        pipeline.stop()

if __name__=="__main__":
    get_all()