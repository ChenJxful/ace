import cv2
# import pykinect_azure as pykinect
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import pyrealsense2 as rs

# from open3d.camera import PinholeCameraIntrinsic


# class Kinect_Camera():
#     def __init__(self,color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32,color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P,\
#                 depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED, camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30,\
#                 synchronized_images_only = False):
#         pykinect.initialize_libraries()
#         device_config = pykinect.default_configuration
#         device_config.color_resolution = color_resolution
#         device_config.color_format = color_format
#         device_config.depth_mode = depth_mode
#         device_config.camera_fps = camera_fps
#         device_config.synchronized_images_only = synchronized_images_only
#         self.device_config = device_config
#         self.device = pykinect.start_device(config=device_config)
        
#     def get_sensor_info(self):
        
#         raise NotImplementedError

class Realsense_Camera():
    def __init__(self):
        align = rs.align(rs.stream.color)
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
        pipeline = rs.pipeline()
        
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        self.intr.fx = 596.382688
        self.intr.fy = 596.701788

        self.intr.ppx = 333.837001
        self.intr.ppy = 254.401211
        self.intr.model = rs.distortion.modified_brown_conrady
        self.coeffs = [0.130405, -0.265366, 0.000961, 0.003105, 0.0]
        
        # import argparse
        # self.intr = argparse.Namespace(**{
        #     'width': 640,
        #     'height': 480,
        #     'fx':593.564564,
        #     'fy':596.588446,
        #     'ppx':278.755449,
        #     'ppy':249.592011,
        #     'model':rs.distortion.modified_brown_conrady,
        #     "coeffs" : [0.153998, -0.128291, -0.003375,-0.027473,0.0]
        # })
        # get camera intrinsics

        # print(self.intr.width, self.intr.height, self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy)
        
        # pinhole_camera_intrinsic = PinholeCameraIntrinsic(self.intr.width, self.intr.height, self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy)

        self.config = config
        self.pipeline = pipeline
        self.align = align
        # self.pinhole_camera_intrinsic = pinhole_camera_intrinsic
    def get_sensor_info(self):
        
        raise NotImplementedError

    def get_intr(self):
        return self.intr
