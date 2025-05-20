import cv2
import pykinect_azure as pykinect
import numpy as np
from .utils.plot3dUtils import Open3dVisualizer
from sensors.sensor import Kinect_Camera

class RGB_Kinect(Kinect_Camera):
    def __init__(self, **kwargs):
        super().__init__(**kwargs,color_resolution=pykinect.K4A_COLOR_RESOLUTION_1080P)

    def get_sensor_info(self):

        color_image = None
        depth_img = None
        point_cloud = None
        
        # Get capture
        capture = self.device.update()

        # Get the color image from the capture
        ret, color_image = capture.get_color_image()
        
        return ret, color_image,  depth_img, point_cloud

class RGB_Depth_Kinect(Kinect_Camera):
    def __init__(self, *args):
        super().__init__(*args)
    
    def get_sensor_info(self):

        color_image = None
        depth_img = None
        point_cloud = None
        

        capture = self.device.update()
        ret_color, color_image = capture.get_color_image()
        ret_depth, transformed_colored_depth_image = capture.get_transformed_colored_depth_image()
        
        depth_img = transformed_colored_depth_image
        
        return ret_depth & ret_color, color_image,  depth_img, point_cloud

class PointCloud_Kinect(Kinect_Camera):
    def __init__(self, *args):
        super().__init__(*args)
    
    def get_sensor_info(self):

        color_image = None
        depth_img = None
        point_cloud = None
        

        capture = self.device.update()
        ret_point, points = capture.get_transformed_pointcloud()
        ret_color, color_image = capture.get_color_image()
        
        return ret_point & ret_color, color_image,  depth_img, point_cloud
