import cv2
import time
import argparse
import datetime
import numpy as np
import pyrealsense2 as rs
try:
    import open3d as o3d
except ImportError:
    print("Running without Open3D.")


class Realsense:
    def __init__(self):
        # 相机配置
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)

        # 相机深度参数，包括精度以及 depth_scale
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 3)
        self.depth_scale = depth_sensor.get_depth_scale()
        self.clipping_distance_in_meters = 8  # 8 meter
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

        # color和depth对齐
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # 读取内参
        self.intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        return aligned_depth_frame, color_frame


def save_img_with_timestamp(img, path, name, timestamp):
    time_text = datetime.datetime.fromtimestamp(timestamp).strftime('%m%d_%H%M%S')
    cv2.imwrite(f'{path}/{time_text}_{name}.png', img)
    return time_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--open3d', action='store_true', help='Use Open3D to visualize point cloud.')
    parser.add_argument('--nodisplay', action='store_true', help='Do not display images.')
    args = parser.parse_args()
    USE_OPEN3D = args.open3d
    NO_DISPLAY = args.nodisplay

    cam = Realsense()
    print('\033[32m' + 'Start Detection!' + '\033[0m')

    while True:
        # 读取图像
        depth_frame, color_frame = cam.get_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if USE_OPEN3D and not NO_DISPLAY:
            o3d_inter = o3d.camera.PinholeCameraIntrinsic(cam.intrinsics.width, cam.intrinsics.height,
                                                        cam.intrinsics.fx, cam.intrinsics.fy,
                                                        cam.intrinsics.ppx, cam.intrinsics.ppy)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_image.copy()),
                o3d.geometry.Image(depth_image),
                depth_scale=1.0 / cam.depth_scale,
                depth_trunc=cam.clipping_distance_in_meters,
                convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_inter)
            camera = o3d.geometry.LineSet.c
            o3d.visualization.draw_geometries([pcd])  # 非实时
        else:
            # RGB and Depth
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            hole_filling = rs.hole_filling_filter()
            filled_depth = hole_filling.process(depth_frame)
            colorizer = rs.colorizer()
            colorized_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())

            if NO_DISPLAY:  # 仅保存图像
                save_img_with_timestamp(color_image, 'logs', 'color', color_frame.timestamp/1000)
                time_text = save_img_with_timestamp(colorized_depth, 'logs', 'depth', depth_frame.timestamp/1000)
                print(f'Saved 1 frame at {time_text}.')
                time.sleep(3.0)

            else:
                cv2.imshow('color_image', color_image)
                cv2.imshow('filled depth', colorized_depth)

                k = cv2.waitKey(1)
                if k == ord('q'):
                    break
                elif k == ord('s'):
                    save_img_with_timestamp(color_image, 'logs', 'color', color_frame.timestamp/1000)
                    save_img_with_timestamp(colorized_depth, 'logs', 'depth', depth_frame.timestamp/1000)
