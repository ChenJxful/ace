from pathlib import Path
import os
import time
import pickle
import numpy as np
import pybullet
from pybullet_utils import bullet_client

from vgn.perception import *
from vgn.utils.transform import Rotation, Transform
from vgn.utils.saver import get_mesh_pose_dict_from_world

assert pybullet.isNumpyEnabled(), "Pybullet needs to be built with NumPy"

from cv2 import imwrite, VideoWriter, VideoWriter_fourcc
from datetime import datetime


class BtWorld(object):
    """Interface to a PyBullet physics server.

    Attributes:
        dt: Time step of the physics simulation.
        rtf: Real time factor. If negative, the simulation is run as fast as possible.
        sim_time: Virtual time elpased since the last simulation reset.
    """

    def __init__(self, gui=True, dir=None,
                 save_pkl=False, save_freq=8, log_render=False):
        '''
        dir: pkl 和离线渲染的保存路径
        save: 是否保存 pkl
        log_render: 是否开启离线渲染
        '''
        connection_mode = pybullet.GUI if gui else pybullet.DIRECT
        self.p = bullet_client.BulletClient(connection_mode)

        self.gui = gui
        self.dt = 1.0 / 240.0
        self.solver_iterations = 150
        self.dir = dir
        self.save_pkl = save_pkl
        if self.save_pkl:
            Path(self.dir / "steps_log").mkdir(parents=True, exist_ok=True)
            self.save_freq = save_freq
        self.sim_step = 0

        if log_render:
            intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
            self.log_renderer = OfflineRenderer(self.add_camera(intrinsic, 0.1, 2.0),
                                                self.dir / "render_log/raw")
        else:
            self.log_renderer = None
        self.reset()

    def set_gravity(self, gravity):
        self.p.setGravity(*gravity)

    def load_urdf(self, urdf_path, pose, scale=1.0):
        body = Body.from_urdf(self.p, urdf_path, pose, scale)
        self.bodies[body.uid] = body
        return body

    def remove_body(self, body):
        self.p.removeBody(body.uid)
        del self.bodies[body.uid]

    def add_constraint(self, *argv, **kwargs):
        """See `Constraint` below."""
        constraint = Constraint(self.p, *argv, **kwargs)
        return constraint

    def add_camera(self, intrinsic, near, far):
        camera = Camera(self.p, intrinsic, near, far)
        return camera

    def get_contacts(self, bodyA):
        points = self.p.getContactPoints(bodyA.uid)
        contacts = []
        for point in points:
            contact = Contact(
                bodyA=self.bodies[point[1]],
                bodyB=self.bodies[point[2]],
                point=point[5],
                normal=point[7],
                depth=point[8],
                force=point[9],
            )
            contacts.append(contact)
        return contacts

    def reset(self):
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(
            fixedTimeStep=self.dt, numSolverIterations=self.solver_iterations
        )
        self.bodies = {}
        self.sim_time = 0.0

    def step(self):
        self.p.stepSimulation()

        if self.gui:
            time.sleep(self.dt)
        if self.save_pkl:
            if self.sim_step % self.save_freq == 0:
                mesh_pose_dict = get_mesh_pose_dict_from_world(self, self.p._client)
                with open(os.path.join(self.dir, "steps_log", f'{self.sim_step:08d}.pkl'), 'wb') as f:
                    pickle.dump(mesh_pose_dict, f)

        self.sim_time += self.dt
        self.sim_step += 1
        if self.log_renderer:
            self.log_renderer.add()

    def save_state(self):
        return self.p.saveState()

    def restore_state(self, state_uid):
        self.p.restoreState(stateId=state_uid)

    def close(self):
        self.p.disconnect()


class Body(object):
    """Interface to a multibody simulated in PyBullet.

    Attributes:
        uid: The unique id of the body within the physics server.
        name: The name of the body.
        joints: A dict mapping joint names to Joint objects.
        links: A dict mapping link names to Link objects.
    """

    def __init__(self, physics_client, body_uid, scale):
        self.p = physics_client
        self.uid = body_uid
        self.scale = scale
        self.name = self.p.getBodyInfo(self.uid)[1].decode("utf-8")
        self.joints, self.links = {}, {}
        for i in range(self.p.getNumJoints(self.uid)):
            joint_info = self.p.getJointInfo(self.uid, i)
            joint_name = joint_info[1].decode("utf8")
            self.joints[joint_name] = Joint(self.p, self.uid, i)
            link_name = joint_info[12].decode("utf8")
            self.links[link_name] = Link(self.p, self.uid, i)

    @classmethod  # 不需要实例化也可调用的类方法, 参数 cls 是语法要求
    def from_urdf(cls, physics_client, urdf_path, pose, scale):
        body_uid = physics_client.loadURDF(
            str(urdf_path),
            pose.translation,
            pose.rotation.as_quat(),
            globalScaling=scale,
        )
        return cls(physics_client, body_uid, scale)

    def get_pose(self):
        pos, ori = self.p.getBasePositionAndOrientation(self.uid)
        return Transform(Rotation.from_quat(ori), np.asarray(pos))

    def set_pose(self, pose):
        self.p.resetBasePositionAndOrientation(
            self.uid, pose.translation, pose.rotation.as_quat()
        )

    def get_velocity(self):
        linear, angular = self.p.getBaseVelocity(self.uid)
        return linear, angular


class Link(object):
    """Interface to a link simulated in Pybullet.

    Attributes:
        link_index: The index of the joint.
    """

    def __init__(self, physics_client, body_uid, link_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.link_index = link_index

    def get_pose(self):
        link_state = self.p.getLinkState(self.body_uid, self.link_index)
        pos, ori = link_state[0], link_state[1]
        return Transform(Rotation.from_quat(ori), pos)


class Joint(object):
    """Interface to a joint simulated in PyBullet.

    Attributes:
        joint_index: The index of the joint.
        lower_limit: Lower position limit of the joint.
        upper_limit: Upper position limit of the joint.
        effort: The maximum joint effort.
    """

    def __init__(self, physics_client, body_uid, joint_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.joint_index = joint_index

        joint_info = self.p.getJointInfo(body_uid, joint_index)
        self.lower_limit = joint_info[8]
        self.upper_limit = joint_info[9]
        self.effort = joint_info[10]

    def get_position(self):
        joint_state = self.p.getJointState(self.body_uid, self.joint_index)
        return joint_state[0]

    def set_position(self, position, kinematics=False):
        if kinematics:
            self.p.resetJointState(self.body_uid, self.joint_index, position)
        self.p.setJointMotorControl2(
            self.body_uid,
            self.joint_index,
            pybullet.POSITION_CONTROL,
            targetPosition=position,
            force=self.effort,
        )


class Constraint(object):
    """Interface to a constraint in PyBullet.

    Attributes:
        uid: The unique id of the constraint within the physics server.
    """

    def __init__(
        self,
        physics_client,
        parent,
        parent_link,
        child,
        child_link,
        joint_type,
        joint_axis,
        parent_frame,
        child_frame,
    ):
        """
        Create a new constraint between links of bodies.

        Args:
            parent:
            parent_link: None for the base.
            child: None for a fixed frame in world coordinates.

        """
        self.p = physics_client
        parent_body_uid = parent.uid
        parent_link_index = parent_link.link_index if parent_link else -1
        child_body_uid = child.uid if child else -1
        child_link_index = child_link.link_index if child_link else -1

        self.uid = self.p.createConstraint(
            parentBodyUniqueId=parent_body_uid,
            parentLinkIndex=parent_link_index,
            childBodyUniqueId=child_body_uid,
            childLinkIndex=child_link_index,
            jointType=joint_type,
            jointAxis=joint_axis,
            parentFramePosition=parent_frame.translation,
            parentFrameOrientation=parent_frame.rotation.as_quat(),
            childFramePosition=child_frame.translation,
            childFrameOrientation=child_frame.rotation.as_quat(),
        )

    def change(self, **kwargs):
        self.p.changeConstraint(self.uid, **kwargs)


class Contact(object):
    """Contact point between two multibodies.

    Attributes:
        point: Contact point.
        normal: Normal vector from ... to ...
        depth: Penetration depth
        force: Contact force acting on body ...
    """

    def __init__(self, bodyA, bodyB, point, normal, depth, force):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.point = point
        self.normal = normal
        self.depth = depth
        self.force = force


class Camera(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.

    Attributes:
        intrinsic: The camera intrinsic parameters.
    """

    def __init__(self, physics_client, intrinsic, near, far):
        self.intrinsic = intrinsic
        self.near = near
        self.far = far
        self.proj_matrix = _build_projection_matrix(intrinsic, near, far)
        self.p = physics_client

    def render(self, extrinsic):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.as_matrix()
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")  # 变成一维数组 列优先
        gl_proj_matrix = self.proj_matrix.flatten(order="F")

        result = self.p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER,
        )
        rgb, z_buffer, segmentation = result[2][:, :, :3], result[3], result[4]
        depth = (
            1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )
        return rgb, depth, segmentation


def _build_projection_matrix(intrinsic, near, far):
    perspective = np.array(
        [
            [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
            [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho


class OfflineRenderer():
    """用于 Pybullet 的离线渲染器 (每次世界 step 的时候都会保存图片)

    Attributes:
        camera: The camera used to render the scene.
        root: The root directory to save the rendered images/videos.
        interval: The interval between two consecutive frames.

    Example:
    >>> sim = ClutterRemovalSim("pile", "pile/train", gui=False,
                            log_render=True, save_dir=Path("./results/"))
    >>> sim.world.log_renderer.enable()
    >>> sim.reset(object_count=5)
    >>> sim.world.log_renderer.export_video()
    """
    def __init__(self, camera, root, interval=10):
        self.camera = camera
        self.root = root
        Path(self.root).mkdir(parents=True, exist_ok=True)
        self.naming_index = 0  # 保存图片起始索引
        self.interval = interval  # 保存图片间隔
        self.interval_count = 0
        defalut_extrinsic = np.array([[ 1.    ,  0.    ,  0.    , -0.15  ],
                                      [ 0.    , -0.5   , -0.866 ,  0.1616],
                                      [-0.    ,  0.866 , -0.5   ,  0.5201],
                                      [ 0.    ,  0.    ,  0.    ,  1.    ]])
        self.reset(defalut_extrinsic)
        self.disable()

    def add(self):
        if self.is_enable:
            self.interval_count += 1
            if self.interval_count == self.interval:
                self.interval_count = 0
                result = self.camera.render(self.extrinsic)
                self.frames.append(result[0])

    def enable(self):
        self.is_enable = True

    def disable(self):
        self.is_enable = False

    def reset(self, extrinsic=None):
        """清空缓存 并设置外参, 不影响 enable 状态

        Args:
            extrinsic: vgn.utils.transform.Transform
                       or 4*4 matrix
        """
        self.frames = []
        self.naming_index = 0
        if extrinsic is not None:
            if type(extrinsic) == Transform:
                self.extrinsic = extrinsic
            elif extrinsic.shape == (4, 4):
                self.extrinsic = Transform.from_matrix(extrinsic)
            else:
                raise ValueError("extrinsic must be a Transform object or 4*4 matrix")

    def _export_image(self, image, naming="index"):
        '''保存图片
        naming: time 按照时间命名; index 按照序号命名; 其他按照输入命名
        '''
        if self.is_enable:
            filename = ""
            if naming == "time":
                filename += datetime.now().strftime('%H%M%S')
            elif naming == "index":
                filename += str(self.naming_index).zfill(4)
                self.naming_index += 1
            else:
                filename += naming
            filename += ".jpg"
            filepath = str(Path(self.root) / filename)
            imwrite(filepath, image)  # 非中文路径保存图片
            # cv2.imencode('.jpg', image)[1].tofile(filepath)  # 中文路径保存图片
            # print("image saved successfuly at", filepath)

    def export_images(self, naming="index"):
        '''保存图片序列
        naming: time 按照时间命名; index 按照序号命名; 其他按照输入命名
        '''
        if self.is_enable:
            if len(self.frames) == 0:
                print("export failed, empty frames")
                return 0

            for image in self.frames:
                self._export_image(image, naming)

            print(len(self.frames), "images saved successfuly at", self.root)

            self.reset()

    def export_video(self, naming="time", format="mp4", reset=True):
        '''保存图片序列到视频, 时间命名
        Args:
            naming: 文件命名, 默认按照时间命名
            format: 视频格式, 支持 avi mp4 (需要 ffmpeg 转 h264 才能在 VSCode 中播放)
        '''
        if self.is_enable:
            if len(self.frames) == 0:
                print("export failed, empty frames")
                return 0

            frame_size = (self.frames[0].shape[1], self.frames[0].shape[0])
            fps = 20
            if naming == "time":  # 命名
                filename = datetime.now().strftime('%H%M%S')
            else:
                filename = naming
            if format == "avi":  # 格式
                video_dir = os.path.join(self.root, filename + '.avi')
                fourcc = VideoWriter_fourcc(*'MJPG')
            elif format == "mp4":
                video_dir = os.path.join(self.root, filename + '.mp4')
                fourcc = VideoWriter_fourcc(*'mp4v')
            elif format == "h264":
                video_dir = os.path.join(self.root, filename + '.avi')
                fourcc = VideoWriter_fourcc(*'XVID')
            else:
                raise ValueError("format must be avi or mp4 or h264")

            videowriter = VideoWriter(video_dir, fourcc, fps, frame_size)
            for frame in self.frames:  # 写入文件
                videowriter.write(frame)
            videowriter.release()

            # 依赖 ffmpeg
            # if format == "h264":
            #     os.system("ffmpeg -i {} -vcodec h264 {}".format(video_dir, video_dir[:-4] + ".mp4"))
            #     os.remove(video_dir)

            print("video saved successfuly at", video_dir)
            if reset:
                self.reset()

    def _export_video_imageio(self, naming="time", format="h264"):
        '''使用 imageio 的实现, 也需要 ffmpeg'''

        if self.is_enable:
            import imageio
            if len(self.frames) == 0:
                print("export failed, empty frames")
                return 0

            if naming == "time":  # 命名
                filename = datetime.now().strftime('%H%M%S')
            else:
                filename = naming

            video_dir = os.path.join(self.root, filename + '.mp4')
            writer = imageio.get_writer(video_dir, fps=20)
            for frame in self.frames:  # 写入文件
                writer.append_data(frame)
            writer.close()

            print("video saved successfuly at", video_dir)
            self.reset()
