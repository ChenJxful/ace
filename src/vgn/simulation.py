from pathlib import Path
import time

import numpy as np
import pybullet

from vgn.grasp import Label
from vgn.perception import *
from vgn.utils import btsim, workspace_lines
from vgn.utils.transform import Rotation, Transform
from vgn.utils.misc import apply_noise, apply_translational_noise


class ClutterRemovalSim(object):
    def __init__(self, scene, object_set,
                 gui=True, seed=None, add_noise=False, sideview=False,
                 save_dir=None,
                 save_pkl=False, save_freq=8,
                 log_render=True):
        assert scene in ["pile", "packed"]

        self.urdf_root = Path("data/urdfs")
        self.scene = scene
        self.object_set = object_set
        self.discover_objects()

        self.global_scaling = {
            "blocks": 1.67,
            "google": 0.7,
            'google_pile': 0.7,
            'google_packed': 0.7,
        }.get(object_set, 1.0)  # depends on the different object set, default 1.0
        self.gui = gui
        self.add_noise = add_noise
        self.sideview = sideview

        self.rng = np.random.RandomState(seed) if seed else np.random  # 类似随机数种子
        self.world = btsim.BtWorld(gui=self.gui,
                                   dir=save_dir,
                                   save_pkl=save_pkl,
                                   save_freq=save_freq,
                                   log_render=log_render)
        self.gripper = Gripper(self.world)
        self.size = 0.3  # 6 * self.gripper.finger_depth, no specific reason
        intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)


    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    def discover_objects(self):
        root = self.urdf_root / self.object_set
        self.object_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count, rng=None):
        '''Reset the simulation environment.
        add plane, set valid volume, generate objects
        '''
        self.world.reset()  # time step, etc.
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,    # 偏转角
                cameraPitch=-45,  # 俯仰角
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        table_height = self.gripper.finger_depth
        self.place_table(table_height)

        if rng is not None:
            self.reset_legacy(object_count, rng)
        elif self.scene == "pile":
            self.generate_pile_scene(object_count, table_height)
        elif self.scene == "packed":
            self.generate_packed_scene(object_count, table_height)
        else:
            raise ValueError("Invalid scene argument")

    def reset_legacy(self, object_count, rng):
        '''reset code from original active_grasp
        '''
        attempts=10
        center = np.r_[0.15, 0.15, 0.05]
        length = 0.3
        origin = center - np.r_[0.5 * length, 0.5 * length, 0.0]

        urdfs = rng.choice(self.object_urdfs, object_count)  # 从大约16个物体中随机选取4个
        for urdf in urdfs:
            scale = rng.uniform(0.8, 1.0)
            pose = Transform(Rotation.identity(), np.zeros(3))
            body = self.world.load_urdf(urdf, pose, scale=scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z_offset = 0.5 * (upper[2] - lower[2]) + 0.002
            state_id = self.world.p.saveState()
            for _ in range(attempts):
                # Try to place and check for collisions
                ori = Rotation.from_euler("z", rng.uniform(0, 2 * np.pi))  # 随机旋转
                pos = np.r_[rng.uniform(0.2, 0.8, 2) * 0.3, z_offset]  # 场景中随机位置
                self.world.p.resetBasePositionAndOrientation(body.uid, origin + pos, ori.as_quat())  # 把物体从原点移动过来
                self.world.p.stepSimulation()
                if not self.world.p.getContactPoints(body.uid):  # step之后检查碰撞
                    break
                else:
                    self.world.p.restoreState(stateId=state_id)
            else:
                # No placement found, remove the object
                self.world.remove_body(body)

    def draw_workspace(self):
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self, height):
        '''add plane.urdf, set valid volume
        '''
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        self.world.load_urdf(urdf, pose, scale=0.6)

        # define valid volume for sampling grasps
        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        self.lower = np.r_[lx, ly, lz]
        self.upper = np.r_[ux, uy, uz]
        # e.g. [0.02, 0.02, 0.055], [0.28, 0.28, 0.3]

    def generate_pile_scene(self, object_count, table_height):
        '''add box, wait; add object, wait; remove box, wait
        '''
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)

        # drop objects
        urdfs = self.rng.choice(self.object_urdfs, size=object_count)
        for urdf in urdfs:
            rotation = Rotation.random(random_state=self.rng)
            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            pose = Transform(rotation, np.r_[xy, table_height + 0.2])  # 从 1/3-2/3 水平位置、0.2 高度随机丢
            scale = self.rng.uniform(0.8, 1.0)
            self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=1.0)

        # remove box
        self.world.remove_body(box)
        self.remove_and_wait()

    def generate_packed_scene(self, object_count, table_height):
        attempts = 0
        max_attempts = 12

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            urdf = self.rng.choice(self.object_urdfs)
            x = self.rng.uniform(0.08, 0.22)
            y = self.rng.uniform(0.08, 0.22)
            z = 1.0
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            pose = Transform(rotation, np.r_[x, y, z])
            scale = self.rng.uniform(0.7, 0.9)
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.world.step()

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
            attempts += 1

    def acquire_tsdf(self, n, N=None, resolution=40):
        """Render synthetic depth images from n viewpoints and integrate into a TSDF.

        If N is None, the n viewpoints are equally distributed on circular trajectory.

        If N is given, the first n viewpoints on a circular trajectory consisting of N points are rendered.
        """
        tsdf = TSDFVolume(self.size, resolution)  # 40*40*40 大小的空白 TSDF
        high_res_tsdf = TSDFVolume(self.size, 120)

        if self.sideview:
            origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, self.size / 3])
            theta = np.pi / 3.0
        else:
            origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, 0])
            theta = np.pi / 6.0
        r = 2.0 * self.size

        N = N if N else n
        if self.sideview:  # 按照规则生成外参 (比如环绕一周的 或者单个相机的)
            assert n == 1
            phi_list = [- np.pi / 2.0]
        else:
            phi_list = 2.0 * np.pi * np.arange(n) / N
        extrinsics = [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

        timing = 0.0
        for extrinsic in extrinsics:  # 按照外参渲染深度图 并加噪声
            depth_img = self.camera.render(extrinsic)[1]

            # add noise
            depth_img = apply_noise(depth_img, self.add_noise)

            tic = time.time()
            tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)  # 往空白 TSDF 上加入深度图所获得的数据
            timing += time.time() - tic
            high_res_tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.lower, self.upper)
        pc = high_res_tsdf.get_cloud()
        pc = pc.crop(bounding_box)

        return tsdf, pc, timing

    def execute_grasp(self, grasp, remove=True, allow_contact=False):
        '''执行抓取
        '''
        # 抓取前的位姿
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp  # 抓取之前抬升 5cm

        # 抓取后的位姿
        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        self.gripper.reset(T_world_pregrasp)
        self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=False)
        self.gripper.move(0.0)
        self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
        if self.check_success(self.gripper):
            result = Label.SUCCESS, self.gripper.read()
            if remove:
                contacts = self.world.get_contacts(self.gripper.body)
                self.world.remove_body(contacts[0].bodyB)
        else:
            result = Label.FAILURE, self.gripper.max_opening_width

        self.world.remove_body(self.gripper.body)

        if remove:
            self.remove_and_wait()

        return result

    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            # if objects fell out, wait and check again
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        '''Remove objects that fell outside the workspace.
        Returns:
            True if an object was removed, False otherwise.
        '''
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object

    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts = self.world.get_contacts(gripper.body)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res


class Gripper(object):
    """Simulated Panda hand.
    """

    def __init__(self, world):
        self.world = world
        self.urdf_path = Path("data/urdfs/panda/hand.urdf")

        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.022])
        self.T_tcp_body = self.T_body_tcp.inverse()

    def reset(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)  # sets the position of the COM, not URDF link
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.update_tcp_constraint(T_world_tcp)
        # constraint to keep fingers centered
        self.world.add_constraint(
            self.body,
            self.body.links["panda_leftfinger"],
            self.body,
            self.body.links["panda_rightfinger"],
            pybullet.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        ).change(gearRatio=-1, erp=0.1, maxForce=50)
        self.joint1 = self.body.joints["panda_finger_joint1"]
        self.joint1.set_position(0.5 * self.max_opening_width, kinematics=True)
        self.joint2 = self.body.joints["panda_finger_joint2"]
        self.joint2.set_position(0.5 * self.max_opening_width, kinematics=True)

    def update_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )

    def set_tcp(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self, threshold=5):
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def move(self, width):
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read(self):
        width = self.joint1.get_position() + self.joint2.get_position()
        return width
