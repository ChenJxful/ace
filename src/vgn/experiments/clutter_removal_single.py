# Abstract: 用于单个场景抓取测试, 保存 .pkl 同时开启渲染

import collections
import argparse
from datetime import datetime
import os
import uuid

import numpy as np
import pandas as pd
import tqdm

from vgn import io  #, vis
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_mesh_pose_list_from_world, get_scene_from_mesh_pose_list
from vgn.utils.implicit import as_mesh

MAX_CONSECUTIVE_FAILURES = 2


State = collections.namedtuple("State", ["tsdf", "pc"])


def run_multi_scene(num_rounds=40,
                    silence=False,
                    logdir=None,
                    description=None):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    logger = Logger(logdir, description)

    planning_times = []
    total_times = []

    for _ in tqdm.tqdm(range(num_rounds), disable=silence):  # 这是一个不同
        pass


def run_one_scene(grasp_plan_fn, save_dir, scene, object_set,
                  num_objects=5, num_view=6, N=None,
                  seed=1,  # 随机种子
                  sim_gui=False,
                  add_noise=False,
                  sideview=False,
                  resolution=40,
                  silence=False,
                  save_pkl=True,
                  save_freq=8,
                  bullet_log_render=False):
    '''
    Returns: success, cnt, total_objs
        场景内 total_objs 个物体, 算法尝试了 cnt 次, 成功抓取了 success 次
    '''

    sim = ClutterRemovalSim(scene, object_set,
                            gui=sim_gui, seed=seed, add_noise=add_noise, sideview=sideview,
                            save_dir=save_dir, save_pkl=save_pkl, save_freq=save_freq,
                            log_render=bullet_log_render)

    if bullet_log_render:
        sim.world.log_renderer.enable()  # 开启离线渲染

    print(f"Resetting simulation with {num_objects} objects")

    while sim.num_objects == 0:
        sim.reset(num_objects)  # 注意场景可能会生成失败？例如物体靠在 box 上，移去之后落地
        total_objs = sim.num_objects  # 场景内物体数量

    success, cnt, _ = _run_grasp_loop(sim, grasp_plan_fn, resolution, num_view, N)

    left_objs = sim.num_objects  # 最后剩下的物品数量
    print("Scene end. Saving results...")
    if bullet_log_render:
        sim.world.log_renderer.export_video()  # 导出离线渲染的视频
    print(f"Left objects: {left_objs} / {total_objs} ({left_objs / total_objs * 100:.2f}%)")
    return success, cnt, total_objs


def _run_grasp_loop(
    sim,
    grasp_plan_fn,
    resolution,
    num_view,
    N,
    logger=None
):
    cnt = 0  # 尝试次数
    success = 0  # 成功抓取次数
    cons_fail = 0  # 因为连续失败退出的计数（没用上）
    no_grasp = 0  # 因为没有合适的抓取而退出的计数（没用上）
    trial_id = -1  # 不知道什么计数
    last_label = None  # 记录前一次抓取结果
    consecutive_failures = 1  # 记录连续失败

    timings = {}
    while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES:
        # 当桌面上还有物体 且连续失败次数小于最大连续失败次数
        trial_id += 1

        # scan the scene
        tsdf, pc, timings["integration"] = sim.acquire_tsdf(n=num_view, N=N, resolution=40)  # 在当前仿真环境下 从单个侧视图或者多个视图中获取 TSDF 和点云
        state = argparse.Namespace(tsdf=tsdf, pc=pc)  # 创建了一个打包?
        if resolution != 40:
            extra_tsdf, _, _ = sim.acquire_tsdf(n=num_view, N=N, resolution=resolution)
            state.tsdf_process = extra_tsdf

        if pc.is_empty():
            break  # empty point cloud, abort this round TODO this should not happen

        mesh_pose_list = get_mesh_pose_list_from_world(sim.world, "pile/test")
        scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list)
        grasps, scores, timings["planning"], visual_mesh = grasp_plan_fn(state, scene_mesh)
        as_mesh(visual_mesh).export('test.ply')

        # logger.log_mesh(scene_mesh, visual_mesh, f'round_{round_id:03d}_trial_{trial_id:03d}')
        # else:
        # 利用网络, 输入 TSDF 获得抓取
        # grasps, scores, timings["planning"] = grasp_plan_fn(state)

        if len(grasps) == 0:
            no_grasp += 1
            break  # no detections found, abort this round

        # execute grasp
        grasp, score = grasps[0], scores[0]
        label, _ = sim.execute_grasp(grasp, allow_contact=True)
        cnt += 1
        if label != Label.FAILURE:
            success += 1
            print(f"grasp attempt {cnt}: success")
        else:
            print(f"grasp attempt {cnt}: failure")

        # # log the grasp
        # if logger is not None:
        #     logger.log_grasp(round_id, state, timings, grasp, score, label)


        if last_label == Label.FAILURE and label == Label.FAILURE:  # 连续失败
            consecutive_failures += 1
        else:  # 成功或者第一次失败 重置计数
            consecutive_failures = 1

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            cons_fail += 1
        last_label = label

    return success, cnt, timings


class Logger(object):
    def __init__(self, root, description):
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        description = "{}_{}".format(time_stamp, description).strip("_")

        self.logdir = root / description
        self.scenes_dir = self.logdir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.mesh_dir = self.logdir / "meshes"
        self.mesh_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.grasps_csv_path = self.logdir / "grasps.csv"
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "integration_time",
                "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_mesh(self, scene_mesh, aff_mesh, name):
        scene_mesh.export(self.mesh_dir / (name + "_scene.obj"))
        aff_mesh.export(self.mesh_dir / (name + "_aff.obj"))

    def log_grasp(self, round_id, state, timings, grasp, score, label):
        # log scene
        tsdf, points = state.tsdf, np.asarray(state.pc.points)
        scene_id = uuid.uuid4().hex
        scene_path = self.scenes_dir / (scene_id + ".npz")
        np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

        # log grasp
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label = int(label)
        io.append_csv(
            self.grasps_csv_path,
            round_id,
            scene_id,
            qx,
            qy,
            qz,
            qw,
            x,
            y,
            z,
            width,
            score,
            label,
            timings["integration"],
            timings["planning"],
        )


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label
