# Abstract: 用于检查网络中间结果的工具函数
import os
import torch
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from models.networks import *
from utils.misc import EasyDict
from utils.visualization import *
from data.data_loader import create_train_val_loaders
import matplotlib.pyplot as plt
from matplotlib import colors
from models.networks import Runner
from grasp import get_ckpts


class PrepareNet:
    def __init__(self, experiment_name, ckpt_index=None):
        '''从指定的实验以及模型编号中加载网络
        '''
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base=None,
                   config_path="../experiments/" + experiment_name + "/configs",
                   job_name="test_app")
        if ckpt_index is None:
            ckpt_index = -1
        ckpt_name = get_ckpts(experiment_name)[ckpt_index]
        print(f"Using model from \n {ckpt_name}")
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.cfg = compose(config_name="config", overrides=[f"load_path='{ckpt_name}'",
                                                            f"device={device}"])

        self.net = Runner(self.cfg).load_network(self.cfg)
        self.net.eval()

    def _get_net(self):
        return self.net

    def __call__(self, batch, **kwargs):
        '''利用输入 batch 进行一轮 inference
        '''
        if "new_resolution" in kwargs:
            self.resolution = kwargs["new_resolution"]
        with torch.no_grad():
            prediction = self.net(batch, **kwargs)
        self.batch = batch
        self.batch_size = len(batch.tsdf)
        self.prediction = prediction
        return prediction

    def _get_img_gt(self):
        return self.batch["depth_img"][:, 0, :].cpu()

    def _get_img_pred(self):
        return self.prediction["rendered_depth"].reshape(-1, self.resolution[1], self.resolution[0]).cpu()

    def draw_img_gt_pred(self, fig, share_colorbar=True):
        '''可视化深度图（即用于 geometry 监督的数据）
        '''
        img_pred = self._get_img_pred()
        img_gt = self._get_img_gt()

        axes = assign_axis_group(fig, img_pred.shape[0], 3, 2, subplot_kw={'xticks': [], 'yticks': []})
        if share_colorbar:
            norm = colors.Normalize(vmin=max(min(img_gt.min(), img_pred.min()), 0.4),
                                    vmax=min(max(img_gt.max(), img_pred.max()), 1.0))  # 放在统一坐标系
            kwargs = {"norm": norm, "cmap": "GnBu_r"}
        else:
            kwargs = {"cmap": "GnBu_r"}
        for i in range(img_pred.shape[0]):
            _img_gt = axes[0, i].imshow(img_gt[i], **kwargs)
            axes[0, i].set_title(f"gt {i}", fontsize=8)
            _img_pred = axes[1, i].imshow(img_pred[i], **kwargs)
            axes[1, i].set_title(f"pred {i}", fontsize=8)
        if share_colorbar:
            fig.colorbar(_img_gt, ax=axes.ravel().tolist(), shrink=0.5)
        else:
            fig.colorbar(_img_gt, ax=axes[0].ravel().tolist(), shrink=0.9)
            fig.colorbar(_img_pred, ax=axes[1].ravel().tolist(), shrink=0.9)
            # l_ax = axes.ravel().tolist()  # 另一种有点笨的写法
            # fig.colorbar(_img_gt, ax=l_ax[:len(l_ax) // 2], shrink=0.5)
            # fig.colorbar(_img_pred, ax=l_ax[len(l_ax) // 2:], shrink=0.5)

    def _get_sdf(self):
        '''恢复 padding
        '''
        original_size = 0.3
        pts = self.prediction.points.reshape(self.batch_size, -1, 64, 3)
        pts = pts * original_size * (1 + self.cfg.decoder_padding)
        pts += original_size * 0.5
        pts_sdf = self.prediction.sdf.reshape(self.batch_size, -1, 64, 1)
        return pts, pts_sdf

    def draw_sdf_animation(self, fig, scene_index=0):
        '''环绕工作空间的动画 展示采样点的 SDF 数值
        '''
        pts, pts_sdf = self._get_sdf()

        def my_plt(ax):
            draw_floor_grid(ax)
            # plot_3D_points(ax, pts[0, [0,1000,1536,2000,3000], ...].cpu(),
            #                pts_sdf[0, [0,1000,1536,2000,3000], ...].cpu(), down_sample=1000)
            plot_3D_points(ax, pts[scene_index].cpu(),
                           pts_sdf[scene_index].cpu(), down_sample=1000)

            # 画相机
            # cmap = get_cmap(self.batch["camera_extrinsic"].shape[0] + 1)
            draw_pyramid(ax,
                         Transform.from_list(list(self.batch["camera_extrinsic"][scene_index, 0])).as_matrix(),
                         # color=cmap(scene_index),
                         intrinsic=self.batch["camera_intrinsic"][scene_index].cpu(),
                         height=0.1,
                         label=f'cam{scene_index}')
            draw_box(ax, [0, 0, 0], [0.3, 0.3, 0.3], alpha=0.05)
            mark_axis_label(ax)
            fix_3d_axis_equal(ax)
            set_axis_range(ax, [-0.15, 0.45, -0.15, 0.45, -0.05, 0.55])

        ax = fig.add_subplot(111, projection='3d')
        my_plt(ax)

        ani = make_view_change_animation(fig, ax, 20, [35, -10], [0, 360])
        return ani

    def _discretize_to_planes(self, discrete_level=0.01):
        '''在 z 轴方向离散化
        '''
        pts, pts_sdf = self._get_sdf()
        pts_z_discritized = torch.round(pts[..., 2] / discrete_level) * discrete_level
        return pts, pts_z_discritized, pts_sdf

    def _show_discritized_sdf(self, fig, pts, pts_z_discritized, scene_index=0):
        '''直方图展示离散化后的效果
        '''
        cmap = get_cmap(pts_z_discritized.shape[1])
        ax = fig.add_subplot(141)
        ax.hist(pts[scene_index, ..., 0].reshape(-1).cpu(), bins=80)
        ax.set_title("x axis")
        ax = fig.add_subplot(142)
        ax.hist(pts[scene_index, ..., 1].reshape(-1).cpu(), bins=80)
        ax.set_title("y axis")
        ax = fig.add_subplot(143)
        ax.hist(pts[scene_index, ..., 2].reshape(-1).cpu(), bins=80)
        ax.set_title("z axis")
        ax = fig.add_subplot(144)
        ax.hist(pts_z_discritized[scene_index, ...].reshape(-1).cpu(), bins=80)
        ax.set_title("discrete z axis")
        fig.tight_layout()

    def draw_sdf_slices_animation(self, fig,
                                  only_inside=True,
                                  with_contour=True,
                                  down_sample=10000,
                                  discrete_level=0.01, scene_index=0):
        '''z 轴分层 展示采样点的 SDF 数值（最好配合 sample_sdf_on_meshgrid）
        '''
        pts, pts_z_discritized, pts_sdf = self._discretize_to_planes(discrete_level)
        norm = colors.Normalize(vmin=pts_sdf.min(), vmax=pts_sdf.max())  # 放在统一坐标系

        def draw_one_layer_sdf(ax, z, norm, down_sample=down_sample):
            '''绘制一层的 SDF'''
            ax.clear()
            ax.set_aspect("equal")
            # 筛选对应层
            layer_filter = torch.isclose(pts_z_discritized[scene_index],
                                         torch.tensor(z, dtype=pts_z_discritized.dtype),
                                         atol=1e-08)
            selected_pts = pts[scene_index][layer_filter].cpu()
            selected_sdf = pts_sdf[scene_index][layer_filter].cpu()
            # 筛选 <0
            if only_inside:
                inside_filter = (selected_sdf < 0)[...,0]  # 注意去掉最后一个维度
                selected_pts = selected_pts[inside_filter]
                selected_sdf = selected_sdf[inside_filter]
            # 下采样
            if down_sample and down_sample < selected_pts.shape[0]:
                sample_index = np.random.randint(0, selected_pts.shape[0], size=down_sample)
                selected_pts = selected_pts[sample_index]
                selected_sdf = selected_sdf[sample_index]
            # 画散点
            img = ax.scatter(selected_pts[:, 0],
                             selected_pts[:, 1],
                             c=selected_sdf[:, 0],
                             cmap="Spectral",
                             s=3,
                             norm=norm)
            # 画等高线
            if with_contour and selected_pts.shape[0] > 3:
                ax.tricontour(selected_pts[:, 0],
                              selected_pts[:, 1],
                              selected_sdf[:, 0], levels=15, linewidths=0.5)

            ax.set_title(f'layer height={z}')
            ax.set_xlim([-0.05, 0.35])
            ax.set_ylim([-0.05, 0.35])
            return img

        def animate(i):
            z = discrete_level * i
            img = draw_one_layer_sdf(ax, z, norm)
            return img

        import matplotlib.animation
        frames_count = int(0.3 / discrete_level + 1)
        fig, ax = plt.subplots()
        ani = matplotlib.animation.FuncAnimation(fig, animate, frames=frames_count)

        # 第一帧加个 colorbar
        img = draw_one_layer_sdf(ax, 0, norm)
        cax = fig.add_axes([ax.get_position().x1 + 0.01,
                            ax.get_position().y0,
                            0.02,
                            ax.get_position().height])
        plt.colorbar(img, cax=cax)  # Similar to fig.colorbar(im, cax = cax)
        return ani

    def draw_surface_from_sdf(self, fig, scene_index=0, sdf_threshold=0.01):
        '''展示采样点中 SDF 数值在 0 附近的点（最好配合 sample_sdf_on_meshgrid）
        '''
        pts, pts_sdf = self._get_sdf()
        # 挑选出 sdf 在 0 附近的点
        pts = pts[scene_index].reshape(-1, 3)
        pts_sdf = pts_sdf[scene_index].reshape(-1, 1)
        pts_filter = torch.isclose(pts_sdf, torch.tensor(0, dtype=pts_sdf.dtype), atol=sdf_threshold).reshape(-1)
        pts_surface = pts[pts_filter]
        pts_surface_sdf = pts_sdf[pts_filter]

        def my_plt(ax):
            draw_floor_grid(ax)
            plot_3D_points(ax, pts_surface.cpu(),
                           pts_surface[:, [2]].cpu(),
                           cmap="GnBu_r",
                           down_sample=10000)

            draw_box(ax, [0, 0, 0], [0.3, 0.3, 0.3], alpha=0.05)
            mark_axis_label(ax)
            fix_3d_axis_equal(ax)
            set_axis_range(ax, [-0.15, 0.45, -0.15, 0.45, -0.05, 0.55])

        ax = fig.add_subplot(111, projection='3d')
        my_plt(ax)

        ani = make_view_change_animation(fig, ax, 20, [35, -10], [0, 360])
        return ani, pts_surface


if __name__ == '__main__':
    # remember to use -m option to run this file
    breakpoint()
    Data = PrepareData(batch_size=2)
    batch = Data()

    experiment_name = "2023-02-24-15-28-35"
    Net = PrepareNet(experiment_name)
    prediction = Net(batch, render_full_image=True, new_resolution=[64, 48],  # [64, 48], [256, 192]
                     sample_sdf_on_meshgrid=True)
    Net.draw_surface_from_sdf(None)
