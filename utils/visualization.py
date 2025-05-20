# Abstract: 用于 matplotlib 可视化的工具函数

from vgn.utils.transform import Rotation, Transform
from vgn.perception import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from math import ceil, floor
from typing import Union, Tuple


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    Example:
    >>> cmap = get_cmap(count)
    >>> for i in range(count):
    >>>     ...color=cmap(i)...
    '''
    return plt.cm.get_cmap(name, n)


def draw_pyramid(ax, extrinsic, color: Union[str, Tuple[float, float, float]] = 'r',
                 alpha=0.35, height=0.3, intrinsic=None, fov=None, label=None):
    """ 可视化相机, 像素坐标(0,0)画黑点(即左上角), (0,0)-(W,0)画黑线(即上边)
    Args:
        extrinsic: 相机外参 (4, 4)矩阵
        height: 锥高度
        intrinsic: 相机内参 (fx, fy, cx, cy, W, H)
        fov: 宽高比 (W/(2*fx), H/(2*fy))
            [相机视角与内参的关系 - 简书](https://www.jianshu.com/p/935044175ca4)
        注: intrinsic 和 fov 二选一

    original code from: https://github.com/demul/extrinsic2pyramid

    Example:
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(projection='3d')

    >>> intrinsic = [540.0, 540.0, 320.0, 240.0, 640, 480]
    >>> extrinsic = np.array([[-0.7818,  0.6235,  0.    ,  0.0238],
                              [ 0.3117,  0.3909, -0.866 , -0.1054],
                              [-0.54  , -0.6771, -0.5   ,  0.7826],
                              [ 0.    ,  0.    ,  0.    ,  1.    ]])
    >>> draw_pyramid(ax, extrinsic, intrinsic=intrinsic, height=0.1, label='camera')
    """
    if intrinsic is not None:
        fov_w = intrinsic[4] / intrinsic[0] / 2.0
        fov_h = intrinsic[5] / intrinsic[1] / 2.0
    elif fov is not None:
        fov_w = fov[0]
        fov_h = fov[1]
    else:
        fov_w = 0.5
        fov_h = 0.5
    vertex_std = np.array([[0, 0, 0, 1],  # 四棱锥顶点
                           [ height * fov_w, -height * fov_h, height, 1],
                           [ height * fov_w,  height * fov_h, height, 1],
                           [-height * fov_w,  height * fov_h, height, 1],
                           [-height * fov_w, -height * fov_h, height, 1]])

    # 从相机坐标系变换到世界坐标系
    # vertex_transformed = vertex_std @ extrinsic.T  # 原代码这句话是错的
    vertex_transformed = vertex_std @ np.linalg.inv(extrinsic).T

    ax.scatter(vertex_transformed[:4, 0], vertex_transformed[:4, 1], vertex_transformed[:4, 2], color=color, s=10)
    meshes = [[vertex_transformed[0, :-1], vertex_transformed[1, :-1], vertex_transformed[2, :-1]],
              [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
              [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
              [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
              [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
    poly = ax.add_collection3d(
        Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=alpha))

    # 特殊标记
    ax.scatter(vertex_transformed[4, 0],
               vertex_transformed[4, 1],
               vertex_transformed[4, 2],
               color='black', s=10)
    ax.plot3D([vertex_transformed[1, 0], vertex_transformed[4, 0]],    # x 轴坐标
              [vertex_transformed[1, 1], vertex_transformed[4, 1]],    # y 轴坐标
              [vertex_transformed[1, 2], vertex_transformed[4, 2]],    # z 轴坐标
              c='black', linewidth=2)

    if label is not None:
        ax.text(vertex_transformed[0, 0], vertex_transformed[0, 1], vertex_transformed[0, 2], label)


def plot_3D_points(ax, pts, color_axis=None, cmap='jet', with_colorbar=True, colorbar_shrink=0.5,
                   size=10, with_origin=True,
                   axis_lim=None, down_sample=None):
    """
    从矩阵可视化 3D 点, 例如 NeRf sample 的点、物体点云等
    pts: (..., 3)
    color_axis: 选择某个维度作为不同系列, 上不同颜色进行区分/或者直接给 color 赋值 (..., 1)
    """
    if color_axis is not None:
        if type(color_axis) == int:
            colors = np.zeros(pts.shape[:-1])
            colors = colors.swapaxes(0, color_axis)
            for i in range(colors.shape[0]):
                colors[i] = i
            colors = colors.swapaxes(0, color_axis)
        elif color_axis.shape[:-1] == pts.shape[:-1]:
            colors = color_axis
        else:
            raise ValueError('color_axis not valid')
    pts = np.reshape(pts, (-1, 3))
    colors = np.reshape(colors, (-1, 1))

    if down_sample and down_sample < pts.shape[0]:
        sample_index = np.random.randint(0, pts.shape[0], size=down_sample)
        pts = pts[sample_index]
        if color_axis is not None:
            colors = colors[sample_index]

    if with_origin:
        ax.scatter(0, 0, 0, c='black', marker='x', s=100)  # 原点
    if color_axis is not None:
        i = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c=colors, cmap=cmap)
    else:
        i = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size)

    if axis_lim:
        set_axis_range(ax, axis_lim * 3)

    if with_colorbar:
        plt.colorbar(i, shrink=colorbar_shrink)
    return i


def draw_floor_grid(ax, grid_range=[0, 0.3, 0, 0.3], grid_size=0.1, color='k', alpha=0.2):
    """
    绘制地面网格
    grid_range: [x_min, x_max, y_min, y_max]
    """
    x = np.arange(grid_range[0], grid_range[1] + 1e-4, grid_size)
    y = np.arange(grid_range[2], grid_range[3] + 1e-4, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    surf = ax.plot_wireframe(X, Y, Z, color=color, alpha=alpha)
    return surf


def draw_sphere(ax, center, radius, color='b', alpha=0.2):
    """
    绘制球体
    """
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha)


def draw_box(ax, origin, size, facecolors='b', edgecolors='k', alpha=0.2):
    """ 绘制立方体
    用 origin, size 对一个单位立方体进行 transform 变形

    Example:
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(projection='3d')
    >>> draw_box(ax, [0, 0, 0], [0.3, 0.3, 0.3])
    """
    x, y, z = np.indices((2, 2, 2)).astype(np.float64)  # 这个写法非常妙
    x *= size[0]
    y *= size[1]
    z *= size[2]
    x += origin[0]
    y += origin[1]
    z += origin[2]

    filled = np.ones((1, 1, 1))
    ax.voxels(x, y, z, filled=filled, facecolors=facecolors, edgecolors=edgecolors, alpha=alpha)


def mark_axis_label(ax, axis='xyz', fontsize=10):
    """
    标记坐标轴
    """
    if 'x' in axis:
        ax.set_xlabel('X Label', fontsize=fontsize)
    if 'y' in axis:
        ax.set_ylabel('Y Label', fontsize=fontsize)
    if 'z' in axis:
        ax.set_zlabel('Z Label', fontsize=fontsize)


def assign_axis_group(fig, img_num, plot_rows, img_series=1, subplot_kw=None):
    """
    展示多个系列的多张图, 分配在指定行数的空间中, 返回 axes 矩阵
    Args:
        fig: plt.figure
        img_num: 图片数量(单个系列)
        plot_rows: 目标图片行数(单个系列)
        img_series: 图片系列数
    Returns:
        axes: [img_series, img_num] ndarray of axes

    Example:
    >>> fig = plt.figure(dpi=150, figsize=(18, 10))
    >>> axes = assign_axis_group(fig, 7, 2, 2)
    >>> axes[0, 0].plot(x, y)
    """

    # 给定 img_num 和 plot_rows, img_in_a_row 需要同时满足以下两个条件
    # plot_rows * img_in_a_row >= img_num
    # plot_rows * (img_in_a_row-1) < img_num
    img_in_a_row = floor(img_num / plot_rows + 1 - 1e-10)
    actual_img_rows = ceil(img_num / img_in_a_row)

    axes = []

    for i in range(img_series):
        axes_in_a_series = []
        for j in range(img_num):
            axes_in_a_series.append(fig.add_subplot(actual_img_rows * img_series,
                                                    img_in_a_row,
                                                    i * actual_img_rows * img_in_a_row + j + 1,
                                                    **subplot_kw))
        axes.append(axes_in_a_series)
    return np.array(axes)


def fix_3d_axis_equal(ax):
    """ 使三维坐标轴等比例
    """
    ax.set_box_aspect((1, 1, 1))


def set_axis_range(ax, axis_range):
    """ 设置坐标轴范围
    axis_range: [x_min, x_max, y_min, y_max, z_min, z_max]
    """
    ax.set_xlim(axis_range[0], axis_range[1])
    ax.set_ylim(axis_range[2], axis_range[3])
    ax.set_zlim(axis_range[4], axis_range[5])


def make_view_change_animation(fig, ax, frames_count=40, elev=[35, -10], azim=[0, 360]):
    """ 生成视角变化动画
    Args:
        frames_count: 动画帧数
        elev: [起始仰角, 终止仰角]
        azim: [起始方位角, 终止方位角]
    Example:
    >>> %matplotlib inline //or// %matplotlib widget
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(projection='3d')
    >>> plot_something()
    >>> ani = make_view_change_animation(fig, ax)
    >>> ani.save('result.gif', writer='imagemagick', fps=10)
    >>> ani
    """
    import matplotlib.animation
    plt.rcParams["animation.html"] = "jshtml"

    def animate(i):
        ax.view_init(elev[0] + (elev[1] - elev[0]) * i / frames_count,
                     azim[0] + (azim[1] - azim[0]) * i / frames_count)

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=frames_count)
    return ani
