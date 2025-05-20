# 寻找 root 下所有包含 render_log/raw 的路径
# 将 raw 目录下的所有视频文件转换为 H.264 格式, 输出到 render_log 目录下
# .render_log
# ├── ......            # render_log
# └── raw               # render_log/raw
#     ├── 163238.mp4
#     ├── 163337.mp4
#     ├── 163413.avi
#     ├── 163524.avi
#     └── ......

import os
import subprocess
import argparse
import glob


parser = argparse.ArgumentParser(description='Convert video files to H.264 format')
parser.add_argument('root_dir', type=str, help='input directory path')
args = parser.parse_args()
root_dir = args.root_dir


def find_paths_with_substring(root_dir, substring):
    # 递归遍历目录树，并返回所有包含指定子字符串的路径
    paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if substring in dirpath:
            paths.append(dirpath)
    return paths


if not os.path.exists(root_dir):
    raise Exception('Input directory does not exist')

# 寻找所有包含 "render_log/raw" 的路径
target_paths = find_paths_with_substring(root_dir, "render_log/raw")

for path in target_paths:
    input_dir = path
    output_dir = path[:-4]
    print(f"Converting videos in {input_dir} to H.264 format, output to {output_dir} ...")

    # 路径检查 清空已有文件
    for filename in os.listdir(output_dir):
        if filename.endswith(".mp4"):
            os.remove(os.path.join(output_dir, filename))

    # 获取目录下的所有视频文件
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4') or f.endswith('.avi')]

    # 循环处理每个视频文件
    for video_file in video_files:
        # 构造输入和输出文件名
        input_file = os.path.join(input_dir, video_file)
        output_file = os.path.join(output_dir, video_file[:-4] + '.mp4')

        # 构造 ffmpeg 命令
        cmd = ['ffmpeg', '-i', input_file, '-vcodec', 'h264', '-acodec', 'copy', output_file]

        # 调用 ffmpeg 命令
        subprocess.call(cmd)

print("Done! All videos have been converted to H.264 format.")
