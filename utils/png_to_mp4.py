# 将指定目录下的 png 图片序列转换成 mp4 视频文件

import subprocess
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Convert png sequence to mp4 video')
parser.add_argument('--input_dir', type=Path, help='input directory path')
parser.add_argument('--output_dir', type=str, help='output directory path')
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir

input_pattern = input_dir / '%05d.png'
output_filename = output_dir / 'video.mp4'

command = [
    'ffmpeg',  # 命令名
    '-framerate', '30',  # 每秒帧数
    '-i', input_pattern,  # 输入图片序列的格式
    '-c:v', 'libx264', '-profile:v', 'high', '-crf', '20', '-pix_fmt', 'yuv420p',  # 输出视频的参数
    output_filename  # 输出文件名
]

# 调用 ffmpeg 命令
subprocess.run(command)
