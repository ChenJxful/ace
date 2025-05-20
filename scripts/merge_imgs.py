# 将指定目录下的 png 图片序列拼接起来

import cv2
import numpy as np
import os
import argparse


def merge_images(dir, series=2):
    # 遍历路径下所有图片
    files = [f for f in os.listdir(dir) if f.endswith('png')]
    files.sort()
    img = lambda index: cv2.imread(os.path.join(dir, files[index]))

    assert series in [1, 2], "series must be 1 or 2"
    if series == 1:
        for i in range(len(files)):
            if i == 0:
                merged_image = img(i)
            else:
                merged_image = np.hstack((merged_image, img(i)))

    else:
        # 横向拼接图片
        for i in range(len(files)//2):
            if i == 0:
                img1 = img(i)
                img2 = img(i+len(files)//2)
            else:
                img1 = np.hstack((img1, img(i)))
                img2 = np.hstack((img2, img(i+len(files)//2)))

        # 纵向拼接图片
        merged_image = np.vstack((img1, img2))

    # 保存到文件
    cv2.imwrite(f'{dir}/merged_image.jpg', merged_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str)
    parser.add_argument("--series", type=int, default=2)
    args = parser.parse_args()
    merge_images(args.root, args.series)
    print(f"Merge images in {args.root}")
