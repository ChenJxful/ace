# Given experiments log directory, remove empty log files
# "empty": less than 2 ckpt files in <log_dir>/ckpts

import os
import re
import sys
import argparse
import shutil


def remove_empty_logs(log_dir):
    empty_dir_count = 0
    for subdir, dirs, files in os.walk(log_dir):
        for dir in dirs:
            if bool(re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', dir)):  #  or just use os.listdir()
                ckpt_dir = os.path.join(subdir, dir, "ckpts")
                num_ckpts = 0  # 路径不存在也是 0
                if os.path.isdir(ckpt_dir):
                    num_ckpts = len(os.listdir(ckpt_dir))
                if num_ckpts < 5:
                    print('\033[1;31m' + "Removing" + '\033[0m',
                          f"empty log: {os.path.join(subdir, dir)}")
                    if args.iamsure:
                        shutil.rmtree(os.path.join(subdir, dir))
                    empty_dir_count += 1
                else:
                    print(f"Log {os.path.join(subdir, dir)} has",
                          '\033[1;32m' + f"{num_ckpts:>3} ckpts" + '\033[0m' + ". Skip.")
    if empty_dir_count == 0:
        print('\033[1;34m' + 'No empty log directory found.' + '\033[0m')
    else:
        print('\033[1;34m' + f"Removed {empty_dir_count} empty log directories." + '\033[0m')
        if not args.iamsure:
            print('\033[1;34m' + "I'm just kidding." + '\033[0m')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove empty log files')
    parser.add_argument('--log_dir', type=str, help='input experiments log directory')
    parser.add_argument('--iamsure', action='store_true', help='I am sure to remove empty logs')
    args = parser.parse_args()
    log_dir = args.log_dir

    remove_empty_logs(log_dir)
