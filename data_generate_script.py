import os

os.system(f"python -m data.data_generator ./data/set/data_facing_grasp_10K --observation-type facing --num-grasps 10000 --num-proc 1")
print('\033[1;33m' + f"data_packed_facing_grasp done!" + '\033[0m')

os.system(f"python -m data.data_generator ./data/set/data_side_grasp_10K --observation-type side --num-grasps 10000 --num-proc 1")
print('\033[1;33m' + f"data_packed_side_grasp done!" + '\033[0m')

os.system(f"python -m data.data_generator ./data/set/data_multiview_grasp_20K --observation-type multiview --num-grasps 20000 --num-proc 1")
print('\033[1;33m' + f"data_packed_multiview_grasp done!" + '\033[0m')
