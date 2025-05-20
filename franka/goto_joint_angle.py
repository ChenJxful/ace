import argparse

from control.franka_control import OSC_Control, Joint_Control
from deoxys.franka_interface import FrankaInterface
from deoxys import config_root
from pre_defined_pose import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, default="reset_position")
    args = parser.parse_args()

    joint_controller_type = "JOINT_POSITION"
    controller_config = "charmander.yml"  # 控制模式 和 配置文件
    robot_interface = FrankaInterface(
        config_root + f"/{controller_config}", use_visualizer=False)
    joint_controller = Joint_Control(robot_interface, joint_controller_type)

    joint_controller.control(globals()[args.t], grasp=False)
