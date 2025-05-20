from control.franka_control import OSC_Control, Joint_Control
from deoxys.franka_interface import FrankaInterface
from deoxys import config_root

if __name__ == "__main__":
    osc_controller_type = "OSC_POSE"
    controller_config = "charmander.yml"  # 控制模式 和 配置文件
    robot_interface = FrankaInterface(
        config_root + f"/{controller_config}", use_visualizer=False)
    osc_controller = OSC_Control(robot_interface, osc_controller_type, num_steps=1)

    osc_controller.control([0, 0, 0, 0, 0, 0], grasp=False)

    print('last_state: ')
    print(osc_controller.last_state.theta)
