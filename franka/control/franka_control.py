from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig,transform_utils
from deoxys.utils.input_utils import input2action
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils.config_utils import get_default_controller_config
import numpy as np
from deoxys.utils.transform_utils import pose2mat,mat2pose,pose_in_A_to_pose_in_B,quat2axisangle,axisangle2quat
import time




class FrankaControl():
    def __init__(self, robot_interface, controller_type) -> None:

        self.robot_interface = robot_interface
        self.controller_type = controller_type
        
        self.logger = get_deoxys_example_logger()
        self.controller_cfg = get_default_controller_config(controller_type)
        
    def control(self, action, grasp = False):
        raise NotImplementedError
    
    def close(self):
        self.robot_interface.close()
    
    @property
    def last_state(self):
        return self.robot_interface.last_state
    
    @property
    def last_eef_pos(self):
        while self.robot_interface.state_buffer_size == 0:
            time.sleep(0.5)
        return self.robot_interface.last_eef_pose    
    
    @property
    def last_eef_quat_and_pos(self):
        return self.robot_interface.last_eef_quat_and_pos
    
    @property
    def last_eef_rot_and_pos(self):
        return self.robot_interface.last_eef_rot_and_pos

class Joint_Control(FrankaControl):
    def __init__(self, robot_interface, controller_type):
        super().__init__(robot_interface, controller_type)
    
    def control(self, target_joint_pos,grasp = True):
        action = target_joint_pos + [grasp * 2 - 1]
        while True:
            if len(self.robot_interface._state_buffer) > 0:
                self.logger.info(f"Current Robot joint: {np.round(self.robot_interface.last_q, 3)}")
                self.logger.info(f"Desired Robot joint: {np.round(self.robot_interface.last_q_d, 3)}")
                
                if (
                    np.max(
                        np.abs(
                            np.array(self.robot_interface._state_buffer[-1].q)
                            - np.array(target_joint_pos)
                        )
                    )
                    < 5e-3
                ):
                    break
                
            self.robot_interface.control(
                controller_cfg=self.controller_cfg,
                action=action,
                controller_type=self.controller_type,
            )
    
    def joint_move(self, target_joint, num_steps):
        current_joint = self.robot_interface.last_state.joint_positions
        action = None
        for _ in range(num_steps):
            current_joint = self.robot_interface.last_state.joint_positions
            action = target_joint - current_joint
            self.control(action)
        return action


class OSC_Control(FrankaControl):
    def __init__(self, robot_interface, controller_type, num_steps=40, additional_steps=10, interpolation_method="linear"):
        super().__init__(robot_interface, controller_type)
        
        self.interpolation_method = interpolation_method
        self.num_steps = num_steps
        self.additional_steps = additional_steps
    
    def osc_move(self, target_pose, grasp, num_steps):
        target_pos, target_quat = target_pose
        target_axis_angle = transform_utils.quat2axisangle(target_quat)
        current_rot, current_pos = self.robot_interface.last_eef_rot_and_pos
        action = None
        for _ in range(num_steps):
            current_pose = self.robot_interface.last_eef_pose
            current_pos = current_pose[:3, 3:]
            current_rot = current_pose[:3, :3]
            current_quat = transform_utils.mat2quat(current_rot)
            if np.dot(target_quat, current_quat) < 0.0:
                current_quat = -current_quat
            quat_diff = transform_utils.quat_distance(target_quat, current_quat)
            current_axis_angle = transform_utils.quat2axisangle(current_quat)
            axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
            action_pos = (target_pos - current_pos).flatten() * 10
            action_axis_angle = axis_angle_diff.flatten() * 1
            action_pos = np.clip(action_pos, -0.5, 0.5)  # default 1.0
            action_axis_angle = np.clip(action_axis_angle, -0.2, 0.2)  # default 0.5

            action = action_pos.tolist() + action_axis_angle.tolist() + [grasp * 2 - 1]
            self.logger.info(f"Axis angle action {action_axis_angle.tolist()}")
            # print(np.round(action, 2))
            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=self.controller_cfg,
            )
        return action
    
    def __move_to_target_pose(self,target_delta_pose,grasp,num_steps,num_additional_steps,interpolation_method):
        while self.robot_interface.state_buffer_size == 0:
            self.logger.warn("Robot state not received")
            time.sleep(0.5)

        # target_delta_pos, target_quat = (
        #     target_delta_pose[:3],
        #     target_delta_pose[3:],
        # )

        target_delta_pos, target_delta_axis_angle = (
            target_delta_pose[:3],
            target_delta_pose[3:],
        )
        current_ee_pose = self.robot_interface.last_eef_pose
        current_pos = current_ee_pose[:3, 3:]
        target_pos = np.array(target_delta_pos).reshape(3, 1) + current_pos
        
        
        current_rot = current_ee_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)
        current_axis_angle = transform_utils.quat2axisangle(current_quat)

        target_axis_angle = np.array(target_delta_axis_angle) + current_axis_angle
        self.logger.info(f"Before conversion {target_axis_angle}")
        target_quat = transform_utils.axisangle2quat(target_axis_angle)
        target_pose = target_pos.flatten().tolist() + target_quat.flatten().tolist()

        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat
        target_axis_angle = transform_utils.quat2axisangle(target_quat)
        self.logger.info(f"After conversion {target_axis_angle}")
        current_axis_angle = transform_utils.quat2axisangle(current_quat)

        start_pose = current_pos.flatten().tolist() + current_quat.flatten().tolist()

        self.osc_move(
            (target_pos, target_quat),
            grasp,
            num_steps,
        )
        self.osc_move(
            (target_pos, target_quat),
            grasp,
            num_additional_steps,
        )

    def __move_to_target_quat(self,target_delta_pose,grasp,num_steps,num_additional_steps,interpolation_method):
        while self.robot_interface.state_buffer_size == 0:
            self.logger.warn("Robot state not received")
            time.sleep(0.5)

        target_delta_pos, target_quat = (
            target_delta_pose[:3],
            target_delta_pose[3:],
        )

        # target_delta_pos, target_delta_axis_angle = (
        #     target_delta_pose[:3],
        #     target_delta_pose[3:],
        # )
        current_ee_pose = self.robot_interface.last_eef_pose
        current_pos = current_ee_pose[:3, 3:]
        target_pos = np.array(target_delta_pos).reshape(3, 1) + current_pos
        
        
        current_rot = current_ee_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)
        # current_axis_angle = transform_utils.quat2axisangle(current_quat)

        # target_axis_angle = np.array(target_delta_axis_angle) + current_axis_angle
        # self.logger.info(f"Before conversion {target_axis_angle}")
        # target_quat = transform_utils.axisangle2quat(target_axis_angle)
        # target_pose = target_pos.flatten().tolist() + target_quat.flatten().tolist()

        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat
        target_axis_angle = transform_utils.quat2axisangle(target_quat)
        self.logger.info(f"After conversion {target_axis_angle}")
        current_axis_angle = transform_utils.quat2axisangle(current_quat)

        # start_pose = current_pos.flatten().tolist() + current_quat.flatten().tolist()

        self.osc_move(
            (target_pos, target_quat),
            grasp,
            num_steps,
        )
        self.osc_move(
            (target_pos, target_quat),
            grasp,
            num_additional_steps,
        )

    def control_quat(self,action,grasp = False):
        ''' pos incremental, quat target. by Wenke.
        '''
        self.__move_to_target_quat(action,grasp,num_steps = self.num_steps,\
                num_additional_steps= self.additional_steps,interpolation_method = self.interpolation_method)
    
    def control(self, action,grasp = False):
        
        self.__move_to_target_pose(action,grasp,num_steps = self.num_steps,\
                num_additional_steps= self.additional_steps,interpolation_method = self.interpolation_method)

def test_joint_control():
    controller_type = "JOINT_POSITION"
    controller_config = "charmander.yml"
    robot_interface = FrankaInterface(
        config_root + f"/{controller_config}", use_visualizer=False)
    controller = Joint_Control(robot_interface, controller_type)
    
    cur_quat, cur_pos = controller.last_eef_quat_and_pos
    
    cur_state = controller.last_state
    
    # reset_joint_positions = [
    #     0.09162008114028396,
    #     0.3826458111314524,
    #     -0.01990020486871322,
    #     -2.2732269941140346,
    #     -0.01307073642274261,
    #     3.50396583422025,
    #     0.8480939705504309,
    # ]
    reset_joint_positions = [0.06892712901558792, -1.4553648041340341, -0.03624805815574314, -2.567270957047893, -0.21795883350239859, 1.985748862451977, 0.828087004284064]
    controller.control(reset_joint_positions, grasp = False)

    a = [0.07144812651707072, 0.2831797722557135, -0.11681363324994308, -2.3334752416573012, -0.019030430297526962, 3.3266541982473425, 1.0238853632019358]
    controller.control(a, grasp = False)
    
        # action = reset_joint_positions + [-1.0]
    # print(len(controller.robot_interface._state_buffer))
    
    # while True:
    #     if len(controller.robot_interface._state_buffer) > 0:
    #         controller.logger.info(f"Current Robot joint: {np.round(controller.robot_interface.last_q, 3)}")
    #         controller.logger.info(f"Desired Robot joint: {np.round(controller.robot_interface.last_q_d, 3)}")
            
    #         if (
    #             np.max(
    #                 np.abs(
    #                     np.array(controller.robot_interface._state_buffer[-1].q)
    #                     - np.array(reset_joint_positions)
    #                 )
    #             )
    #             < 1e-3
    #         ):
    #             break
    # controller.control(reset_joint_positions, grasp = False)
    controller.close()

def test_osc_control():

    controller_type = "OSC_POSE"
    controller_config = "charmander.yml"
    robot_interface = FrankaInterface(
        config_root + f"/{controller_config}", use_visualizer=False)
    controller = OSC_Control(robot_interface, controller_type)
    
    while controller.robot_interface.state_buffer_size == 0:
        controller.logger.warn("Robot state not received")
        time.sleep(0.5)
    
    cur_quat, cur_pos = controller.last_eef_quat_and_pos
    print(cur_pos)
    print(cur_quat)
    
    # target_quat = np.array([ 0.58876944, -0.6990674,   0.27874663,  0.29488218])
    
    
    # controller.osc_move((cur_pos,target_quat),grasp = False, num_steps = 40)
    print(controller.last_state.q)
    POSE_CAMERA_IN_EE = [0.0467162 , -0.0388289 , -0.0419308]  
    QUAT_CAMERA_IN_EE = [-0.008896206339478014, 0.001529932861133454, 0.7130445599416992, 0.7010606053371948]
    CAMERA_IN_EE_MAT = pose2mat([POSE_CAMERA_IN_EE, QUAT_CAMERA_IN_EE])
    
    ee_in_base_mat = pose2mat((cur_pos[0],cur_quat))
    
    print("camera in ee mat is:",CAMERA_IN_EE_MAT)
    
    print("ee in base mat is:",ee_in_base_mat)
    # delta_pos =[
    #     [0.2, 0.1, 0.2,0,0,0],
    #     [-0.2, 0.2, -0.1,0,0,0],
    #     [0.2, -0.1, -0.2,0,0,0],
    #     [-0.1, -0.1, -0.2,0,0,0],
    #     [-0.2, -0.1, 0.1,0,0,0],
    #     [0.1, 0.1, 0.14,0,0,0],
    # ] 
    
    
    
    # for i in range(6):
        
    #     controller.control(delta_pos[i])
        
    #     print("current pose is:",controller.robot_interface.last_eef_pose)
    
if __name__ == "__main__":
    # test_osc_control()
    test_joint_control()