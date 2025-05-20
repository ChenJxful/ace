
import numpy as np
from franka_control import OSC_Control
from robosuite.devices import Keyboard


def input2action(device, controller_type):
    state = device.get_controller_state()
    
    dpos, rotation, raw_drotation, grasp, reset = (
        state["dpos"],
        state["rotation"],
        state["raw_drotation"],
        state["grasp"],
        state["reset"],
    )
    if reset:
        return None, None
    
    drotation = raw_drotation[[1,0,2]]
    
    if controller_type == "OSC_POSE":
        drotation[2] = -drotation[2]
        drotation = drotation * 1.5
        dpos = dpos * 1
        grasp = 1 if grasp else -1
        
        action = np.concatenate([dpos, drotation])
        return action, grasp
    
    else:
        return None, None 
    

def main():
    controller_type = "OSC_POSE"
    controller_config = "charmander.yml"
    controller = OSC_Control(controller_config, controller_type)
    pos_sensitivity = 1.5
    rot_sensitivity = 1.5
    device = Keyboard(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
    
    device.start_control()
    while True:
        action, grasp = input2action(device, controller_type)
        
        controller.control(action)
if __name__ == "__main__":
    main()