# Monitor keyboard, send signal to /object_pose and /predefined_command

import rospy
from std_msgs.msg import Float64MultiArray, String
from pynput import keyboard
import tf2_ros
from tf.transformations import quaternion_from_euler, euler_from_quaternion

xyz_step = 0.05
rpy_step = 0.1


def on_press(key):
    if key == keyboard.Key.space:
        # 控制抓手
        global gripper_state
        gripper_state = not gripper_state
        command_pub.publish('close' if gripper_state else 'open')
        print('\033[1;32m' + '\nSend grasp command:\n' + '\033[0m', 'close' if gripper_state else 'open')
    elif key == keyboard.Key.esc:
        # 复位关节
        command_pub.publish('reset')
        # 发送对应位置的空间坐标（否则会一直规划）
        targetpose.pos.data = [0.307, -0.002, 0.484, 1.000, -0.004, -0.015, 0.006]
        pose_pub.publish(targetpose.pos)
        print('\033[1;31m' + '\nReturn to safe pose!\n' + '\033[0m')
    elif 'char' in dir(key) and key.char in 'mn':
        global xyz_step, rpy_step
        xyz_step *= 1.5 if key.char == 'm' else 0.6666667
        rpy_step *= 1.5 if key.char == 'm' else 0.6666667
        print('\033[1;32m' + '\nChange step size:\n' + '\033[0m',
              'xyz_step =', xyz_step, 'rpy_step =', rpy_step, '\n')
    elif 'char' in dir(key) and key.char in 'wasdqeujikolr':
        # 控制位姿
        targetpose.move(key.char, xyz_step, rpy_step)
        pose_pub.publish(targetpose.pos)
        print('\033[1;32m' + '\nSend new pose:\n' + '\033[0m', targetpose.pos, '\n')


def on_release(key):
    pass


class TargetPose:
    def __init__(self, init_pose):
        self.pos = Float64MultiArray()
        self.pos.data = [0, 0, 0, 0, 0, 0, 0]
        self.init_pose = init_pose
        self.reset_pose()

    def reset_pose(self):
        self.pos.data[0] = self.init_pose.transform.translation.x
        self.pos.data[1] = self.init_pose.transform.translation.y
        self.pos.data[2] = self.init_pose.transform.translation.z
        self.pos.data[3] = self.init_pose.transform.rotation.x
        self.pos.data[4] = self.init_pose.transform.rotation.y
        self.pos.data[5] = self.init_pose.transform.rotation.z
        self.pos.data[6] = self.init_pose.transform.rotation.w

    def move(self, key, xyz_step=0.05, rpy_step=0.1):
        '''解析key，控制目标相对位移/姿态/复位
        '''
        # xyz
        if key == 'w':
            self.pos.data[0] += xyz_step
        elif key == 's':
            self.pos.data[0] -= xyz_step
        elif key == 'a':
            self.pos.data[1] += xyz_step
        elif key == 'd':
            self.pos.data[1] -= xyz_step
        elif key == 'q':
            self.pos.data[2] += xyz_step
        elif key == 'e':
            self.pos.data[2] -= xyz_step
        # rpy
        elif key in ['u', 'i', 'o', 'j', 'k', 'l']:
            rpl = list(euler_from_quaternion(self.pos.data[3:7]))
            if key == 'u':
                rpl[0] += rpy_step
            elif key == 'j':
                rpl[0] -= rpy_step
            elif key == 'i':
                rpl[1] += rpy_step
            elif key == 'k':
                rpl[1] -= rpy_step
            elif key == 'o':
                rpl[2] += rpy_step
            elif key == 'l':
                rpl[2] -= rpy_step
            self.pos.data[3:7] = list(quaternion_from_euler(*rpl))
        # reset
        elif key == 'r':
            self.reset_pose()


if __name__ == '__main__':
    rospy.init_node('keyboard_control')
    rate = rospy.Rate(10)

    pose_pub = rospy.Publisher('object_pose', Float64MultiArray, queue_size=1)
    command_pub = rospy.Publisher('predefined_command', String, queue_size=1)
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    print("Reading current position...")

    got_init_pose = False
    while not got_init_pose:
        try:
            init_pose = tfBuffer.lookup_transform('panda_link0', 'panda_hand_tcp', rospy.Time())
            print('\033[1;32m' + '\nGet init_pose:\n' + '\033[0m', init_pose)
            targetpose = TargetPose(init_pose)
            got_init_pose = True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

    print("\n\033[1;32mInstructions:\033[0m",
          "\nUse 'wasdqe' and 'ujikol' keys to control the pose.",
          "\nUse 'm' and 'n' keys to change the step size.",
          "\nPress 'Esc' to reset the pose."
          "\nUse Space to control the gripper.",
          "\nPress Ctrl-C to quit.",
          "\n--------------------------------------------------\n")

    gripper_state = True

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    while not rospy.is_shutdown():
        rate.sleep()
