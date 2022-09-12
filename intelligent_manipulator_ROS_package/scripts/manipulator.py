import rospy
import numpy as np
from rospy.numpy_msg import numpy_msg
from time import sleep
from math import radians
from random import randrange
from std_msgs.msg import Float64
from open_manipulator_msgs.srv import *
from open_manipulator_msgs.msg import *
from intelligent_manipulator.msg import *

class Manipulator:
    def __init__(self):
        self.n_joints = 4
        self.joint_angles = {"joint1": None,
                             "joint2": None,
                             "joint3": None,
        }
        self.position = None
        self.lengths = np.array([7.7, 12.8, 2.4, 12.4, 12.6])
        self.min_angles_d = [-45, -45, -45, -45]
        self.max_angles_d = [45, 0, 45, 45]
        self.min_angles = [radians(n) for n in self.min_angles_d]
        self.max_angles = [radians(n) for n in self.max_angles_d]
        self.default_time = 0.2

    def position_callback(self, data):
        position = np.array([data.pose.position.x,
                            data.pose.position.y,
                            data.pose.position.z]
        )
        self.position = position

    def get_position(self):
        rospy.Subscriber("/gripper/kinematics_pose",
                          KinematicsPose,
                          self.position_callback
        )
        while self.position is None:
            print("Getting gripper position:", self.position)
        print(f"Position got! {self.position[0]:.2f} {self.position[1]:.2f} {self.position[2]:.2f}")
        return self.position

    def set_position(self, position):
        x, y, z = position
        service_name = "/goal_task_space_path_position_only"
        rospy.wait_for_service(service_name)
        try:
            set_position = rospy.ServiceProxy(
                service_name, SetKinematicsPose
            )
            arg = SetKinematicsPoseRequest()
            arg.end_effector_name = "gripper"
            arg.kinematics_pose.pose.position.x = x
            arg.kinematics_pose.pose.position.y = y
            arg.kinematics_pose.pose.position.z = z
            arg.path_time = self.default_time
            resp1 = set_position(arg)
            print("Position set!", position)
            return resp1
        except rospy.ServiceException as e:
            print("Position not set! Service call failed: %s"%e)
            return False
        sleep(self.default_time)


    def joint_angle_callback(self, data, args):
        joint_name = args
        joint_angle = data.data
        self.joint_angles[joint_name] = joint_angle

    def get_joint_angles(self):
        rospy.Subscriber("/joint1_position/command",
                         Float64,
                         self.joint_angle_callback,
                         ("joint1")
        )
        while self.joint_angles["joint1"] is None:
            print("Getting joint1", self.joint_angles["joint1"])

        rospy.Subscriber("/joint2_position/command",
                         Float64,
                         self.joint_angle_callback,
                         ("joint2")
        )
        while self.joint_angles["joint2"] is None:
            print("Getting joint2", self.joint_angles["joint2"])

        rospy.Subscriber("/joint3_position/command",
                         Float64,
                         self.joint_angle_callback,
                         ("joint3")
        )
        while self.joint_angles["joint3"] is None:
            print("Getting joint3", self.joint_angles["joint3"])

        rospy.Subscriber("/joint4_position/command",
                         Float64,
                         self.joint_angle_callback,
                         ("joint4")
        )
        while self.joint_angles["joint4"] is None:
            print("Getting joint4", self.joint_angles["joint4"])

        angles = self.joint_angles.values()
        angles = list(angles)
        print(f"Angles got! {angles[0]:.2f} {angles[1]:.2f} {angles[2]:.2f} {angles[3]:.2f}")
        return np.array(angles)        
    
    def set_joint_angles(self, angles):
        service_name = "/goal_joint_space_path"
        rospy.wait_for_service(service_name)
        try:
            set_joint = rospy.ServiceProxy(
                service_name, SetJointPosition
            )
            arg = SetJointPositionRequest()
            arg.joint_position.joint_name = [
                "joint1", "joint2", "joint3", "joint4"
            ]
            arg.joint_position.position =  [
                angles[0], angles[1], angles[2], angles[3]
            ]
            arg.path_time = self.default_time
            resp1 = set_joint(arg)
            pos = arg.joint_position.position
            print(f"Joint angles set! {pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f} {pos[3]:.2f}")
            return resp1
        except rospy.ServiceException as e:
            print("Joint angles not set! Service call failed: %s"%e)
            return False
        sleep(self.default_time)

    def clip_angles(self, angles):
        for i in range(self.n_joints):
            if angles[i] < self.min_angles[i]:
                angles[i] = self.min_angles[i]
            elif angles[i] > self.max_angles[i]:
                angles[i] = self.max_angles[i]
        return angles


def callback(data, args):
    manipulator = args[0]
    pub = args[1]
    action = data.action
    print(f"Action received: {action[0]:.2f} {action[1]:.2f} {action[2]:.2f} {action[3]:.2f}")

    angles = manipulator.get_joint_angles()
    angles_ = angles + action
    angles_ =  manipulator.clip_angles(angles_)
    manipulator.set_joint_angles(angles_)

    sleep(manipulator.default_time)
    print()

def manipulator_subscriber(manipulator, pub):
    rospy.Subscriber("action_topic",
                      numpy_msg(Action),
                      callback,
                      (manipulator, pub),
                      queue_size=1
    )
    rospy.spin()

def initiate_manipulator():
    node = rospy.init_node("manipulator_node", anonymous=True)
    print("Manipulator node initiated!")
    manipulator = Manipulator()
    print("Manipulator initiated!")
    pub = rospy.Publisher("position_topic", Position)
    print("Manipulator publisher created!")
    try:
        manipulator_subscriber(manipulator, pub)
    except rospy.ROSInterruptException:
        pass

initiate_manipulator()