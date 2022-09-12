#!/usr/bin/env python

import rospy
from rospy.numpy_msg import numpy_msg
from open_manipulator_msgs.msg import *
from open_manipulator_msgs.srv import *
from intelligent_manipulator.msg import *
import numpy as np
from math import sin, cos, tan, radians, degrees
from random import randrange
from time import sleep
from intelligent_manipulator.direct_kinematics import *
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion
import os
import threading
from intelligent_manipulator.graphs import save_results, get_graphs

class Env:
    def __init__(self):
        self.step_counter = 0
        self.steps_list = []
        self.max_steps = 100
        self.n_episodes = 10
        self.target_name = "red_sphere"
        self.target_position = None
        self.margin = 1
        self.position = None
        
        self.n_joints = 4
        self.min_angles_d = [-45, -45, -45, -45]
        self.max_angles_d = [45, 0, 45, 45]
        self.min_angles = [radians(n) for n in self.min_angles_d]
        self.max_angles = [radians(n) for n in self.max_angles_d]
        self.manipulator_lengths = np.array([7.7, 12.8, 2.4, 12.4, 12.6])
        
        self.default_time = 0.5
        self.delete_target_flag = False

    def reset(self, node):
        if self.delete_target_flag:
            self.delete_target()

        target_angles = self.rand_joint_angles()
        self.target_position = direct_kinematics(
            target_angles,
            self.manipulator_lengths
        )
        self.create_target(self.target_position)
        self.delete_target_flag = True

        sleep(self.default_time)
        self.step_counter = 0
        print("Reset finished!")
        print("  Target position:", self.target_position)

    def create_target(self, target_position):
        target_position = target_position/100
        x = target_position[0]
        y = target_position[1]
        z = target_position[2]

        service_name = "/gazebo/spawn_sdf_model"
        spawn_model = rospy.ServiceProxy(service_name, SpawnModel)
        model_xml = "/home/jaqueline/catkin_ws/src/intelligent_manipulator/model/sphere_r/model.sdf"
        spawn_model(
            model_name=self.target_name,
            model_xml=open(model_xml, 'r').read(),
            robot_namespace="/foo",
            initial_pose=Pose(Point(x,y,z),Quaternion(0,0,0,0)),
            reference_frame="world"
        )
        print("New target created!", target_position)

    def delete_target(self):
        service_name = "/gazebo/delete_model"
        delete_service = rospy.ServiceProxy(service_name, DeleteModel)
        delete_service(self.target_name)
        sleep(self.default_time)
        print("Target deleted!")

    def rand_joint_angles(self):
        min_angles_d = self.min_angles_d
        max_angles_d = self.max_angles_d
        rand_angles_d = [
            randrange(min_angles_d[i], max_angles_d[i]) \
            for i in range(self.n_joints)
        ]
        rand_angles = [radians(n) for n in rand_angles_d]
        return np.array(rand_angles)

    def step(self, position, node):
        self.step_counter += 1
        observation = self.get_observation(position)
        reward = self.reward(
            position,
            self.target_position
        )
        step_done, distance_done = self.is_done(
            position,
            self.target_position,
            self.step_counter
        )
        done = step_done or distance_done
        print("Step finished!", self.step_counter)
        ob = [f"{e:.2f}" for e in observation]
        ob = ' '.join(ob)
        print(f"Observation: {ob}")
        print(f"Reward: {reward:.2f}")
        print("Step_done:", step_done, "    distance_done:", distance_done)
        return observation, reward, done

    def target_reach(self, position, target_position):
        distance = self.distance(position, target_position)
        return distance <= self.margin

    def distance(self, position, target_position):
        dx = target_position[0] - position[0]
        dy = target_position[1] - position[1]
        dz = target_position[2] - position[2]
        distance = (dx**2 + dy**2 + dz**2)**0.5
        return distance

    def reward(self, position, target_position):
        distance = self.distance(position, target_position)
        reward = -distance**2
        return reward

    def is_done(self, position_, target_position, step_counter):
        step_done = step_counter == self.max_steps
        distance_done = self.target_reach(position_, target_position)
        return step_done, distance_done

    def get_observation(self, position):
        observation = np.concatenate(
            (position,
             self.target_position)
        )
        return observation

    def position_callback(self, data):
        position = np.array([data.pose.position.x,
                            data.pose.position.y,
                            data.pose.position.z]
        )
        self.position = position

    def get_position(self):
        rospy.Subscriber("/gripper/kinematics_pose",
                          KinematicsPose,
                          self.position_callback,
                          queue_size=1
        )
        while self.position is None:
            pass
        print(f"Position got! {self.position[0]:.2f} {self.position[1]:.2f} {self.position[2]:.2f}")
        return self.position


def environment_subscriber(node, env, pub):
    T = 30
    score_history = []
    for e in range(env.n_episodes):
        print("Environment reset!")
        env.reset(node)
        total_reward = 0
        path = [list(env.target_position)]
        while env.step_counter < env.max_steps:
            print(f"Episode: {e}")
            position = env.get_position()
            position = position*100
            observation, reward, done = env.step(position, node)
            pub.publish(observation)
            ob = [f"{e:.2f}" for e in observation]
            ob = ' '.join(ob)
            print(f"Observation published: {ob}\n")
            path.append(list(position))
            total_reward += reward
            if done:
                break
            sleep(env.default_time)

        score_history.append(total_reward)
        if len(score_history) > T:
            score_history.pop(0)
        avg_score = np.mean(score_history)
        save_results(f'results/steps_per_episode.txt', env.step_counter)
        save_results(f'results/total_score_per_episode.txt', total_reward)
        save_results(f'results/paths.txt', path)
        save_results(f'results/avg_scores.txt', avg_score)
        sleep(5)

def initiate_environment(env):
    node = rospy.init_node("environment_subscriber", anonymous=True)
    print("Environment node initiated!")
    pub = rospy.Publisher("observation_topic", Observation)
    print("Environment publisher created!")
    environment_subscriber(node, env, pub)
    get_graphs(env.max_steps)


env = Env()
print("Environment initiated!")
initiate_environment(env)