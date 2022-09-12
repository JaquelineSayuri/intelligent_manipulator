#!/usr/bin/env python

import rospy
from rospy.numpy_msg import numpy_msg
from open_manipulator_msgs.srv import *
from intelligent_manipulator.msg import *
from tensorflow import convert_to_tensor
from tensorflow.keras.optimizers import Adam
from intelligent_manipulator.networks import ActorNetwork
from intelligent_manipulator.networks import CriticNetwork
from intelligent_manipulator.networks import ValueNetwork
from math import radians

class Agent:
    def __init__(self, alpha=0.0001, beta=0.0001,
                 max_action=None, n_actions=3):
        self.actor = ActorNetwork(n_actions=n_actions,
                                  name='actor', 
                                  max_action=max_action
        )
        self.critic_1 = CriticNetwork(name='critic_1')
        self.critic_2 = CriticNetwork(name='critic_2')
        self.value = ValueNetwork(name='value')
        self.target_value = ValueNetwork(name='target_value')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.value.compile(optimizer=Adam(learning_rate=beta))
        self.target_value.compile(optimizer=Adam(learning_rate=beta))

    def choose_action(self, observation):
        state = convert_to_tensor([observation])
        actions = self.actor.sample_normal(state)
        return actions[0]

    def load_models(self):
        print('... loading models ...')
        print(self.actor.checkpoint_file)
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.value.load_weights(self.value.checkpoint_file)
        self.target_value.load_weights(
            self.target_value.checkpoint_file
        )


def callback(data, args):
    agent = args[0]
    pub = args[1]
    ob = [f"{e:.2f}" for e in data.observation]
    ob = ' '.join(ob)
    print(f"Observation received: {ob}")
    observation = data.observation
    action = agent.choose_action(observation)
    pub.publish(action)
    print(f"Action published: {action[0]:.2f} {action[1]:.2f} {action[2]:.2f} {action[3]:.2f}\n")

def agent_subscriber(node, agent, pub):
    rospy.Subscriber("observation_topic",
                     numpy_msg(Observation),
                     callback,
                     (agent, pub),
                     queue_size=1)
    rospy.spin()

def initiate_agent(max_action, n_actions):
    node = rospy.init_node("agent_subscriber", anonymous=True)
    print("Agent node initiated!")
    agent = Agent(max_action=max_action, n_actions=n_actions)
    print("Agent initiated!")
    agent.load_models()
    print("Agent models loaded!")
    pub = rospy.Publisher("action_topic", Action)
    print("Agent publisher created!")
    try:
        agent_subscriber(node, agent, pub)
    except rospy.ROSInterruptException:
        pass

initiate_agent(radians(2), 4)