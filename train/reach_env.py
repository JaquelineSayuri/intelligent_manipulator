import numpy as np
from math import sin, cos, tan, radians, degrees
from random import randrange
from manipulator import manipulator

class env:
	def __init__(self):
		self.num_states = 6
		self.num_actions = 4
		self.max_action = radians(2)

		self.manipulator = manipulator()

		self.max_steps = 350
		self.step_counter = 0

		#self.goal_boundary = 1
		#self.action_step_size = 0.4
		#self.env_noise = 0.002
		#self.env_noise = 0
		self.margin = 1

	def target_reach(self, state, goal):
		'''
		if (state <= goal + self.margin).all():
			if (state >= goal - self.margin).all():
				return True
		return False
		'''
		distance = abs(state - goal)
		#margin = self.goal_boundary*self.action_step_size
		if (distance <= self.margin).all():
			return True
		else:
			return False

	def distance(self, state, goal):
		dx = goal[0] - state[0]
		dy = goal[1] - state[1]
		dz = goal[2] - state[2]
		distance = (dx**2 + dy**2 + dz**2)**0.5
		return distance

	def reward(self, state, goal):
		distance = self.distance(state, goal)
		reward = -distance**2
		return reward

	def is_done(self, state_, step_counter, goal):
		step_done = step_counter == self.max_steps
		if self.target_reach(state_, goal):
			distance_done = True
		else:
			distance_done = False

		return step_done or distance_done

	def reset(self):
		target_thetas = self.rand_joint_position()
		self.target_position = self.manipulator.direct_kinematics(target_thetas)
		#print('self.target_position ', self.target_position)
		
		initial_thetas = self.rand_joint_position()
		self.manipulator.move(initial_thetas)
		initial_state = self.state()
		#print('initial_state ', initial_state)

		self.step_counter = 0
		self.manipulator.actions_denied = 0
		
		return initial_state

	def rand_joint_position(self):
		# Drawn a position inside the reaching area of the robot
		rand_thetas_d = [randrange(self.manipulator.min_thetas_d[i], self.manipulator.max_thetas_d[i]) for i in range(self.num_actions)]

		rand_thetas = [radians(n) for n in rand_thetas_d]

		return rand_thetas
		
	def step(self, action):
		self.step_counter += 1
		self.manipulator.increment(action, self.num_actions)
		state_ = self.state()
		reward = self.reward(self.manipulator.position, self.target_position)
		done = self.is_done(self.manipulator.position, self.step_counter, self.target_position)

		return state_, reward, done

	def state(self):
		state = np.concatenate((self.manipulator.position, self.target_position))
		#print(state)
		return state