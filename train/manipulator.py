import numpy as np
from math import sin, cos, tan, radians, degrees

class manipulator:
	def __init__(self):
		# link lengths
		self.lengths = np.array([7.7,
								12.8,
								2.4,
								12.4,
								12.6])


		self.min_thetas_d = [-45, -45, -45, -45]
		self.max_thetas_d = [45, 0, 45, 45]

		self.min_thetas = [radians(n) for n in self.min_thetas_d]
		self.max_thetas = [radians(n) for n in self.max_thetas_d]

		# position of the end-effector
		self.position = self.move([0, 0, 0, 0])

	def move(self, thetas):
		self.theta1 = thetas[0]
		self.theta2 = thetas[1]
		self.theta3 = thetas[2]
		self.theta4 = thetas[3]

		self.position = self.direct_kinematics(thetas)

	def increment(self, action, num_actions):
		dtheta1, dtheta2, dtheta3, dtheta4 = action
		theta1_ = self.theta1 + dtheta1
		theta2_ = self.theta2 + dtheta2
		theta3_ = self.theta3 + dtheta3
		theta4_ = self.theta4 + dtheta4

		thetas_ = np.array([theta1_,
						   theta2_,
						   theta3_,
						   theta4_])

		thetas_ = self.clip_actions(thetas_, num_actions)
		self.move(thetas_)

	def clip_actions(self, thetas_, num_actions):
		for i in range(num_actions):
			if thetas_[i] < self.min_thetas[i]:
				thetas_[i] = self.min_thetas[i]
			elif thetas_[i] > self.max_thetas[i]:
				thetas_[i] = self.max_thetas[i]
		return thetas_

	def homog_transf_matrix(self, theta, alfa, r, d):
		s_theta = sin(theta)
		c_theta = cos(theta)

		s_alfa = sin(alfa)
		c_alfa = cos(alfa)

		H = [[c_theta, -s_theta*c_alfa, s_theta*s_alfa,  r*c_theta],
			 [s_theta, c_theta*c_alfa,  -c_theta*s_alfa, r*s_theta],
			 [0,       s_alfa,          c_alfa,          d],
			 [0 ,      0,               0,               1]]

		H = np.array(H)

		return H

	def direct_kinematics(self, thetas):
		theta1, theta2, theta3, theta4 = thetas
		l1, l2, l3, l4, l5 = self.lengths

		DH_table = [[theta1,	            radians(-90),   0,                   l1 ],
				   	[theta2-radians(90),	0, 				l2,                  0  ],
				   	[theta3+radians(90),	0, 				l3+l4*cos(theta3),   0  ],
				   	[theta4,	            0,				l5,                  0  ]]

		DH_table = np.array(DH_table)

		H0_1 = self.homog_transf_matrix(DH_table[0][0], DH_table[0][1], DH_table[0][2], DH_table[0][3])
		H1_2 = self.homog_transf_matrix(DH_table[1][0], DH_table[1][1], DH_table[1][2], DH_table[1][3])
		H2_3 = self.homog_transf_matrix(DH_table[2][0], DH_table[2][1], DH_table[2][2], DH_table[2][3])
		H3_4 = self.homog_transf_matrix(DH_table[3][0], DH_table[3][1], DH_table[3][2], DH_table[3][3])

		H0_2 = np.matmul(H0_1, H1_2)
		H0_3 = np.matmul(H0_2, H2_3)
		H0_4 = np.matmul(H0_3, H3_4)

		position = np.array([H0_4[0][3], H0_4[1][3], H0_4[2][3]])

		return position
