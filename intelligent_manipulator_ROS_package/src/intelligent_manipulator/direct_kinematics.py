import numpy as np
from math import sin, cos, tan, radians, degrees

def homog_transf_matrix(theta, alfa, r, d):
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

def direct_kinematics(thetas, lenghts):
    theta1, theta2, theta3, theta4 = thetas
    l1, l2, l3, l4, l5 = lenghts

    DH_table = [[theta1,                radians(-90),   0,                  l1],
                [radians(-90)+theta2,   0,              l2,                 0],
                [radians(90)+theta3,    0,              l3+l4*cos(theta3),  0],
                [theta4,                0,              l5,                 0]]

    DH_table = np.array(DH_table)

    H0_1 = homog_transf_matrix(DH_table[0][0], DH_table[0][1], DH_table[0][2], DH_table[0][3])
    H1_2 = homog_transf_matrix(DH_table[1][0], DH_table[1][1], DH_table[1][2], DH_table[1][3])
    H2_3 = homog_transf_matrix(DH_table[2][0], DH_table[2][1], DH_table[2][2], DH_table[2][3])
    H3_4 = homog_transf_matrix(DH_table[3][0], DH_table[3][1], DH_table[3][2], DH_table[3][3])

    H0_2 = np.matmul(H0_1, H1_2)
    H0_3 = np.matmul(H0_2, H2_3)
    H0_4 = np.matmul(H0_3, H3_4)

    position = np.array([H0_4[0][3], H0_4[1][3], H0_4[2][3]])

    return position
