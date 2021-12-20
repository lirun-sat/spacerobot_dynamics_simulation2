import numpy as np
from Get_global_value import num_q
from Get_global_value import J_type
from Get_global_value import Ez
from Get_global_value import BB
from skew_sym import skew_sym


def calc_A_joints_dot(A_base, omega_base_in_body, q_dot, A_joints):

    omega_base = A_base @ omega_base_in_body
    omega_joint = np.zeros((num_q, 3, 1))
    A_joints_dot = np.zeros((num_q, 3, 3))
    if num_q == 0:
        print('Single body, there is no link')
    else:
        for i in range(num_q):
            if BB[i] == -1:
                omega_joint[i, :] = omega_base

            else:
                if J_type[BB[i]] == 'R':
                    omega_joint[i, :, :] = omega_joint[BB[i], :, :] + q_dot[BB[i], :] * (A_joints[BB[i], :, :] @ Ez)  # Ez is tuple(3, 1)
                else:
                    omega_joint[i, :, :] = omega_joint[BB[i], :, :]

            A_joints_dot[i, :, :] = skew_sym(omega_joint[i, :, :].reshape(3,)) @ A_joints[i, :, :]

    return A_joints_dot



