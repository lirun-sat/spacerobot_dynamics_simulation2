import numpy as np
from Get_global_value import num_q
from Get_global_value import J_type
from Get_global_value import Ez
from Get_global_value import BB
from skew_sym import skew_sym


def calc_Rjoints_dot(omega_base, dq, R_joints):

    omega_joint = np.zeros((num_q, 3))
    R_joints_dot = np.zeros((num_q, 3, 3))
    if num_q == 0:
        print('Single body, there is no link')

    else:
        for i in range(num_q):
            if BB[i] == -1:
                omega_joint[i, :] = omega_base

            else:
                if J_type[BB[i]] == 'R':
                    omega_joint[i, :] = omega_joint[BB[i], :] + dq[BB[i]] * (R_joints[BB[i], :, :] @ Ez)
                else:
                    omega_joint[i, :] = omega_joint[BB[i], :]

            R_joints_dot[i, :, :] = skew_sym(omega_joint[i]) @ R_joints[i, :, :]

    return R_joints_dot


