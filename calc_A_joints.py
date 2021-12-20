import numpy as np
from Get_global_value import num_q
from Get_global_value import BB
from Get_global_value import J_type
from Get_global_value import Qi
from rpy2dc import rpy2dc


def calc_A_joints(A_base):

    A_joints = np.zeros((num_q, 3, 3))
    if num_q == 0:
        print('Single body, there is no link')
    else:
        for i in range(num_q):
            if BB[i] == -1:
                if J_type[i] == 'R':
                    A_joint_i = rpy2dc(Qi[i, 0], Qi[i, 1], Qi[i, 2]).T
                    # A_joint_i expresses the unit base vector of the previous frame using the unit base vector of the current moving frame
                else:
                    A_joint_i = rpy2dc(Qi[i, :], 0, 0).T
                A_joints[i, 0:3, 0:3] = A_base @ A_joint_i
            else:
                if J_type[i] == 'R':
                    A_joint_i = rpy2dc(Qi[i, 0], Qi[i, 1], Qi[i, 2]).T
                else:
                    A_joint_i = rpy2dc(Qi[i, 0:3], 0, 0).T
                A_joints[i, 0:3, 0:3] = A_joints[BB[i], 0:3, 0:3] @ A_joint_i

    return A_joints
