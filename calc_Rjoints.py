import numpy as np
from Get_global_value import num_q
from Get_global_value import BB
from Get_global_value import J_type
from Get_global_value import Qi
from rpy2dc import rpy2dc



def calc_Rjoints(A0):

    R_joints = np.zeros((num_q, 3, 3))
    if num_q == 0:
        print('Single body, there is no link')

    else:
        R_base = A0

        for i in range(num_q):
            if BB[i] == -1:
                if J_type[i] == 'R':
                    R_joint_i = rpy2dc(Qi[i, 0], Qi[i, 1], Qi[i, 2]).T
                    # R_joint_i expresses the unit base vector of the previous frame using the unit base vector of the current moving frame
                else:
                    R_joint_i = rpy2dc(Qi[i, :], 0, 0).T

                R_joints[i, 0:3, 0:3] = R_base @ R_joint_i

            else:
                if J_type[i] == 'R':
                    R_joint_i = rpy2dc(Qi[i, 0], Qi[i, 1], Qi[i, 2]).T
                else:
                    R_joint_i = rpy2dc(Qi[i, 0:3], 0, 0).T

                R_joints[i, 0:3, 0:3] = R_joints[BB[i], 0:3, 0:3] @ R_joint_i

    return R_joints

