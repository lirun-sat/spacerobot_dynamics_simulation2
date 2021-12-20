import numpy as np
from Get_global_value import num_q
from Get_global_value import J_type
from Get_global_value import Ez
from Get_global_value import BB


def calc_JR(R_joints):

    JR = np.zeros((num_q, num_q, 3))

    if num_q == 0:
        print('Single body, there is no link')

    else:
        for i in range(num_q):
            j = i
            while j > -1:
                R_joints_j = R_joints[j, :, :]
                if J_type[j] == 'R':
                    JR[i, j, :] = R_joints_j @ Ez
                else:
                    JR[i, j, :] = np.zeros(3)

                if BB[j] == -1:
                    j = -1
                else:
                    j = BB[j]

    return JR
