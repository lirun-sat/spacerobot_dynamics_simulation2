import numpy as np
from Get_global_value import num_q
from Get_global_value import BB
from Get_global_value import J_type
from Get_global_value import Qi
from rpy2dc import rpy2dc
from Cxyz import cxyz


def calc_A_links(A_joints, q):
    """

    :param A_joints: tuple (n, 3, 3)
    :param q: tuple (n, 1)
    :return:
    """

    A_links = np.zeros((num_q, 3, 3))
    if num_q == 0:
        print('Single body, there is no link')

    else:
        for i in range(num_q):
            if J_type == 'R':
                A_links[i, :, :] = A_joints[i, :, :] @ cxyz(q[i, :], 0, 0, 1).T
            else:
                A_links[i, :, :] = A_joints[i, :, :]

    return A_links
