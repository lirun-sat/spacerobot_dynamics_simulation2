import numpy as np
from Get_global_value import num_q
from Get_global_value import J_type
from Get_global_value import Ez
from Get_global_value import cc


def calc_joints_velocity(v_links, A_links_dot, q):

    v_joints = np.zeros((num_q, 3))

    for j in range(num_q):
        if J_type == 'R':
            v_joints[j, :] = v_links[j, :] + A_links_dot[j, :, :] @ cc[j, j, :]
        else:
            v_joints[j, :] = v_links[j, :] + A_links_dot[j, :, :] @ (cc[j, j, :] + q[j] * (-Ez))

    return v_joints
