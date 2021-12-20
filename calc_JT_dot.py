import numpy as np
from Get_global_value import num_q
from Get_global_value import J_type
from Get_global_value import Ez
from Get_global_value import cc
from Get_global_value import BB
from cross import cross


def calc_JT_dot(links_positions, v_links, v_joints, R_links, R_links_dot):

    JT_dot = np.zeros((num_q, num_q, 3))
    if num_q == 0:
        print('Single body, there is no link')

    else:
        for i in range(num_q):
            j = i
            while j > -1:
                R_links_dot_j = R_links_dot[j, :, :]
                if J_type[j] == 'R':
                    JT_dot[i, j, :] = cross((R_links_dot_j @ Ez),
                                            (links_positions[i, :] - (links_positions[i, :] - links_positions[j, :] - R_links[j, :, :] @ cc[j, j, :]))) \
                                      + cross(R_links[j, :, :] @ Ez, (v_links[i, :] - v_joints[j, :]))
                else:
                    JT_dot[i, j, :] = R_links_dot_j @ Ez

                if BB[j] == -1:
                    j = -1
                else:
                    j = BB[j]

    return JT_dot

