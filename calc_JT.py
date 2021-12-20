import numpy as np
from Get_global_value import num_q
from Get_global_value import J_type
from Get_global_value import Ez
from Get_global_value import cc
from Get_global_value import BB
from cross import cross


def calc_JT(R_links, links_positions):
    JT = np.zeros((num_q, num_q, 3))
    if num_q == 0:
        print('Single body, there is no link')
    else:
        for i in range(num_q):
            j = i
            while j > -1:
                R_links_j = R_links[j, :, :]
                if J_type == 'R':
                    JT[i, j, :] = cross((R_links_j @ Ez), (links_positions[i, :] - links_positions[j, :] - R_links_j @ cc[j, j, :]))
                else:
                    JT[i, j, :] = R_links_j @ Ez
                if BB[j] == -1:
                    j = -1
                else:
                    j = BB[j]

    return JT
