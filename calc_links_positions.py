import numpy as np
from Get_global_value import num_q
from Get_global_value import BB
from Get_global_value import J_type
from Get_global_value import Qi
from Get_global_value import c0
from Get_global_value import cc
from Get_global_value import Ez
from rpy2dc import rpy2dc


def calc_links_positions(R_base, A_base, R_links, q):

    links_positions = np.zeros((num_q, 3))
    if num_q == 0:
        print('Single body, there is no link')

    else:
        for i in range(num_q):
            R_links_i = R_links[i, 0:3, 0:3]
            if BB[i] == -1:
                if J_type[i] == 'R':
                    links_positions[i, 0:3] = R_base[0:3] + A_base @ c0[i, 0:3] + R_links_i @ (-cc[i, i, 0:3])
                else:
                    links_positions[i, 0:3] = R_base[0:3] + A_base @ c0[i, 0:3] + R_links_i @ (q[i] * Ez - cc[i, i, 0:3])
            else:
                R_links_i_BB = R_links[BB[i], 0:3, 0:3]
                if J_type[i] == 'R':
                    links_positions[i, :] = links_positions[BB[i], :] + R_links_i_BB @ cc[BB[i], i, :] - R_links_i @ cc[i, i, :]
                else:
                    links_positions[i, :] = links_positions[BB[i], :] + R_links_i_BB @ cc[BB[i], i, :] + R_links_i @ (q[i] * Ez - cc[i, i, 0:3])

    return links_positions
