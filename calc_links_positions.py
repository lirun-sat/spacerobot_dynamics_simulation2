import numpy as np
from Get_global_value import num_q
from Get_global_value import BB
from Get_global_value import J_type
from Get_global_value import c0
from Get_global_value import cc
from Get_global_value import Ez


def calc_links_positions(base_position, A_base, A_links, q):
    """

    :param base_position: tuple(3, 1)
    :param A_base:
    :param A_links:
    :param q: tuple (num_q, 1)
    :return:
    """

    links_positions = np.zeros((num_q, 3, 1))
    if num_q == 0:
        print('Single body, there is no link')

    else:
        for i in range(num_q):
            A_links_i = A_links[i, 0:3, 0:3]
            if BB[i] == -1:
                if J_type[i] == 'R':
                    links_positions[i, :, :] = base_position[0:3, :] + A_base @ c0[i, 0:3, :] \
                                               + A_links_i @ (-np.expand_dims(cc[i, i, 0:3], axis=0).T)
                else:
                    links_positions[i, :, :] = base_position[0:3, :] + A_base @ c0[i, 0:3, :] \
                                               + A_links_i @ (q[i, :] * Ez - np.expand_dims(cc[i, i, 0:3], axis=0).T)
            else:
                A_links_i_BB = A_links[BB[i], 0:3, 0:3]
                if J_type[i] == 'R':
                    links_positions[i, :, :] = links_positions[BB[i], :, :] + A_links_i_BB @ np.expand_dims(cc[BB[i], i, :], axis=0).T \
                                               - A_links_i @ np.expand_dims(cc[i, i, 0:3], axis=0).T
                else:
                    links_positions[i, :, :] = links_positions[BB[i], :, :] + A_links_i_BB @ np.expand_dims(cc[BB[i], i, :], axis=0).T \
                                               + A_links_i @ (q[i, :] * Ez - np.expand_dims(cc[i, i, 0:3], axis=0).T)

    return links_positions


























