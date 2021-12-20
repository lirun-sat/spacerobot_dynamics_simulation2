import numpy as np
from Get_global_value import num_q
from Get_global_value import branch
from skew_sym import skew_sym


def calc_Nl(base_position, links_positions):

    B_b2i_temp = np.zeros((num_q, 6, 6))
    B_j2i_temp_temp = np.zeros((num_q, num_q, 6, 6))
    B_j2i_temp = np.zeros((6, 6))

    for i in range(num_q):
        B_b2i_temp[i, :, :] = np.block([
            [np.eye(3), np.zeros((3, 3))],
            [skew_sym(links_positions[i, :, :] - base_position[:, :]).T, np.eye(3)]
        ])

    B_b2i = np.eye(6)
    for i in range(num_q):
        B_b2i = np.concatenate((B_b2i, B_b2i_temp[i, :, :]), axis=0)

    for j in range(num_q):
        for i in range(num_q):
            if j == i:
                B_j2i_temp_temp[j, i, :, :] = np.eye(6)
            elif j > i or branch[j, i] == 0:
                B_j2i_temp_temp[j, i, :, :] = np.zeros((6, 6))
            else:
                B_j2i_temp_temp[j, i, :, :] = np.block([
                    [np.eye(3), np.zeros((3, 3))],
                    [skew_sym(links_positions[i, :, :] - links_positions[j, :, :]).T, np.eye(3)]
                ])

    B_j2i_buff = np.zeros((num_q, 6*(num_q+1), 6))

    for i in range(num_q):
        B_j2i_temp = np.concatenate((B_j2i_temp, B_j2i_temp_temp[0, i, :, :]), axis=0)

    for j in range(num_q):
        B_j2i_buff[j, :, :] = B_j2i_temp
        B_j2i_temp = np.zeros((6, 6))
        for i in range(num_q):
            if j == num_q - 1:
                break
            else:
                B_j2i_temp = np.concatenate((B_j2i_temp, B_j2i_temp_temp[j+1, i, :, :]), axis=0)

    B_j2i = B_j2i_buff[0, :, :]
    for i in range(num_q):
        if i == num_q - 1:
            break
        else:
            B_j2i = np.concatenate((B_j2i, B_j2i_buff[i+1, :, :]), axis=1)

    Nl = np.concatenate((B_b2i, B_j2i), axis=1)

    # print('Nl.shape', Nl.shape)

    # B_b2i_temp = np.expand_dims(B_b2i_temp, axis=0)
    # B_b2i = B_j2i_temp_temp.transpose((1, 0, 2, 3))
    # B_j2i = B_j2i_temp_temp.transpose((1, 0, 2, 3))
    #
    # Nl_lower_part = np.concatenate((B_b2i, B_j2i), axis=1)
    # Nl_upper_part = np.zeros((num_q+1, 6, 6))
    # Nl_upper_part[0, :, :] = np.eye(6)
    # Nl_upper_part = np.expand_dims(Nl_upper_part, axis=0)
    # Nl = np.concatenate((Nl_upper_part, Nl_lower_part), axis=0)

    return Nl
