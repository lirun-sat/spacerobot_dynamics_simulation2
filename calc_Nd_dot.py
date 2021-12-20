import numpy as np
from Get_global_value import num_q
from Get_global_value import cc
from Get_global_value import Ez
from skew_sym import skew_sym


def calc_Nd_dot(A_base, omega_base, R_links, R_links_dot):

    P_b_dot = np.block([
        [skew_sym(omega_base) @ A_base, np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3))]
    ])

    for i in range(num_q):
        P_b_dot = np.concatenate((P_b_dot, np.zeros((6, 6))), axis=0)

    p_i_dot_buff = np.zeros((num_q, 6 * (num_q + 1), 1))

    for i in range(num_q):
        p_i_omega_dot = R_links_dot[i, :, :] @ Ez

        p_i_linear_dot = skew_sym(R_links_dot[i, :, :] @ Ez) @ (R_links[i, :, :] @ (-cc[i, i, :])) \
                         + skew_sym(R_links[i, :, :] @ Ez) @ (R_links_dot[i, :, :] @ (-cc[i, i, :]))

        p_i_omega_dot = np.expand_dims(p_i_omega_dot, axis=0)
        p_i_linear_dot = np.expand_dims(p_i_linear_dot, axis=0)
        p_i_dot = np.concatenate((p_i_omega_dot, p_i_linear_dot), axis=0)
        p_i_dot_buff_temp = np.concatenate((np.zeros((6 * (i + 1), 1)), p_i_dot), axis=0)
        p_i_dot_buff[i, :, :] = np.concatenate((p_i_dot_buff_temp, np.zeros((6 * (num_q - i - 1), 1))), axis=0)

    p_i_dot = p_i_dot_buff[0, :, :]

    for i in range(num_q):
        if i == num_q - 1:
            break
        else:
            p_i_dot = np.concatenate((p_i_dot, p_i_dot_buff[i + 1, :, :]), axis=1)

    Nd_dot = np.concatenate((P_b_dot, p_i_dot), axis=1)

    return Nd_dot

