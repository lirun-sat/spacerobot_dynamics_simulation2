import numpy as np
from Get_global_value import num_q
from Get_global_value import cc
from Get_global_value import Ez
from skew_sym import skew_sym


def calc_Nd_dot(A_base, omega_base_in_body, A_links, A_links_dot):
    """

    :param A_base:
    :param omega_base_in_body:
    :param A_links:
    :param A_links_dot:
    :return:
    """
    omega_base = A_base @ omega_base_in_body
    P_b_dot = np.block([
        [skew_sym(omega_base) @ A_base, np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3))]
    ])

    for i in range(num_q):
        P_b_dot = np.concatenate((P_b_dot, np.zeros((6, 6))), axis=0)

    p_i_dot_buff = np.zeros((num_q, 6 * (num_q + 1), 1))

    for i in range(num_q):
        p_i_omega_dot = A_links_dot[i, :, :] @ Ez  # tuple (3, 1)

        p_i_linear_dot = skew_sym(A_links_dot[i, :, :] @ Ez) @ (A_links[i, :, :] @ (-np.expand_dims(cc[i, i, :], axis=0).T)) \
                         + skew_sym(A_links[i, :, :] @ Ez) @ (A_links_dot[i, :, :] @ (-np.expand_dims(cc[i, i, :], axis=0).T))

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

    # print('Nd_dot.shape', Nd_dot.shape)

    return Nd_dot
