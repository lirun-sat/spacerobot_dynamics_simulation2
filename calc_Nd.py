import numpy as np
from Get_global_value import num_q
from Get_global_value import cc
from Get_global_value import Ez
from skew_sym import skew_sym


def calc_Nd(A_base, A_links):

    P_b = np.block([
        [A_base, np.zeros((3, 3))],
        [np.zeros((3, 3)), np.eye(3)]
    ])
    for i in range(num_q):
        P_b = np.concatenate((P_b, np.zeros((6, 6))), axis=0)

    p_i_buff = np.zeros((num_q, 6*(num_q+1), 1))

    for i in range(num_q):
        p_i_omega = A_links[i, :, :] @ Ez  # tuple (3, 1)
        p_i_linear = skew_sym(A_links[i, :, :] @ Ez) @ (A_links[i, :, :] @ (-np.expand_dims(cc[i, i, :], axis=0).T))

        p_i = np.concatenate((p_i_omega, p_i_linear), axis=0)
        p_i_buff_temp = np.concatenate((np.zeros((6*(i+1), 1)), p_i), axis=0)
        p_i_buff[i, :, :] = np.concatenate((p_i_buff_temp, np.zeros((6*(num_q-i-1), 1))), axis=0)
    
    p_i = p_i_buff[0, :, :]

    for i in range(num_q):
        if i == num_q-1:
            break
        else:
            p_i = np.concatenate((p_i, p_i_buff[i+1, :, :]), axis=1)

    Nd = np.concatenate((P_b, p_i), axis=1)

    # print('Nd.shape', Nd.shape)

    return Nd






















