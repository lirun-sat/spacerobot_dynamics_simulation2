import numpy as np
from Get_global_value import num_q
from Get_global_value import J_type
from Get_global_value import Ez
from Get_global_value import BB
from Get_global_value import cc
from cross import cross
from Get_global_value import c0


def calc_links_velocity(A_base, A_links, v_base, omega_base_in_body, q, q_dot):
    """

    :param A_base: tuple (3, 3)
    :param A_links: tuple (num_q, 3, 3)
    :param v_base: linear velocity in inertial frame, tuple (3, 1)
    :param omega_base_in_body: angular velocity in body axis, tuple (3, 1)
    :param q: tuple (num_q, 1)
    :param q_dot: tuple (num_q, 1)
    :return:
    """
    v_links = np.zeros((num_q, 3, 1))
    w_links = np.zeros((num_q, 3, 1))
    omega_base = A_base @ omega_base_in_body

    if num_q == 0:
        print('Single body, there is no link')
    else:
        for i in range(num_q):
            if BB[i] == -1:
                A_links_i = A_links[i, :, :]
                if J_type == 'R':
                    w_links[i, :, :] = omega_base[0:3, :] + A_links_i @ Ez * q_dot[i, :]
                    v_links[i, :, :] = v_base[0:3, :] + cross(omega_base[:, :], A_base @ c0[i, :, :]) \
                                       + cross(w_links[i, 0:3, :], A_links_i @ (-np.expand_dims(cc[i, i, 0:3], axis=0).T))
                else:
                    w_links[i, :, :] = omega_base[0:3, :]

                    print('v_base.shape', v_base.shape)
                    print('omega_base.shape', omega_base.shape)
                    print('A_base.shape', A_base.shape)
                    print('cross(omega_base[:, :], A_base @ c0[i, :, :]).shape', (cross(omega_base[:, :], A_base @ c0[i, :, :]).shape))

                    v_links[i, :, :] = v_base[:, :] + cross(omega_base[:, :], A_base @ c0[i, :, :]) \
                                       + cross(w_links[i, :, :], A_links_i @ (-np.expand_dims(cc[i, i, 0:3], axis=0).T)) \
                                       + cross(w_links[i, :, :], A_links_i @ (q[i, :] * Ez)) + (A_links_i @ Ez) * q_dot[i, :]

            else:
                A_links_i_BB = A_links[BB[i], :, :]
                A_links_i = A_links[i, :, :]
                if J_type == 'R':
                    w_links[i, :, :] = w_links[BB[i], :, :] + A_links_i @ (q_dot[i, :] * Ez)
                    v_links[i, :, :] = v_links[BB[i], :, :] + cross(w_links[BB[i], :, :], A_links_i_BB @ np.expand_dims(cc[BB[i], i, :], axis=0).T) \
                                       - cross(w_links[i, :, :], A_links_i @ np.expand_dims(cc[i, i, :], axis=0).T)
                else:
                    w_links[i, :, :] = w_links[BB[i], :, :]
                    v_links[i, :, :] = v_links[BB[i], :, :] + cross(w_links[BB[i], :, :], A_links_i_BB @ np.expand_dims(cc[BB[i], i, :], axis=0).T) \
                                       - cross(w_links[i, :, :], A_links_i @ np.expand_dims(cc[i, i, :], axis=0).T) \
                                       + cross(w_links[i, :, :], A_links_i @ (q[i, :] * Ez)) + A_links_i @ (q_dot[i, :] * Ez)

    return v_links, w_links






















