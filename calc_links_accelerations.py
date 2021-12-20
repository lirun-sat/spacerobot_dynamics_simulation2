import numpy as np
from Get_global_value import num_q
from Get_global_value import J_type
from Get_global_value import Ez
from Get_global_value import BB
from Get_global_value import cc
from cross import cross
from Get_global_value import c0


def calc_links_accelerations(A_base, R_links, omega_base_in_body, omega_base_dot_in_body, v_base_dot, w_links, q, q_dot, q_dot_dot):
    v_links_dot = np.zeros((num_q, 3))
    w_links_dot = np.zeros((num_q, 3))
    omega_base = A_base @ omega_base_in_body
    omega_base_dot = A_base @ omega_base_dot_in_body

    if num_q == 0:
        print('Single body, there is no link')

    else:
        A_I_base = A_base

        for i in range(num_q):
            if BB[i] == -1:
                A_I_i = R_links[i, :, :]

                if J_type[i] == 'R':
                    w_links_dot[i, :] = omega_base_dot[0:3] + cross(w_links[i, :], A_I_i @ Ez * q_dot[i]) + A_I_i @ Ez * q_dot_dot[i]
                    v_links_dot[i, :] = v_base_dot[0:3] \
                               + cross(omega_base_dot[0:3], A_I_base @ c0[i, :]) \
                               + cross(omega_base[0:3], cross(omega_base[0:3], A_I_base @ c0[i, :])) \
                               - cross(w_links_dot[i, :], A_I_i @ cc[i, i, :]) \
                               - cross(w_links[i, :], cross(w_links[i, :], A_I_i @ cc[i, i, :]))

                else:
                    w_links_dot[i, :] = omega_base_dot[0:3]
                    v_links_dot[i, :] = v_base_dot[0:3] \
                               + cross(omega_base_dot[0:3], np.dot(A_I_base, c0[i, :])) \
                               + cross(omega_base[i, :], cross(omega_base[0:3], np.dot(A_I_base, c0[i, :]))) \
                               + cross(w_links_dot[i, :], np.dot(np.dot(A_I_i, Ez), q[i])) \
                               + cross(w_links[i, :], cross(w_links[i, :], np.dot(np.dot(A_I_i, Ez), q[i]))) \
                               + 2 * cross(w_links[i, :], np.dot(np.dot(A_I_i, Ez), q_dot[i])) \
                               + np.dot(np.dot(A_I_i, Ez), q_dot_dot[i]) \
                               - cross(w_links_dot[i, :], np.dot(A_I_i, cc[i, i, :])) \
                               - cross(w_links[i, :], cross(w_links[i, :], np.dot(A_I_i, cc[i, i, :])))

            else:
                A_I_BB = R_links[BB[i], :, :]
                A_I_i = R_links[i, :, :]

                if J_type[i] == 'R':
                    w_links_dot[i, :] = w_links_dot[BB[i], :] + cross(w_links[i, :], np.dot(np.dot(A_I_i, Ez), q_dot[i])) + np.dot(np.dot(A_I_i, Ez), q_dot_dot[i])
                    v_links_dot[i, :] = v_links_dot[BB[i], :] \
                               + cross(w_links_dot[BB[i], :], np.dot(A_I_BB, cc[BB[i], i, :])) \
                               + cross(w_links[BB[i], :], cross(w_links[BB[i], :], np.dot(A_I_BB, cc[BB[i], i, :]))) \
                               - cross(w_links_dot[i, :], np.dot(A_I_i, cc[i, i, :])) \
                               - cross(w_links[i, :], cross(w_links[i, :], np.dot(A_I_i, cc[i, i, :])))

                else:
                    w_links_dot[i, :] = w_links_dot[BB[i], :]
                    v_links_dot[i, :] = v_links_dot[BB[i], :] \
                               + cross(w_links_dot[BB[i], :], np.dot(A_I_BB, cc[BB[i], i, :])) \
                               + cross(w_links[BB[i], :], cross(w_links[BB[i], :], np.dot(A_I_BB, cc[BB[i], i, :]))) \
                               + cross(w_links_dot[i, :], np.dot(np.dot(A_I_i, Ez), q[i])) \
                               + cross(w_links[i, :], cross(w_links[i, :], np.dot(np.dot(A_I_i, Ez), q[i]))) \
                               + 2 * cross(w_links[i, :], np.dot(np.dot(A_I_i, Ez), q_dot[i])) \
                               + np.dot(np.dot(A_I_i, Ez), q_dot_dot[i]) \
                               - cross(w_links_dot[i, :], np.dot(A_I_i, cc[i, i, :])) \
                               - cross(w_links[i, :], cross(w_links[i, :], np.dot(A_I_i, cc[i, i, :])))

    v_links_dot = v_links_dot.T
    w_links_dot = w_links_dot.T
    links_accelerations = np.concatenate((w_links_dot, v_links_dot), axis=0)

    return links_accelerations
