import numpy as np
from Get_global_value import num_q
from Get_global_value import J_type
from Get_global_value import Ez
from Get_global_value import BB
from Get_global_value import m0
from Get_global_value import m
from Get_global_value import mass
from Get_global_value import inertia0
from Get_global_value import inertia
from Get_global_value import cc
from cross import cross
from Get_global_value import c0


def calc_links_velocity(A_base, R_links, v_base, omega_base, q, dq):
    v_links = np.zeros((num_q, 3))
    w_links = np.zeros((num_q, 3))
    if num_q == 0:
        print('Single body, there is no link')
    else:
        for i in range(num_q):
            if BB[i] == -1:
                R_links_i = R_links[i, :, :]
                if J_type == 'R':
                    w_links[i, :] = omega_base[0:3] + R_links_i @ Ez * dq[i]
                    v_links[i, :] = v_base[0:3] + cross(omega_base[0:3], A_base @ c0[i, :]) + cross(w_links[i, :], R_links_i @ (-cc[i, i, :]))
                else:
                    w_links[i, :] = omega_base[0:3]
                    v_links[i, :] = v_base[0:3] + + cross(omega_base[0:3], A_base @ c0[i, :]) + cross(w_links[i, :], R_links_i @ (-cc[i, i, :])) \
                                    + cross(w_links[i, :], R_links_i @ (Ez * q[i])) + (R_links_i @ Ez) * dq[i]

            else:
                R_links_i_BB = R_links[BB[i], :, :]
                R_links_i = R_links[i, :, :]
                if J_type == 'R':
                    w_links[i, :] = w_links[BB[i], :] + R_links_i @ (Ez * dq[i])
                    v_links[i, :] = v_links[BB[i], :] + cross(w_links[BB[i], :], R_links_i_BB @ cc[BB[i], i, :]) - cross(w_links[i, :], R_links_i @ cc[i, i, :])
                else:
                    w_links[i, :] = w_links[BB[i], :]
                    v_links[i, :] = v_links[BB[i], :] + cross(w_links[BB[i], :], R_links_i_BB @ cc[BB[i], i, :]) - cross(w_links[i, :], R_links_i @ cc[i, i, :]) \
                                    + cross(w_links[i, :], R_links_i @ (Ez * q[i])) + R_links_i @ (Ez * dq[i])

    return v_links, w_links
