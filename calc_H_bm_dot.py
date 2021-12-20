import numpy as np
from Get_global_value import num_q
from Get_global_value import m
from Get_global_value import inertia
from skew_sym import skew_sym


def calc_H_bm_dot(JR, JR_dot, JT, JT_dot, inertia_dot, links_positions, R_base, v_links, v_base):
    H_bm_dot = np.zeros((6, num_q))
    J_TS_dot = np.zeros((3, num_q))
    H_Sq_dot = np.zeros((3, num_q))
    for i in range(num_q):
        J_TS_dot += m[i] * JT_dot[i, :, :]
        H_Sq_dot += inertia_dot[i, :, :] @ JR[i, :, :] + inertia[i, :, :] @ JR_dot[i, :, :] + m[i] * (skew_sym(v_links[i, :] - v_base) @ JT[i, :, :] + skew_sym(links_positions[i, :] - R_base) @ JT_dot[i, :, :])

    H_bm_dot[0:3, 0:num_q] = J_TS_dot
    H_bm_dot[3:6, 0:num_q] = H_Sq_dot

    return H_bm_dot
