import numpy as np
from Get_global_value import num_q
from Get_global_value import m
from Get_global_value import inertia
from skew_sym import skew_sym


def calc_H_bm(JR, JT, R_links, R_base, links_positions):
    H_bm = np.zeros((6, num_q))
    J_TS = np.zeros((3, num_q))
    H_Sq = np.zeros((3, num_q))
    for i in range(num_q):
        J_TS += m[i] * JT[i, :, :]
        H_Sq += R_links[i, :, :] @ inertia[i, :, :] @ R_links[i, :, :].T @ JR[i, :, :] + m[i] * skew_sym(links_positions[i, :] - R_base) @ JT[i, :, :]

    H_bm[0:3, 0:num_q] = J_TS
    H_bm[3:6, 0:num_q] = H_Sq

    return H_bm
