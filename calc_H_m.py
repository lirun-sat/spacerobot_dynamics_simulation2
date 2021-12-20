import numpy as np
from Get_global_value import num_q
from Get_global_value import m
from Get_global_value import inertia


def calc_H_m(JR, R_links, JT):

    H_m = np.zeros((num_q, num_q))

    for i in range(num_q):
        H_m += JR[i, :, :].T @ (R_links[i, :, :] @ inertia[i, :, :] @ R_links[i, :, :].T) @ JR[i, :, :] \
               + m[i] * JT[i, :, :].T @ JT[i, :, :]

    return H_m

