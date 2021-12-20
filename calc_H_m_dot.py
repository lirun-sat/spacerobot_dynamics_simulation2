import numpy as np
from Get_global_value import num_q
from Get_global_value import m
from Get_global_value import inertia


def calc_H_m_dot(JR, JR_dot, R_links, inertia_dot, JT, JT_dot):
    """
    Calculate Time Derivative of the Manipulator Inertia Matrix
    :param JR:
    :param JR_dot:
    :param R_links:
    :param inertia_dot:
    :param JT:
    :param JT_dot:
    :return:
    """
    H_m_dot = np.zeros((num_q, num_q))
    for i in range(num_q):
        H_m_dot += JR_dot[i, :, :].T @ (R_links[i, :] @ inertia[i, :, :] @ R_links[i, :].T) @ JR[i, :, :] \
                   + JR[i, :, :].T @ inertia_dot[i, :, :] @ JR[i, :, :] \
                   + JR[i, :, :].T @ (R_links[i, :] @ inertia[i, :, :] @ R_links[i, :].T) @ JR_dot[i, :, :] \
                   + m[i] * JT_dot[i, :, :].T @ JT[i, :, :] + m[i] * JT[i, :, :].T @ JT_dot[i, :, :]

    return H_m_dot
