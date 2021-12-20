import numpy as np
import scipy.linalg
from Get_global_value import num_q
from Get_global_value import mass
from Get_global_value import m
from Get_global_value import inertia
from Get_global_value import inertia0
from skew_sym import skew_sym


def calc_M_and_M_dot(A_base, omega_base_in_body, A_links, A_links_dot):

    omega_base = A_base @ omega_base_in_body
    M_tempt = np.zeros((num_q+1, 6, 6))
    M_dot_tempt = np.zeros((num_q+1, 6, 6))

    inertia_b = A_base @ inertia0 @ A_base.T
    inertia_b_dot = (skew_sym(omega_base) @ A_base) @ inertia0 @ A_base.T + A_base @ inertia0 @ (skew_sym(omega_base) @ A_base).T
    Mb = np.block([
        [inertia_b, np.zeros((3, 3))],
        [np.zeros((3, 3)), mass * np.eye(3)]
    ])

    Mb_dot = np.block([
        [inertia_b_dot, np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3))]
    ])

    M = Mb
    M_dot = Mb_dot

    for i in range(num_q):

        inertia_i = A_links[i, :, :] @ inertia[i, :, :] @ A_links[i, :, :].T

        M_tempt[i, :, :] = np.block([
            [inertia_i, np.zeros((3, 3))],
            [np.zeros((3, 3)), m[i] * np.eye(3)]
        ])

        inertia_i_dot = A_links_dot[i, :, :] @ inertia[i, :, :] @ A_links[i, :, :].T + A_links[i, :, :] @ inertia[i, :, :] @ A_links_dot[i, :, :].T

        M_dot_tempt[i, :, :] = np.block([
            [inertia_i_dot, np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3))]
        ])

        M = scipy.linalg.block_diag(M, M_tempt[i, :, :])

        M_dot = scipy.linalg.block_diag(M_dot, M_dot_tempt[i, :, :])

    return M, M_dot
