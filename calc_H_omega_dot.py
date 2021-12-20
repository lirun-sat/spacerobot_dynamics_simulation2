import numpy as np
from Get_global_value import num_q
from Get_global_value import m
from Get_global_value import inertia0
from skew_sym import skew_sym


def calc_H_omega_dot(A_base, R_base, links_positions, v_base, omega_base, v_links, inertia_dot):
    """
    Calculate Part of the Time Derivative of the Base-Spacecraft Inertia Matrix
    :param A_base:
    :param R_base:
    :param links_positions:
    :param v_base:
    :param omega_base:
    :param v_links:
    :param inertia_dot:
    :return:
    """

    A_base_dot = skew_sym(omega_base) @ A_base
    inertia_b_dot = A_base_dot @ inertia0 @ A_base.T + A_base @ inertia0 @ A_base_dot.T
    H_omega_dot = inertia_b_dot
    for i in range(num_q):
        R_base2link_i = links_positions[i, :] - R_base
        R_base2link_i_dot = v_links[i, :] - v_base
        H_omega_dot += inertia_dot[i, :, :] + m[i] * (R_base2link_i_dot.T @ R_base2link_i) + m[i] * (R_base2link_i.T @ R_base2link_i_dot)

    return H_omega_dot


