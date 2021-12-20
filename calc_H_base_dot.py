import numpy as np
from Get_global_value import num_q
from Get_global_value import m
from Get_global_value import inertia0
from skew_sym import skew_sym


def calc_H_base_dot(v_base, v_links, H_omega_dot):
    """
    Calculate Time Derivative of the Base-Spacecraft Inertia Matrix
    :param v_base:
    :param v_links:
    :param H_omega_dot:
    :return:
    """
    H_base_dot = np.zeros((6, 6))
    H_base_dot[0:3, 0:3] = np.zeros((3, 3))
    H_base_dot[3:6, 3:6] = H_omega_dot
    for i in range(num_q):
        H_base_dot[3:6, 0:3] += skew_sym(m[i] * (v_links[i, :] - v_base))

    H_base_dot[0:3, 3:6] = H_base_dot[3:6, 0:3].T

    return H_base_dot
