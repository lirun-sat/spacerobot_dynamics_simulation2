import numpy as np
from Get_global_value import num_q
from Get_global_value import inertia


def calc_inertia_dot(A_links_dot, A_links):
    inertia_dot = np.zeros((num_q, 3, 3))
    for i in range(num_q):
        inertia_dot[i, :, :] = A_links_dot[i, :, :] @ inertia[i, :, :] @ A_links[i, :, :].T + A_links[i, :, :] @ inertia[i, :, :] @ A_links_dot[i, :, :].T

    return inertia_dot
