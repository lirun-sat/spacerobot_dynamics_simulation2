import numpy as np
from Get_global_value import num_q
from Get_global_value import inertia


def calc_inertia_dot(R_links_dot, R_links):
    inertia_dot = np.zeros((num_q, 3, 3))
    for i in range(num_q):
        inertia_dot[i, :, :] = R_links_dot[i, :, :] @ inertia[i, :, :] @ R_links[i, :, :].T + R_links[i, :, :] @ inertia[i, :, :] @ R_links_dot[i, :, :].T

    return inertia_dot
