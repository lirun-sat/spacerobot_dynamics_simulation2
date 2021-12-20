import numpy as np
from Get_global_value import num_q
from Get_global_value import mass
from Get_global_value import m
from Get_global_value import inertia
from Get_global_value import inertia0
from skew_sym import skew_sym


def calc_H_base(R_base, A_base, links_positions, R_links):
    H_base = np.zeros((6, 6))
    m_tot = mass
    R_system_center_temp = mass * R_base
    inertia_base = inertia0
    H_omega = A_base @ inertia_base @ A_base.T
    for i in range(num_q):
        m_tot += m[i]
        R_system_center_temp += m[i] * links_positions[i, :]
        H_omega += R_links[i, :] @ inertia[i, :, :] @ R_links[i, :, :].T \
                   + m[i] * skew_sym(links_positions[i, :] - R_base).T @ skew_sym(links_positions[i, :] - R_base)

    R_system_center = R_system_center_temp / m_tot
    H_base[0:3, 0:3] = m_tot * np.eye(3)
    H_base[0:3, 3:6] = m_tot * skew_sym(R_system_center - R_base).T
    H_base[3:6, 0:3] = m_tot * skew_sym(R_system_center - R_base)
    H_base[3:6, 3:6] = H_omega

    return H_base

