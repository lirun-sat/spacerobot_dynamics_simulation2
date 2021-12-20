import numpy as np
from Get_global_value import num_q
from Get_global_value import J_type
from Get_global_value import Ez
from Get_global_value import BB
from skew_sym import skew_sym


def calc_Rlinks_dot(omega_base, dq, R_links):
    omega_link = np.zeros((num_q, 3))
    R_links_dot = np.zeros((num_q, 3, 3))
    if num_q == 0:
        print('Single body, there is no link')
    else:
        for i in range(num_q):
            if BB[i] == -1:
                if J_type[i] == 'R':
                    omega_link[i, :] = omega_base + dq[i] * (R_links[i, :, :] @ Ez)
                else:
                    omega_link[i, :] = omega_base
            else:
                if J_type[i] == 'R':
                    omega_link[i, :] = omega_link[BB[i], :] + dq[i] * (R_links[i, :, :] @ Ez)
                else:
                    omega_link[i, :] = omega_link[BB[i], :]

            R_links_dot[i, :, :] = skew_sym(omega_link[i, :]) @ R_links[i, :, :]

    return R_links_dot

