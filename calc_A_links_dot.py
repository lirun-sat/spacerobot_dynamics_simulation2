import numpy as np
from Get_global_value import num_q
from Get_global_value import J_type
from Get_global_value import Ez
from Get_global_value import BB
from skew_sym import skew_sym


def calc_A_links_dot(A_base, omega_base_in_body, q_dot, A_links):
    omega_link = np.zeros((num_q, 3, 1))
    A_links_dot = np.zeros((num_q, 3, 3))
    omega_base = A_base @ omega_base_in_body

    # print('omega_base.shape', omega_base.shape)
    # print('calc_A_links_dot  q_dot.shape', q_dot.shape)

    if num_q == 0:
        print('Single body, there is no link')
    else:
        for i in range(num_q):
            if BB[i] == -1:
                if J_type[i] == 'R':
                    omega_link[i, :, :] = omega_base + q_dot[i, :] * (A_links[i, :, :] @ Ez)  # Ez is tuple(3, 1)
                else:
                    omega_link[i, :, :] = omega_base
            else:
                if J_type[i] == 'R':
                    omega_link[i, :, :] = omega_link[BB[i], :, :] + q_dot[i, :] * (A_links[i, :, :] @ Ez)
                else:
                    omega_link[i, :, :] = omega_link[BB[i], :, :]

            A_links_dot[i, :, :] = skew_sym(omega_link[i, :, :]) @ A_links[i, :, :]

    return A_links_dot
