import numpy as np


def calc_q_dot_general(omega_base_in_body, v_base, q_dot):

    q_dot_general_temp = np.concatenate((omega_base_in_body, v_base), axis=0)
    q_dot_general = np.concatenate((q_dot_general_temp, q_dot), axis=0)

    return q_dot_general

