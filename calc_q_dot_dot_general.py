import numpy as np


def calc_q_dot_and_q_dot_dot_general(omega_base_dot_in_body, v_base_dot, q_dot_dot):

    q_dot_dot = np.expand_dims(q_dot_dot, axis=0)
    q_dot_dot = q_dot_dot.T

    q_dot_dot_general_temp = np.concatenate((omega_base_dot_in_body, v_base_dot), axis=0)
    q_dot_dot_general = np.concatenate((q_dot_dot_general_temp, q_dot_dot), axis=0)

    return q_dot_dot_general
