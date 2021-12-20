import numpy as np
from Get_global_value import num_q
from calc_A_joints import calc_A_joints
from calc_A_links import calc_A_links
from calc_links_positions import calc_links_positions
from calc_links_velocity import calc_links_velocity
from calc_H import calc_H
from calc_Nl import calc_Nl
from calc_Nd import calc_Nd
from calc_Nl_dot import calc_Nl_dot
from calc_Nd_dot import calc_Nd_dot
from calc_N_and_N_dot import calc_N_and_N_dot
from calc_M_and_M_dot import calc_M_and_M_dot
from calc_C_non import calc_C_non
from calc_A_links_dot import calc_A_links_dot
from calc_q_dot_general import calc_q_dot_general


def calc_forward_dynamics(A_base, base_position, omega_base_in_body, v_base, q, q_dot, tau):

    A_joints = calc_A_joints(A_base)
    A_links = calc_A_links(A_joints, q)
    A_links_dot = calc_A_links_dot(A_base, omega_base_in_body, q_dot, A_links)

    links_positions = calc_links_positions(base_position, A_base, A_links, q)
    v_links, w_links = calc_links_velocity(A_base, A_links, v_base, omega_base_in_body, q, q_dot)

    M, M_dot = calc_M_and_M_dot(A_base, omega_base_in_body, A_links, A_links_dot)

    Nl = calc_Nl(base_position, links_positions)
    Nl_dot = calc_Nl_dot(v_base, v_links)
    Nd = calc_Nd(A_base, A_links)
    Nd_dot = calc_Nd_dot(A_base, omega_base_in_body, A_links, A_links_dot)

    N, N_dot = calc_N_and_N_dot(Nl, Nl_dot, Nd, Nd_dot)

    C_non = calc_C_non(M, M_dot, N, N_dot)

    H = calc_H(N, M)

    q_dot_general = calc_q_dot_general(omega_base_in_body, v_base, q_dot)

    Non_linear_term = C_non @ q_dot_general

    q_dot_dot_general = np.linalg.inv(H) @ (tau - Non_linear_term)

    return q_dot_dot_general
