def calc_links_accelerations_using_N(N, N_dot, q_dot_general, q_dot_dot_general):

    twist_dot = N_dot @ q_dot_general + N @ q_dot_dot_general

    return twist_dot

