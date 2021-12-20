def calc_C_non(M, M_dot, N, N_dot):
    C_non = N.T @ (M @ N_dot + M_dot @ N)

    return C_non
