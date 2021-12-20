def calc_N_and_N_dot(Nl, Nl_dot, Nd, Nd_dot):
    # print()
    N = Nl @ Nd
    N_dot = Nl_dot @ Nd + Nl @ Nd_dot
    return N, N_dot

