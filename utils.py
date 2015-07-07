def get_coarse_params(v_0, p_0, chi):
    D_rho = v_0 ** 2 / p_0
    mu = chi * v_0
    return D_rho, mu


def get_reduced_params(D_rho, mu, rho_0, phi, delta, D_c):
    D_rho_red = D_rho / D_c
    mu_red = mu * phi * rho_0 / (delta * D_c)
    return D_rho_red, mu_red
