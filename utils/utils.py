import numpy as np


def find_nearest_index(v, a):
    return np.argmin(np.abs(a - v))


def get_rho_agent(m_agent, m_coarse):
    ns, bins = np.histogram(m_agent.r[:, 0], bins=m_coarse.get_x().shape[0],
                            range=[-m_agent.L / 2.0, m_agent.L / 2.0])
    rho_agent = ns / m_coarse.dx
    return rho_agent


def get_D_rho(v_0, p_0, dim):
    return v_0 ** 2 / (dim * p_0)


def get_mu(chi, v_0, p_0, onesided, L):
    beta = 2.0 * v_0 / L

    if onesided:
        # mu = p_0 * chi / (2.0 * (p_0 + beta) - p_0 * chi)
        # Fudge
        mu = p_0 * chi / (2.0 * (p_0 + beta))
    else:
        mu = p_0 * chi / (p_0 + beta)
    return v_0 * mu


def get_steady_state_c(rho_0, phi, delta):
    return rho_0 * phi / delta


def get_steady_state_c_grad(rho_0, phi, delta, D_c):
    return rho_0 * phi / (delta * D_c)


def get_reduced_c(delta, rho_0, phi, c):
    return c / get_steady_state_c(rho_0, phi, delta)


def get_reduced_rho(rho_0, rho):
    return rho / rho_0


def get_reduced_D_rho(D_rho, D_c):
    return D_rho / D_c


def get_reduced_mu(mu, phi, rho_0, delta, D_c):
    return mu * get_steady_state_c_grad(rho_0, phi, delta, D_c)


def get_reduced_length(delta, D_rho, x):
    return x * np.sqrt(delta / D_rho)


def get_reduced_time(delta, t):
    return t * delta


def get_left_face(I, J, nx, ny):
    # First left face is the total number of horizontal faces.
    j_l_0 = (nx + 1) * ny
    # Then step along x by `I`, and up y in steps of the number of vertical
    # faces.
    j_l = j_l_0 + I + J * (nx + 1)
    return j_l


def get_right_face(I, J, nx, ny):
    # Right face is next to left face.
    return get_left_face(I, J, nx, ny) + 1


def get_bottom_face(I, J, nx, ny):
    # First bottom face is at index zero.
    j_b_0 = 0
    # Then step along x by `I`, and up y in steps of the number of horizontal
    # faces.
    j_b = j_b_0 + I + J * nx
    return j_b


def get_top_face(I, J, nx, ny):
    # Top cell is bottom cell + number of horizontal faces
    return get_bottom_face(I, J, nx, ny) + nx


def get_neighbs(I, J, nx, ny, periodic=False):
    args = [I, J, nx, ny]
    get_funcs = [get_bottom_face, get_top_face, get_left_face, get_right_face]
    neighbs = [get_func(*args) for get_func in get_funcs]
    if periodic:
        if I == 0:
            neighbs.append(get_right_face(nx - 1, J, nx, ny))
        elif I == nx - 1:
            neighbs.append(get_left_face(0, J, nx, ny))
        if J == 0:
            neighbs.append(get_top_face(I, ny - 1, nx, ny))
        elif J == ny - 1:
            neighbs.append(get_bottom_face(I, 0, nx, ny))
    return neighbs


def make_mask(mesh, maze):
    mask = np.zeros([mesh.numberOfFaces], dtype=np.bool)
    for I in range(maze.a.shape[0]):
        for J in range(maze.a.shape[1]):
            if maze.a[I, J]:
                for j in get_neighbs(I, J, *maze.a.shape, periodic=True):
                    mask[j] = True
    return mask
