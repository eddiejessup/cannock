from __future__ import print_function, division
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import bannock.utils.utils as bu
from cannock.model import ModelCoarse1D, ModelAbstract
from cannock.utils.utils import find_nearest_index, get_rho_agent


def run_param_sweep_2d():
    dx = 0.2
    L = 5.0
    dim = 2
    t_max = 40.0
    dt = 0.1

    D_rhos = np.logspace(-1, 2, 10)
    mus = np.logspace(-1, 2, 10)
    for D_rho, mu in product(D_rhos, mus):
        if D_rho != mu:
            m = ModelAbstract(dim, dt, dx, L, D_rho, mu, walls=None)
            while m.t < t_max:
                m.iterate()
            rho_final = m.get_rho()
            print(D_rho, mu, rho_final.max(), np.var(rho_final))


def run_model_abstract_1d():
    dx = 0.4
    L = 5.0
    dim = 1
    t_max = 160.0
    dt = 2.0

    D_rho = 0.01
    mu = 0.215

    m = ModelAbstract(dim, dt, dx, L, D_rho, mu, walls=None)

    print(m.get_x().shape, m.get_c().shape)
    raw_input()
    fig = plt.figure()
    ax_rho = fig.add_subplot(2, 1, 1)
    ax_c = fig.add_subplot(2, 1, 2)
    plot_rho = ax_rho.plot(m.get_x(), m.get_rho(), c='red')[0]
    plot_c = ax_c.plot(m.get_x(), m.get_c(), c='green')[0]
    plt.ion()
    plt.show()
    every = 1

    while m.t < t_max:
        if not m.i % every:
            plot_rho.set_ydata(m.get_rho())
            plot_c.set_ydata(m.get_c())
            rho = m.get_rho()
            rho_ran = rho.max() - rho.min()
            rho_max = rho.max() + 0.1 * rho_ran
            rho_min = rho.min() - 0.1 * rho_ran
            ax_rho.set_ylim(rho_min, rho_max)
            c = m.get_c()
            c_ran = c.max() - c.min()
            c_max = c.max() + 0.1 * c_ran
            c_min = c.min() - 0.1 * c_ran
            ax_c.set_ylim(c_min, c_max)
            fig.canvas.draw()
            print(m.t, np.var(rho))
        m.iterate()


def run_model_abstract_2d():
    dx = 0.2
    L = 5.0
    dim = 2
    t_max = 40.0
    dt = 0.1

    D_rho = 22.
    mu = 44.

    # walls = bannock.walls.Maze(L, dim=2, dx=dx, d=2 * dx, seed=None)
    # walls = bannock.walls.Traps(L, dx=dx, n=1, d=dx, w=10*dx, s=2*dx)
    walls = None

    m = ModelAbstract(dim, dt, dx, L, D_rho, mu, walls)

    fig = plt.figure()
    ax_rho = fig.add_subplot(2, 1, 1)
    ax_c = fig.add_subplot(2, 1, 2)
    plot_rho = ax_rho.imshow([[0]], cmap='Reds', interpolation='nearest',
                             origin='lower', extent=2 * [-L / 2.0, L / 2.0])
    plot_c = ax_c.imshow([[0]], cmap='Reds', interpolation='nearest',
                         origin='lower', extent=2 * [-L / 2.0, L / 2.0])

    ax_cb = plt.axes([0.875, 0.2, 0.05, 0.7])
    fig.colorbar(plot_rho, cax=ax_cb)

    plt.ion()
    plt.show()
    every = 1

    while m.t < t_max:
        if not m.i % every:
            plot_rho.set_data(m.get_rho())
            plot_c.set_data(m.get_c())
            plot_rho.autoscale()
            plot_c.autoscale()
            fig.canvas.draw_idle()
            print(m.t, np.var(m.get_rho()))
        m.iterate()


def run_model(dt,
              rho_0, rho_D, origin_flag, dx_agent,
              mu,
              L, dx,
              c_D, c_sink, c_source,
              t_max, every):
    rho_D = 400.0
    mu = 100.0
    L = 5000.0
    c_D = 1000.0
    c_sink = 0.01
    c_source = 1.0
    rho_0 = 1.0
    origin_flag = True
    dx_agent = 40.0
    dt = 1.0
    m = 500
    dx = 10.0

    m = ModelCoarse1D(dt,
                      rho_0, rho_D, origin_flag, dx_agent,
                      mu,
                      L, dx,
                      c_D, c_sink, c_source)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_ylim(0.0, 1000.0)
    plot_c = ax.scatter(m.get_x(), m.get_c(), c='yellow')
    plot_rho = ax.plot(m.get_x(), m.get_rho(), c='red')[0]
    # plot_rho = ax.scatter(m.get_x(), m.get_rho(), c='red')
    # plot_arho = ax.scatter(x_agent, m_agent_0.get_density_field(), c='green')
    plt.ion()
    plt.show()

    while m.t < t_max:
        if not m.i % every:
            # plot_rho.set_offsets(np.array([m.get_x(), m.get_rho()]).T)
            plot_rho.set_ydata(m.get_rho())
            # plot_c.set_offsets(np.array([m.get_x(), m.get_c()]).T)
            plot_c.set_ydata(m.get_c())
            fig.canvas.draw()
            print(m.t, np.mean(m.get_rho()))
        m.iterate()
    return m


def match_model_to_agent(agent_dirname, dt, dx):
    agent_fnames = bu.get_filenames(agent_dirname)
    m_agent_0 = bu.filename_to_model(agent_fnames[0])
    # x_agent = np.linspace(-m_agent_0.L / 2.0, m_agent_0.L / 2.0,
    #                       m_agent_0.c.a.shape[0])
    ts = np.array([bu.filename_to_model(f).t for f in agent_fnames])

    rho_0 = m_agent_0.rho_0
    origin_flag = m_agent_0.origin_flag
    dx_agent = m_agent_0.dx
    mu = 0.5 * m_agent_0.chi * m_agent_0.v_0
    if m_agent_0.onesided_flag:
        mu /= 2.0
    rho_D = m_agent_0.v_0 ** 2 / m_agent_0.p_0
    L = m_agent_0.L
    c_D = m_agent_0.c_D
    c_sink = m_agent_0.c_sink
    c_sink = m_agent_0.c_sink
    c_source = m_agent_0.c_source

    m = ModelCoarse1D(dt,
                      rho_0, rho_D, origin_flag, dx_agent,
                      mu,
                      L, dx,
                      c_D, c_sink, c_source)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(-500.0, 500.0)
    ax.set_ylim(0.0, 1000.0)
    plot_rho = ax.plot(m.get_x(), m.get_rho(), c='red')[0]
    plot_arho = ax.plot(m.get_x(), get_rho_agent(m_agent_0, m), c='green')[0]
    plt.ion()
    plt.show()

    every = 10

    while m.t < 20000.0:
        if not m.i % every:
            rho_coarse = m.get_rho()
            plot_rho.set_ydata(rho_coarse)

            i_agent_fname = find_nearest_index(m.t, ts)
            m_agent = bu.filename_to_model(agent_fnames[i_agent_fname])
            rho_agent = get_rho_agent(m_agent, m)
            plot_arho.set_ydata(rho_agent)
            ax.set_ylim(0.0, 1.1 * max(rho_coarse.max(), rho_agent.max()))
            fig.canvas.draw()
            print(m.t, np.mean(m.get_rho()), agent_fnames[i_agent_fname])
        m.iterate()
    return m
