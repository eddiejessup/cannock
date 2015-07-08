from __future__ import print_function, division
import pickle
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import fipy
from fipy.terms import (TransientTerm, DiffusionTerm, ImplicitSourceTerm,
                        ConvectionTerm)
from fipy.meshes.periodicGrid1D import PeriodicGrid1D
from fipy.meshes.periodicGrid2D import PeriodicGrid2D


class ModelAbstract(object):
    def __init__(self, dim, dt, dx, L,
                 D_rho, mu):
        self.dim = dim
        self.dt = dt
        self.dx = dx
        self.L = L
        self.D_rho = D_rho
        self.mu = mu

        self.i = 0
        self.t = 0.0

        nx = int(round((self.L / self.dx)))
        self.dx = L / nx
        if self.dim == 1:
            self.mesh = PeriodicGrid1D(dx=np.array([self.dx]), nx=nx)
        elif self.dim == 2:
            self.mesh = PeriodicGrid2D(dx=np.array([self.dx]),
                                       dy=np.array([self.dx]), nx=nx, ny=nx)

        self.L_perturb = self.L
        eps = 0.001
        x = self.get_x()
        X, Y = np.meshgrid(x, x)
        perturb = eps * np.sin(2.0 * np.pi * X * Y / self.L_perturb ** 2)
        rho_init = 1.0 + perturb.ravel()
        c_init = 1.0 + perturb.ravel()
        self.rho = fipy.CellVariable(mesh=self.mesh, value=rho_init)
        self.c = fipy.CellVariable(mesh=self.mesh, value=c_init)

        eq_rho = (TransientTerm(var=self.rho) ==
                  DiffusionTerm(coeff=self.D_rho, var=self.rho) -
                  ConvectionTerm(coeff=mu * self.c.grad, var=self.rho))

        eq_c = (TransientTerm(var=self.c) ==
                DiffusionTerm(coeff=1.0, var=self.c) +
                self.rho -
                ImplicitSourceTerm(coeff=1.0, var=self.c))

        self.eq = eq_rho & eq_c

    def iterate(self):
        self.eq.solve(dt=self.dt)

        self.t += self.dt
        self.i += 1

    def get_x(self):
        return np.unique(self.mesh.cellCenters.value[0, :]) - self.L / 2.0

    def get_rho(self):
        return self.rho.value.reshape(self.rho.mesh.shape)

    def get_c(self):
        return self.c.value.reshape(self.c.mesh.shape)

dt = 0.01
t_max = 100.0

D_rho = 0.1
mu = 10.9


def main_1d():
    dx = 0.005
    L = 25.0

    m = ModelAbstract(1, dt, dx, L, D_rho, mu)

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


def main_2d():
    dx = 0.05
    L = 1.0
    m = ModelAbstract(2, dt, dx, L, D_rho, mu)

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


def param_sweep_1d():
    dx = 0.005
    L = 25.0
    dim = 1

    D_rhos = np.logspace(-2, 2, 10)
    mus = np.logspace(-2, 2, 10)
    for D_rho, mu in product(D_rhos, mus):
        m = ModelAbstract(dim, dt, dx, L, D_rho, mu)
        while m.t < t_max:
            m.iterate()

        rho_final = m.get_rho()
        print(D_rho, mu, rho_final.max(), np.var(rho_final))
        fname = 'run/abstract_dim={},D_rho={},mu={}.pkl'.format(dim, D_rho, mu)
        with open(fname, 'wb') as f:
            pickle.dump(m, f)


def param_sweep_2d():
    dx = 0.5
    L = 25.0
    dim = 2
    t_max = 1.0

    D_rhos = np.logspace(-2, 2, 4)
    mus = np.logspace(-2, 2, 4)
    for D_rho, mu in product(D_rhos, mus):
        m = ModelAbstract(dim, dt, dx, L, D_rho, mu)
        while m.t < t_max:
            m.iterate()
            # print(m.t)

        rho_final = m.get_rho()
        print(D_rho, mu, rho_final.max(), np.var(rho_final))
        fname = 'run/abstract_dim={},D_rho={},mu={}.pkl'.format(dim, D_rho, mu)
        with open(fname, 'wb') as f:
            pickle.dump(m, f)

if __name__ == '__main__':
    # param_sweep_2d()
    main_2d()
