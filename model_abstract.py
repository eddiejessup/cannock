from __future__ import print_function, division
import pickle
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import fipy
from fipy.terms import (TransientTerm, DiffusionTerm, ImplicitSourceTerm,
                        ConvectionTerm)
from fipy.meshes.periodicGrid1D import PeriodicGrid1D


class ModelAbstract1D(object):
    def __init__(self, dt, dx, L,
                 D_rho, mu):
        self.dt = dt
        self.dx = dx
        self.L = L
        self.D_rho = D_rho
        self.mu = mu

        self.i = 0
        self.t = 0.0

        nx = int(round((self.L / self.dx)))
        self.dx = L / nx
        self.mesh = PeriodicGrid1D(dx=np.array([self.dx]), nx=nx)

        self.L_perturb = self.L
        eps = 0.001
        perturb = eps * np.sin(2.0 * np.pi * self.get_x() / self.L_perturb)
        rho_init = 1.0 + perturb
        c_init = 1.0 + perturb
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
        return self.mesh.cellCenters.value[0] - self.L / 2.0

    def get_rho(self):
        return self.rho.value

    def get_c(self):
        return self.c.value

dt = 0.5
dx = 0.005
L = 25.0

D_rho = 0.9
mu = 1.0

t_max = 100.0


def main():
    m = ModelAbstract1D(dt, dx, L, D_rho, mu)

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


def param_sweep():
    D_rhos = np.logspace(-2, 2, 10)
    mus = np.logspace(-2, 2, 10)
    for D_rho, mu in product(D_rhos, mus):
        m = ModelAbstract1D(dt, dx, L, D_rho, mu)
        while m.t < t_max:
            m.iterate()

        rho_final = m.get_rho()
        print(D_rho, mu, rho_final.max(), np.var(rho_final))
        fname = 'run/abstract_D_rho={},mu={}.pkl'.format(D_rho, mu)
        with open(fname, 'wb') as f:
            pickle.dump(m, f)

if __name__ == '__main__':
    param_sweep()
    # main()
