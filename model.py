from __future__ import print_function, division
import numpy as np
import fipy
from fipy.terms import (TransientTerm, DiffusionTerm, ImplicitSourceTerm,
                        ConvectionTerm)
from fipy.meshes.periodicGrid1D import PeriodicGrid1D
from fipy.meshes.periodicGrid2D import PeriodicGrid2D
from cannock.utils import make_mask


class ModelCoarse1D(object):
    def __init__(self, dt,
                 rho_0, rho_D, origin_flag, dx_agent,
                 mu,
                 L, dx,
                 c_D, c_sink, c_source):
        self.dim = 1
        self.dt = dt
        self.rho_0 = rho_0
        self.rho_D = rho_D
        self.origin_flag = origin_flag
        self.dx_agent = dx_agent
        self.mu = mu
        self.L = L
        self.dx = dx
        self.c_D = c_D
        self.c_sink = c_sink
        self.c_source = c_source

        self.i = 0
        self.t = 0.0

        nx = int(round((self.L / self.dx)))
        self.dx = L / nx
        self.mesh = PeriodicGrid1D(dx=np.array([self.dx]), nx=nx)

        if origin_flag:
            # n = np.exp(-np.abs(self.get_x()) ** 2 / (2.0 * self.dx_agent ** 2))
            # n /= (n.sum() * self.dx)
            # n *= self.rho_0 * self.L * self.dx
            # rho_val = n / self.dx
            rho_val = np.zeros_like(self.get_x())
            rho_val[len(rho_val) // 2] = 1.12 * (self.rho_0 * self.L) / self.dx
        else:
            rho_val = self.rho_0 * (np.ones_like(self.c) *
                                    np.random.uniform(0.99, 1.01,
                                                      size=self.c.shape))

        self.rho = fipy.CellVariable(mesh=self.mesh, value=rho_val)
        self.c = fipy.CellVariable(mesh=self.mesh, value=0.0)

        self.eq_rho = (TransientTerm() ==
                       DiffusionTerm(coeff=self.rho_D) -
                       ConvectionTerm(coeff=self.mu * self.c.grad))

        self.eq_c = (TransientTerm() ==
                     DiffusionTerm(coeff=self.c_D) +
                     self.c_source * self.rho -
                     ImplicitSourceTerm(coeff=self.c_sink))

    def _iterate_rho(self):
        self.eq_rho.solve(var=self.rho, dt=self.dt)

    def _iterate_c(self):
        self.eq_c.solve(var=self.c, dt=self.dt)

    def iterate(self):
        self._iterate_rho()
        self._iterate_c()

        self.t += self.dt
        self.i += 1

    def get_x(self):
        return self.mesh.cellCenters.value[0] - self.L / 2.0

    def get_rho(self):
        return self.rho.value

    def get_c(self):
        return self.c.value


class ModelAbstract(object):
    def __init__(self, dim, dt, dx, L,
                 D_rho, mu, walls):
        self.dim = dim
        self.dt = dt
        self.dx = dx
        self.L = L
        self.D_rho = D_rho
        self.mu = mu
        self.walls = walls

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
        eps = 1e-2
        x = self.get_x()
        if self.dim == 1:
            perturb = eps * np.sin(2.0 * np.pi * x / self.L_perturb)
        else:
            X, Y = np.meshgrid(x, x)
            perturb = eps * np.sin(2.0 * np.pi * X * Y / self.L_perturb ** 2)
        rho_init = 1.0 + perturb
        c_init = 1.0 + perturb

        if self.walls is not None:
            rho_init *= np.logical_not(self.walls.a)
            c_init *= np.logical_not(self.walls.a)

            mask = make_mask(self.mesh, self.walls)
            self.D_rho_var = fipy.FaceVariable(mesh=self.mesh,
                                               value=self.D_rho)
            self.D_rho_var.setValue(0.0, where=mask)
            self.D_c_var = fipy.FaceVariable(mesh=self.mesh, value=1.0)
            self.D_c_var.setValue(0.0, where=mask)
        else:
            self.D_rho_var = self.D_rho
            self.D_c_var = 1.0

        self.rho = fipy.CellVariable(mesh=self.mesh, value=rho_init.T.ravel())
        self.c = fipy.CellVariable(mesh=self.mesh, value=c_init.T.ravel())

        eq_rho = (TransientTerm(var=self.rho) ==
                  DiffusionTerm(coeff=self.D_rho_var, var=self.rho) -
                  ConvectionTerm(coeff=mu * self.c.grad, var=self.rho))

        eq_c = (TransientTerm(var=self.c) ==
                DiffusionTerm(coeff=self.D_c_var, var=self.c) +
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
        if self.dim == 2:
            return self.rho.value.reshape(self.rho.mesh.shape)
        else:
            return self.rho.value

    def get_c(self):
        if self.dim == 2:
            return self.c.value.reshape(self.rho.mesh.shape)
        else:
            return self.c.value
