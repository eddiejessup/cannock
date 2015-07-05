from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt


def laplace(f, dx):
    laplace_f = np.zeros_like(f)
    laplace_f[1:-1] = f[2:] + f[:-2] - 2.0 * f[1:-1]
    laplace_f[0] = f[1] + f[-1] - 2.0 * f[0]
    laplace_f[-1] = f[0] + f[-2] - 2.0 * f[-1]
    return laplace_f / dx ** 2


def grad(f, dx):
    grad_f = np.zeros_like(f)
    grad_f[1:-1] = f[2:] - f[:-2]
    grad_f[0] = f[1] - f[-1]
    grad_f[-1] = f[0] - f[-2]
    return grad_f / (2.0 * dx)


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

        m = self.L / self.dx
        self.x = np.linspace(-self.L / 2.0, self.L / 2.0, m)
        self.dx = self.x[1] - self.x[0]

        self.c = np.zeros_like(self.x)

        if origin_flag:
            n = np.exp(-np.abs(self.x) ** 2 / (2.0 * self.dx_agent ** 2))
            n /= (n.sum() * self.dx)
            n *= self.rho_0 * self.L * self.dx
            self.rho = n / self.dx
        else:
            self.rho = self.rho_0 * (np.ones_like(self.c) *
                                     np.random.uniform(0.99, 1.01,
                                                       size=self.c.shape))

    def grad(self, f):
        return grad(f, self.dx)

    div = grad

    def laplace(self, f):
        return laplace(f, self.dx)

    def iterate_rho(self):
        self.rho += self.dt * (self.rho_D * self.laplace(self.rho) -
                               (self.div(self.mu * self.grad(self.c) *
                                self.rho)))

    def iterate_c(self):
        self.c += self.dt * (self.c_D * self.laplace(self.c) +
                             self.c_source * self.rho - self.c_sink * self.c)

    def iterate(self):
        self.iterate_rho()
        self.iterate_c()

        self.t += self.dt
        self.i += 1


def main():
    mu = 10.0
    rho_D = 400.0
    L = 5000.0
    c_D = 1000.0
    c_sink = 0.01
    c_source = 1.0
    rho_0 = 0.1
    origin_flag = True
    dx_agent = 40.0

    dt = 0.01
    m = 500

    dx = 20.0

    m = ModelCoarse1D(dt,
                      rho_0, rho_D, origin_flag, dx_agent,
                      mu,
                      L, dx,
                      c_D, c_sink, c_source)

    # fig = plt.figure()
    # ax = fig.gca()
    # ax.set_ylim(0.0, m.rho.max())
    # plot_c = ax.scatter(m.x, m.c, c='yellow')
    # plot_rho = ax.scatter(m.x, m.rho, c='red')
    # plt.ion()
    # plt.show()

    every = 1000

    while m.t < 200.0:
        # if not m.i % every:
        #     plot_rho.set_offsets(np.array([m.x, m.rho]).T)
        #     plot_c.set_offsets(np.array([m.x, m.c]).T)
        #     fig.canvas.draw()
        #     print(m.t, np.mean(m.rho))
        m.iterate()

if __name__ == '__main__':
    # main()
    import cProfile
    cProfile.run('main()')
