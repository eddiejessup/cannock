from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import fipy
from fipy.terms import (TransientTerm, DiffusionTerm, ImplicitSourceTerm,
                        ConvectionTerm)
from fipy.meshes.periodicGrid1D import PeriodicGrid1D
import bannock.utils as bu


def find_nearest_index(v, a):
    return np.argmin(np.abs(a - v))


def get_rho_agent(m_agent, m_coarse):
    ns, bins = np.histogram(m_agent.r[:, 0], bins=m_coarse.get_x().shape[0],
                            range=[-m_agent.L / 2.0, m_agent.L / 2.0])
    rho_agent = ns / m_coarse.dx
    return rho_agent


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
            n = np.exp(-np.abs(self.get_x()) ** 2 / (2.0 * self.dx_agent ** 2))
            n /= (n.sum() * self.dx)
            n *= self.rho_0 * self.L * self.dx
            rho_val = n / self.dx
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


def run_model(dt,
              rho_0, rho_D, origin_flag, dx_agent,
              mu,
              L, dx,
              c_D, c_sink, c_source,
              t_max, every):
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


def match_to_agent(agent_dirname, dt, dx):
    agent_fnames = bu.get_filenames(agent_dirname)
    m_agent_0 = bu.filename_to_model(agent_fnames[0])
    # x_agent = np.linspace(-m_agent_0.L / 2.0, m_agent_0.L / 2.0,
    #                       m_agent_0.c.a.shape[0])
    ts = np.array([bu.filename_to_model(f).t for f in agent_fnames])

    rho_0 = m_agent_0.rho_0
    origin_flag = m_agent_0.origin_flag
    dx_agent = m_agent_0.dx
    mu = m_agent_0.chi * m_agent_0.v_0
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
            print(m.t, np.mean(m.get_rho()))
        m.iterate()
    return m

if __name__ == '__main__':
    # agent_dirname = '/Users/ewj/Desktop/cannock/autochemo_model_dim=1,seed=1,dt=0.1,L=5e+03,dx=40,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,origin_flag=1,rho_0=0.1,chi=7,onesided_flag=0,vicsek_R=0'
    # agent_dirname = '/Users/ewj/Desktop/cannock/autochemo_model_dim=1,seed=1,dt=0.1,L=5e+03,dx=40,c_D=1e+03,c_sink=0.01,c_source=1,v_0=10,p_0=1,origin_flag=1,rho_0=1,chi=0.7,onesided_flag=0,vicsek_R=0'
    # agent_dirname = '/Users/ewj/Desktop/cannock/autochemo_model_dim=1,seed=1,dt=0.1,L=5e+03,dx=40,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,origin_flag=1,rho_0=10,chi=0.7,onesided_flag=0,vicsek_R=0'
    # agent_dirname = '/Users/ewj/Desktop/cannock/autochemo_model_dim=1,seed=1,dt=0.1,L=5e+03,dx=40,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,origin_flag=1,rho_0=1,chi=0.7,onesided_flag=0,vicsek_R=0'
    # agent_dirname = '/Users/ewj/Desktop/cannock/autochemo_model_dim=1,seed=1,dt=0.1,L=5e+03,dx=40,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=2,origin_flag=1,rho_0=1,chi=0.7,onesided_flag=0,vicsek_R=0'
    # agent_dirname = '/Users/ewj/Desktop/cannock/autochemo_model_dim=1,seed=1,dt=0.1,L=5e+03,dx=40,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=0.5,origin_flag=1,rho_0=1,chi=0.7,onesided_flag=0,vicsek_R=0'
    # agent_dirname = '/Users/ewj/Desktop/cannock/autochemo_model_dim=1,seed=1,dt=0.1,L=5e+03,dx=40,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,origin_flag=1,rho_0=1,chi=0.7,onesided_flag=1,vicsek_R=0'
    agent_dirname = '/Users/ewj/Desktop/cannock/autochemo_model_dim=1,seed=1,dt=0.1,L=5e+03,dx=40,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=0.5,origin_flag=1,rho_0=1,chi=0.7,onesided_flag=1,vicsek_R=0'
    match_to_agent(agent_dirname, dt=10.0, dx=5.0)
    # import cProfile
    # cProfile.run('main()', sort='cumtime')
