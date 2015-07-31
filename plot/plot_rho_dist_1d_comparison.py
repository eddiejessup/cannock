from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
from ciabatta import ejm_rcparams
from ciabatta.ejm_rcparams import set2
from bannock import utils as bu
from cannock.utils.utils import get_rho_agent, find_nearest_index
from cannock.model import ModelCoarse1D


agent_dirname = '/Users/ewj/Desktop/cannock/agent_data/autochemo_model_dim=1,seed=1,dt=0.1,L=5e+03,dx=40,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,origin_flag=1,rho_0=1,chi=1.5,onesided_flag=1,vicsek_R=0'

save_flag = True

use_latex = save_flag
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

fig = plt.figure(figsize=(12, 12 * ejm_rcparams.golden_ratio))
ax = fig.add_subplot(111)
ejm_rcparams.prettify_axes(ax)

ax = fig.add_subplot(111)

agent_fnames = bu.get_filenames(agent_dirname)
m_agent_0 = bu.filename_to_model(agent_fnames[0])

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

dt = 10.0
dx = 5.0

m_coarse = ModelCoarse1D(dt,
                         rho_0, rho_D, origin_flag, dx_agent,
                         mu,
                         L, dx,
                         c_D, c_sink, c_source)

for _ in range(200):
    m_coarse.iterate()
rho_coarse = m_coarse.get_rho()

agent_fnames = bu.get_filenames(agent_dirname)
ts = np.array([bu.filename_to_model(f).t for f in agent_fnames])
i_agent_fname = find_nearest_index(m_coarse.t, ts)
m_agent = bu.filename_to_model(agent_fnames[i_agent_fname])
rho_agent = get_rho_agent(m_agent, m_coarse)

ms = [bu.filename_to_model(fname) for fname in bu.get_filenames(agent_dirname)]
rho_agent = np.mean([get_rho_agent(m, m_coarse)
                     for m in ms if m.t > m_coarse.t], axis=0)
x_tild = m_coarse.get_x() / np.sqrt(rho_D / c_sink)
ax.plot(x_tild, 1.0 * rho_coarse / rho_0, c=set2[0], label='Coarse')
ax.plot(x_tild + 0.1, rho_agent / rho_0, c=set2[1], label='Agent')

ax.legend(loc='lower right', fontsize=26, frameon=True)
ax.set_xlabel(r'$\tilde{x}$', fontsize=35)
ax.set_ylabel(r'$\rho / \rho_0$', fontsize=35)
ax.tick_params(axis='both', labelsize=26, pad=10.0)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(0.0, 65.0)

if save_flag:
    plt.savefig('rho_dist_1d_comparison.pdf', bbox_inches='tight')
else:
    plt.show()
