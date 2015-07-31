from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
from ciabatta import ejm_rcparams
from ciabatta.ejm_rcparams import set2

save_flag = True

use_latex = save_flag
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

fig = plt.figure(figsize=(12, 12 * ejm_rcparams.golden_ratio))
ax = fig.add_subplot(111)
ejm_rcparams.prettify_axes(ax)

D_rho, mu, m, v = np.loadtxt('data_2d_new.txt', unpack=True)
unstable = v > 1e-0

ax = fig.add_subplot(111)
D_rhos = np.linspace(0.1 * D_rho.min(), 10.0 * D_rho.max(), 1000)
stable_mus = 4 * D_rhos
qstable_mus = D_rhos
ax.scatter(D_rho[np.logical_not(unstable)], mu[np.logical_not(unstable)],
           c=set2[1], marker='o', label='Stable', s=40)
ax.scatter(D_rho[unstable], mu[unstable],
           c=set2[0], marker='s', label='Unstable', s=40)
ax.plot(D_rhos, stable_mus,
        c=set2[2], label=r'$\tilde{\mu} = 4 \tilde{\mathrm{D}}_\rho$', lw=5)
ax.plot(D_rhos, qstable_mus,
        c=set2[3], label=r'$\tilde{\mu} = \tilde{\mathrm{D}}_\rho$', lw=5,
        ls='dashed')

handles, labels = ax.get_legend_handles_labels()
handles = handles[2], handles[3], handles[0], handles[1]
labels = labels[2], labels[3], labels[0], labels[1]
ax.legend(handles, labels, loc='lower right', fontsize=26, frameon=True)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\tilde{\mathrm{D}}_\rho$', fontsize=35)
ax.set_ylabel(r'$\tilde{\mu}$', fontsize=35)
ax.tick_params(axis='both', labelsize=26, pad=10.0)
ax.set_xlim(0.9 * D_rho.min(), 1.1 * D_rho.max())
ax.set_ylim(0.9 * mu.min(), 1.1 * mu.max())

if save_flag:
    plt.savefig('plots/sweep_D_mu_2d.pdf', bbox_inches='tight')
else:
    plt.show()
