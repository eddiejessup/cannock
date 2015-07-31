from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import fipy
from fipy.meshes.periodicGrid2D import PeriodicGrid2D
import bannock.walls
from cannock.utils import make_mask

L = 1.0
dx = 0.02
dt = 0.01
D = 1.0
n = 1000

nx = int(round(L / dx))
mesh = PeriodicGrid2D(dx=np.array([dx]), dy=np.array([dx]), nx=nx, ny=nx)

phi = fipy.CellVariable(mesh=mesh, value=0.0)

maze = bannock.walls.Maze(L, dim=2, dx=dx, d=dx, seed=None)

maze_ravel = maze.a.T.ravel()
phi0 = np.random.uniform(0.0, 1.0, size=maze_ravel.shape)
phi0 *= np.logical_not(maze_ravel)
print(phi0)
phi.setValue(phi0)

mask = make_mask(mesh, maze)
mask_D = fipy.FaceVariable(mesh=mesh, value=D)
mask_D.setValue(0.0, where=mask)

eq = fipy.TransientTerm() == fipy.DiffusionTerm(coeff=mask_D)

viewer = fipy.Matplotlib2DViewer(vars=phi, colorbar=None)
viewer.plotMesh()

for step in range(n):
    eq.solve(var=phi, dt=dt)
    viewer.plot()
    print(step)
raw_input()
