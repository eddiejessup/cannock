from __future__ import print_function, division
import numpy as np
from fipy import *

dt = 0.001
n = 100
large_value = 1e10

mesh = Grid1D(nx=200, dx=1.0)

phi = CellVariable(mesh=mesh, value=0.0)

mask = np.logical_and(mesh.faceCenters > 31.0, mesh.faceCenters < 50.0)
mask_int = np.logical_and(mesh.x > 31.0, mesh.x < 50.0)

print(mask)
phi.faceGrad.constrain((0.0,), where=mask)
phi.constrain(2.0, where=mesh.facesLeft)
phi.constrain(1.0, where=mesh.facesRight)

eq = DiffusionTerm() - ImplicitSourceTerm(large_value * mask_int)

viewer = Matplotlib1DViewer(vars=phi, colorbar=None)
viewer.plotMesh()

for step in range(n):
    eq.solve(var=phi, dt=dt)
    viewer.plot()
    raw_input()
